"""The bulk loader against a fake ingest api: what it sends, how it retries, when it says done."""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from eval.dataset import (
    JudgementRow,
    ManifestMismatch,
    Product,
    QueryRow,
    Selection,
    SelectionConfig,
    write_dataset,
)
from eval.loader import LoadConfig, LoaderNotReady, LoadReport, load_dataset, product_to_request

REVISION = 4


def products(count: int, title: str = "title") -> list[Product]:
    return [Product(f"P{n}", "us", f"{title} {n}", None, None, None, None) for n in range(count)]


def dataset(directory: Path, items: list[Product]) -> None:
    selection = Selection(
        queries=(QueryRow(1, "shoes", "test"),),
        judgements=tuple(JudgementRow(1, item.product_id, "E") for item in items),
        product_ids=tuple(sorted(item.product_id for item in items)),
    )
    write_dataset(directory, selection, items, SelectionConfig(seed=1), REVISION)


class FakeIngest:
    """Accepts posts, and reports a document indexed after a number of status polls."""

    def __init__(self, polls_until_indexed: int = 1) -> None:
        self.polls_until_indexed = polls_until_indexed
        self.post_calls: list[str] = []
        self.posted: dict[str, dict[str, Any]] = {}
        # doc_id -> how many status answers named it since it was posted
        self.polls: dict[str, int] = {}
        # one entry per status request: how many ids it asked about
        self.status_requests: list[int] = []
        # answers handed out first, one per post, before the normal behaviour
        self.script: list[httpx.Response] = []
        # doc_id -> status code the post is refused with
        self.refuse: dict[str, int] = {}
        # these report an indexed_version one behind the version that was sent
        self.never_indexed: set[str] = set()
        # url -> status code of a readiness url, None means the connection is refused
        self.ready: dict[str, int | None] = {}
        self.ready_calls: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            # a pause so requests really overlap
            await asyncio.sleep(0.001)
            if request.method == "GET":
                return self._ready(request)
            if request.url.path == "/v1/document-statuses":
                return self._status_batch(request)
            return self._post(request)
        finally:
            self.in_flight -= 1

    def _ready(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.ready_calls.append(url)
        code = self.ready.get(url, 200)
        if code is None:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(code, json={"ready": code == 200})

    def _post(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.post_calls.append(body["doc_id"])
        if self.script:
            return self.script.pop(0)
        if body["doc_id"] in self.refuse:
            return httpx.Response(self.refuse[body["doc_id"]], json={"error": "content_conflict"})
        self.posted[body["doc_id"]] = body
        return httpx.Response(202, json={"doc_id": body["doc_id"]})

    def _status_batch(self, request: httpx.Request) -> httpx.Response:
        asked = json.loads(request.content)["doc_ids"]
        self.status_requests.append(len(asked))
        statuses = []
        for doc_id in asked:
            if doc_id not in self.posted:
                continue
            self.polls[doc_id] = self.polls.get(doc_id, 0) + 1
            version = self.posted[doc_id]["version"]
            if doc_id in self.never_indexed:
                indexed = version - 1
            else:
                indexed = version if self.polls[doc_id] >= self.polls_until_indexed else None
            statuses.append(
                {
                    "doc_id": doc_id,
                    "latest_version": version,
                    "current_version": indexed,
                    "indexed_version": indexed,
                    "deleted": False,
                }
            )
        return httpx.Response(200, json={"request_id": "r", "statuses": statuses})


class FakeTime:
    """A clock that only moves when the loader sleeps, so no test waits for real time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def clock(self) -> float:
        return self.now


async def run_load(
    directory: Path, ingest: FakeIngest, fake_time: FakeTime, **config: Any
) -> LoadReport:
    transport = httpx.MockTransport(ingest.handle)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
        return await load_dataset(
            client, directory, LoadConfig(**config), sleep=fake_time.sleep, clock=fake_time.clock
        )


# the body of one document


def test_a_product_becomes_an_upsert_body_with_the_dataset_version() -> None:
    product = Product(
        "P1", "us", "Red shoe", "A shoe.", "Light\nStrong sole\n\n  Cheap  ", "Acme", "red"
    )
    assert product_to_request(product, 4) == {
        "corpus": "esci",
        "doc_id": "P1",
        "version": 4,
        "title": "Red shoe",
        "locale": "us",
        "brand": "Acme",
        "color": "red",
        "description": "A shoe.",
        "bullets": ["Light", "Strong sole", "Cheap"],
    }


def test_missing_and_blank_fields_are_left_out() -> None:
    product = Product("P1", "us", "Red shoe", "  ", "\n \n", None, "")
    assert product_to_request(product, 4) == {
        "corpus": "esci",
        "doc_id": "P1",
        "version": 4,
        "title": "Red shoe",
        "locale": "us",
    }


def test_a_product_the_api_would_refuse_is_named_in_the_error() -> None:
    with pytest.raises(ValueError, match=r"P1.*title"):
        product_to_request(Product("P1", "us", "", None, None, None, None), 4)


# the whole load


async def test_every_product_is_sent_once_and_the_marker_says_what_was_loaded(
    tmp_path: Path,
) -> None:
    dataset(tmp_path, products(5))
    ingest = FakeIngest()
    report = await run_load(tmp_path, ingest, FakeTime())
    assert report == LoadReport(documents=5, failures={})
    assert sorted(ingest.post_calls) == [f"P{n}" for n in range(5)]
    assert {body["version"] for body in ingest.posted.values()} == {REVISION}
    marker = json.loads((tmp_path / "loaded.json").read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest()
    assert (marker["documents"], marker["dataset_revision"]) == (5, REVISION)
    assert marker["manifest_sha256"] == manifest_hash


async def test_a_document_is_asked_about_until_its_indexed_version_reaches_the_sent_one(
    tmp_path: Path,
) -> None:
    dataset(tmp_path, products(1))
    ingest = FakeIngest(polls_until_indexed=3)
    fake_time = FakeTime()
    report = await run_load(tmp_path, ingest, fake_time, poll_interval_s=2.0)
    assert report.failures == {}
    assert ingest.polls == {"P0": 3}
    assert fake_time.sleeps == [2.0, 2.0]


async def test_status_is_asked_in_batches_not_per_document(tmp_path: Path) -> None:
    # 2100 documents: a batch holds at most 2000 ids, so every question takes two requests
    dataset(tmp_path, products(2100))
    ingest = FakeIngest()
    report = await run_load(tmp_path, ingest, FakeTime())
    assert report.failures == {}
    # two before sending (nothing is known yet), two in the one wait round
    assert ingest.status_requests == [2000, 100, 2000, 100]


async def test_documents_already_indexed_are_not_sent_again(tmp_path: Path) -> None:
    dataset(tmp_path, products(3))
    ingest = FakeIngest()
    assert (await run_load(tmp_path, ingest, FakeTime())).failures == {}
    assert len(ingest.post_calls) == 3
    assert (await run_load(tmp_path, ingest, FakeTime())).failures == {}
    # the second run found all three indexed and sent nothing, and wrote the marker again
    assert len(ingest.post_calls) == 3
    assert (tmp_path / "loaded.json").exists()


async def test_the_products_file_is_handled_in_chunks(tmp_path: Path) -> None:
    dataset(tmp_path, products(30))
    ingest = FakeIngest()
    report = await run_load(tmp_path, ingest, FakeTime(), chunk_size=7)
    assert report == LoadReport(documents=30, failures={})
    assert sorted(ingest.post_calls) == sorted(f"P{n}" for n in range(30))


async def test_no_more_requests_are_in_flight_than_the_limit(tmp_path: Path) -> None:
    dataset(tmp_path, products(30))
    ingest = FakeIngest()
    await run_load(tmp_path, ingest, FakeTime(), max_in_flight=4)
    assert 1 < ingest.max_in_flight <= 4


async def test_a_503_is_retried_after_the_time_the_api_asked_for(tmp_path: Path) -> None:
    dataset(tmp_path, products(1))
    ingest = FakeIngest()
    ingest.script = [
        httpx.Response(503, headers={"Retry-After": "3"}, json={"error": "overloaded"})
    ]
    fake_time = FakeTime()
    report = await run_load(tmp_path, ingest, fake_time)
    assert report.failures == {}
    assert fake_time.sleeps[0] == 3.0
    assert ingest.post_calls == ["P0", "P0"]


async def test_a_post_that_never_gets_a_202_gives_up_and_names_the_document(
    tmp_path: Path,
) -> None:
    dataset(tmp_path, products(1))
    ingest = FakeIngest()
    ingest.script = [httpx.Response(503, headers={"Retry-After": "1"}, json={})] * 3
    report = await run_load(tmp_path, ingest, FakeTime(), max_post_attempts=3)
    assert report.failures == {"P0": "no 202 after 3 attempts"}
    assert not (tmp_path / "loaded.json").exists()


async def test_a_content_conflict_is_a_failure_and_no_marker_is_written(tmp_path: Path) -> None:
    dataset(tmp_path, products(3))
    ingest = FakeIngest()
    ingest.refuse = {"P1": 409}
    report = await run_load(tmp_path, ingest, FakeTime())
    assert set(report.failures) == {"P1"}
    assert report.failures["P1"].startswith("409")
    assert ingest.post_calls.count("P1") == 1
    assert not (tmp_path / "loaded.json").exists()


async def test_a_product_that_is_never_indexed_fails_after_the_wait_timeout(
    tmp_path: Path,
) -> None:
    dataset(tmp_path, products(3))
    ingest = FakeIngest()
    ingest.never_indexed = {"P2"}
    fake_time = FakeTime()
    report = await run_load(tmp_path, ingest, fake_time, wait_timeout_s=10, poll_interval_s=2)
    assert set(report.failures) == {"P2"}
    assert report.failures["P2"].startswith("not indexed after 10")
    assert fake_time.now >= 10
    assert not (tmp_path / "loaded.json").exists()


async def test_an_old_marker_is_removed_when_the_new_load_fails(tmp_path: Path) -> None:
    dataset(tmp_path, products(2))
    (tmp_path / "loaded.json").write_text("{}", encoding="utf-8")
    ingest = FakeIngest()
    ingest.never_indexed = {"P1"}
    await run_load(tmp_path, ingest, FakeTime(), wait_timeout_s=4, poll_interval_s=2)
    assert not (tmp_path / "loaded.json").exists()


# before anything is sent


async def test_every_readiness_url_is_checked_before_the_first_post(tmp_path: Path) -> None:
    dataset(tmp_path, products(2))
    ingest = FakeIngest()
    urls = ("http://api/readyz", "http://worker/readyz", "http://relay/readyz")
    await run_load(tmp_path, ingest, FakeTime(), ready_urls=urls)
    assert tuple(ingest.ready_calls) == urls
    assert len(ingest.post_calls) == 2


async def test_the_load_does_not_start_when_a_readiness_url_fails(tmp_path: Path) -> None:
    dataset(tmp_path, products(2))
    ingest = FakeIngest()
    ingest.ready = {"http://worker/readyz": 503}
    with pytest.raises(LoaderNotReady, match="http://worker/readyz answered 503"):
        await run_load(tmp_path, ingest, FakeTime(), ready_urls=("http://worker/readyz",))
    assert ingest.post_calls == []
    assert ingest.status_requests == []


async def test_the_load_does_not_start_when_a_readiness_url_does_not_answer(
    tmp_path: Path,
) -> None:
    dataset(tmp_path, products(2))
    ingest = FakeIngest()
    ingest.ready = {"http://worker/readyz": None}
    with pytest.raises(LoaderNotReady, match="http://worker/readyz did not answer"):
        await run_load(tmp_path, ingest, FakeTime(), ready_urls=("http://worker/readyz",))
    assert ingest.post_calls == []
    assert ingest.status_requests == []


async def test_a_dataset_file_that_changed_stops_the_load_before_any_request(
    tmp_path: Path,
) -> None:
    dataset(tmp_path, products(2))
    (tmp_path / "products.jsonl").write_bytes(b"changed\n")
    ingest = FakeIngest()
    with pytest.raises(ManifestMismatch):
        await run_load(tmp_path, ingest, FakeTime())
    assert ingest.post_calls == []
    assert ingest.status_requests == []


async def test_a_product_the_api_would_refuse_stops_the_load_before_any_request(
    tmp_path: Path,
) -> None:
    dataset(tmp_path, [*products(2), Product("P9", "us", "", None, None, None, None)])
    ingest = FakeIngest()
    with pytest.raises(ValueError, match="P9"):
        await run_load(tmp_path, ingest, FakeTime())
    assert ingest.post_calls == []
    assert ingest.status_requests == []
