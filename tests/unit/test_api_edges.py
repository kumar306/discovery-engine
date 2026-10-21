"""The HTTP edges of the ingest api that need no database and no S3: ids, size cap, field rules.

The service behind the app is a recorder, so every test can also say whether it was reached.
"""

import asyncio
import io
import json
import logging
import re
from collections.abc import AsyncIterator, Iterator, Sequence
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from psycopg import OperationalError
from psycopg.errors import AdminShutdown, LockNotAvailable
from psycopg_pool import PoolTimeout

from libs.observability.logging import JsonFormatter
from libs.resilience.bulkhead import Overloaded
from libs.schemas import MAX_STATUS_BATCH, UpsertRequest
from services.ingest.app import create_app
from services.ingest.repository import DocumentStatus
from services.ingest.service import WriteAccepted, WriteRefused
from services.ingest.settings import IngestSettings
from services.ingest.storage import StorageUnavailable

NOW = datetime(2026, 10, 12, 9, 30, tzinfo=UTC)
LIMIT = 2000


class RecordingIngest:
    def __init__(self, result: WriteAccepted | WriteRefused | None = None) -> None:
        self.result = result or WriteAccepted("D1", 1, NOW)
        self.error: BaseException | None = None
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.calls: list[tuple[UpsertRequest, str]] = []
        self.deletes: list[tuple[str, int, str]] = []
        self.statuses: dict[str, DocumentStatus] = {}
        self.status_lookups: list[str] = []
        self.status_batches: list[list[str]] = []

    async def _answer(self) -> WriteAccepted | WriteRefused:
        self.entered.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error
        return self.result

    async def upsert(self, request: UpsertRequest, request_id: str) -> WriteAccepted | WriteRefused:
        self.calls.append((request, request_id))
        return await self._answer()

    async def delete(
        self, doc_id: str, version: int, request_id: str
    ) -> WriteAccepted | WriteRefused:
        self.deletes.append((doc_id, version, request_id))
        return await self._answer()

    async def status(self, doc_id: str) -> DocumentStatus | None:
        self.status_lookups.append(doc_id)
        self.entered.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error
        return self.statuses.get(doc_id)

    async def status_batch(self, doc_ids: Sequence[str]) -> dict[str, DocumentStatus]:
        self.status_batches.append(list(doc_ids))
        self.entered.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error
        return {doc_id: self.statuses[doc_id] for doc_id in doc_ids if doc_id in self.statuses}


def product(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "corpus": "esci",
        "doc_id": "D1",
        "version": 1,
        "title": "Wireless Mouse",
        "locale": "us",
    }
    return {**body, **overrides}


class LogCapture:
    def __init__(self) -> None:
        self.stream = io.StringIO()

    def lines(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.stream.getvalue().splitlines()]


@pytest.fixture
def logs() -> Iterator[LogCapture]:
    capture = LogCapture()
    handler = logging.StreamHandler(capture.stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("ingest.request")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield capture
    logger.removeHandler(handler)


async def make_client(service: RecordingIngest, **settings: Any) -> httpx.AsyncClient:
    app = create_app(service, IngestSettings(max_body_bytes=LIMIT, **settings))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api")


@pytest.fixture
async def service() -> RecordingIngest:
    return RecordingIngest()


@pytest.fixture
async def client(service: RecordingIngest) -> AsyncIterator[httpx.AsyncClient]:
    async with await make_client(service) as client:
        yield client


async def post(client: httpx.AsyncClient, body: Any, **headers: str) -> httpx.Response:
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return await client.post("/v1/documents", content=content, headers=headers)


# the request id


async def test_a_request_id_is_generated_when_the_client_sends_none(
    client: httpx.AsyncClient,
) -> None:
    response = await post(client, product())
    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-request-id"])


async def test_a_valid_request_id_from_the_client_is_kept_and_passed_on(
    client: httpx.AsyncClient, service: RecordingIngest
) -> None:
    response = await post(client, product(), **{"X-Request-ID": "client-abc.123_X"})
    assert response.headers["x-request-id"] == "client-abc.123_X"
    assert response.json()["request_id"] == "client-abc.123_X"
    assert service.calls[0][1] == "client-abc.123_X"


@pytest.mark.parametrize(
    "bad",
    ["has space", "a" * 65, "x;y", "<script>", "id\nforged log line", "id\r\nSet-Cookie: x=1"],
)
async def test_a_request_id_with_unsafe_characters_is_replaced(
    client: httpx.AsyncClient, bad: str
) -> None:
    response = await post(client, product(), **{"X-Request-ID": bad})
    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-request-id"])


async def test_every_kind_of_response_carries_the_request_id(client: httpx.AsyncClient) -> None:
    responses = [
        await post(client, product()),  # 202
        await post(client, b"{"),  # 422
        await post(client, b" " * (LIMIT + 1)),  # 413
        await client.get("/nothing-here"),  # 404
        await client.get("/v1/documents"),  # 405
    ]
    assert [r.status_code for r in responses] == [202, 422, 413, 404, 405]
    assert all(re.fullmatch(r"[0-9a-f]{32}", r.headers["x-request-id"]) for r in responses)


async def test_each_request_writes_one_log_line_with_its_request_id(
    client: httpx.AsyncClient, logs: LogCapture
) -> None:
    response = await post(client, product(doc_id="D7", version=4), **{"X-Request-ID": "trace-1"})
    (line,) = logs.lines()
    assert line["request_id"] == "trace-1" == response.headers["x-request-id"]
    assert (line["method"], line["path"], line["status"]) == ("POST", "/v1/documents", 202)
    assert (line["outcome"], line["doc_id"], line["version"]) == ("accepted", "D7", 4)
    assert line["duration_ms"] >= 0


async def test_a_rejected_request_is_logged_with_its_reason(
    client: httpx.AsyncClient, logs: LogCapture
) -> None:
    await post(client, b" " * (LIMIT + 1))
    (line,) = logs.lines()
    assert (line["status"], line["outcome"]) == (413, "payload_too_large")


# the size cap


async def test_a_declared_size_over_the_limit_is_refused_before_the_service_is_reached(
    client: httpx.AsyncClient, service: RecordingIngest
) -> None:
    response = await post(client, b" " * (LIMIT + 1))
    assert response.status_code == 413
    assert response.json()["error"] == "payload_too_large"
    assert service.calls == []


async def test_a_chunked_body_over_the_limit_is_refused_too(
    client: httpx.AsyncClient, service: RecordingIngest
) -> None:
    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(5):  # no content length is sent, only the stream shows the real size
            yield b" " * (LIMIT // 2)

    response = await client.post("/v1/documents", content=chunks())
    assert response.status_code == 413
    assert service.calls == []


async def test_a_body_of_exactly_the_limit_is_accepted_and_one_more_byte_is_not(
    client: httpx.AsyncClient, service: RecordingIngest
) -> None:
    document = json.dumps(product()).encode()
    exact = document + b" " * (LIMIT - len(document))  # trailing spaces are valid json
    assert len(exact) == LIMIT
    assert (await post(client, exact)).status_code == 202
    assert (await post(client, exact + b" ")).status_code == 413
    assert len(service.calls) == 1


# the field rules


@pytest.mark.parametrize(
    ("body", "code", "fields"),
    [
        pytest.param(b"", "invalid_json", None, id="empty body"),
        pytest.param(b"{", "invalid_json", None, id="broken json"),
        pytest.param(b"[]", "validation_failed", None, id="not an object"),
        pytest.param(
            product(esci_label="E"), "validation_failed", {"esci_label"}, id="unknown field"
        ),
        pytest.param(product(doc_id="has/slash"), "validation_failed", {"doc_id"}, id="bad doc id"),
        pytest.param(product(version="3"), "validation_failed", {"version"}, id="version as text"),
        pytest.param(product(version=True), "validation_failed", {"version"}, id="version as bool"),
        pytest.param(product(version=-1), "validation_failed", {"version"}, id="negative version"),
        pytest.param(product(locale="xx"), "validation_failed", {"locale"}, id="unknown locale"),
        pytest.param(
            product(doc_id="", version=-1, locale="xx"),
            "validation_failed",
            {"doc_id", "version", "locale"},
            id="three problems at once",
        ),
    ],
)
async def test_bad_bodies_get_a_422_naming_the_fields_and_never_reach_the_service(
    client: httpx.AsyncClient,
    service: RecordingIngest,
    body: Any,
    code: str,
    fields: set[str] | None,
) -> None:
    response = await post(client, body)
    assert response.status_code == 422
    assert response.json()["error"] == code
    if fields is not None:
        assert {item["field"] for item in response.json()["errors"]} == fields
    assert service.calls == []


async def test_a_missing_title_is_named(client: httpx.AsyncClient) -> None:
    body = product()
    del body["title"]
    response = await post(client, body)
    assert [item["field"] for item in response.json()["errors"]] == ["title"]


async def test_the_422_never_echoes_what_the_client_sent(client: httpx.AsyncClient) -> None:
    response = await post(client, product(doc_id="TOP-SECRET/../value", title="x" * 500))
    assert response.status_code == 422
    assert "TOP-SECRET" not in response.text
    assert all(set(item) == {"field", "message", "type"} for item in response.json()["errors"])


# what the service answers


async def test_a_valid_body_reaches_the_service_once_and_gets_the_202_body(
    client: httpx.AsyncClient, service: RecordingIngest
) -> None:
    response = await post(client, product(doc_id="D1", version=1))
    assert response.status_code == 202
    body = response.json()
    assert body == {
        "request_id": response.headers["x-request-id"],
        "doc_id": "D1",
        "version": 1,
        "accepted_at": NOW.isoformat(),
    }
    ((received, request_id),) = service.calls
    assert (received.doc_id, received.version, request_id) == ("D1", 1, body["request_id"])


@pytest.mark.parametrize(
    ("refusal", "error", "current"),
    [
        pytest.param(WriteRefused("stale_version", 7), "stale_version", 7, id="stale"),
        pytest.param(WriteRefused("content_conflict", 5), "content_conflict", 5, id="conflict"),
        pytest.param(
            WriteRefused("content_conflict", None), "content_conflict", None, id="leftover object"
        ),
    ],
)
async def test_a_refusal_becomes_a_409_with_the_current_version(
    refusal: WriteRefused, error: str, current: int | None
) -> None:
    async with await make_client(RecordingIngest(refusal)) as client:
        response = await post(client, product())
    assert response.status_code == 409
    body = response.json()
    assert (body["error"], body["current_version"]) == (error, current)
    assert body["request_id"] == response.headers["x-request-id"]
    # the human text has to match the code, a stale write is not a content conflict
    assert ("newer version" in body["message"]) == (error == "stale_version")
    assert ("different content" in body["message"]) == (error == "content_conflict")


# DELETE


async def delete(
    client: httpx.AsyncClient, doc_id: str = "D1", version: str | None = "3"
) -> httpx.Response:
    params = None if version is None else {"version": version}
    return await client.delete(f"/v1/documents/{doc_id}", params=params)


async def test_a_delete_reaches_the_service_and_gets_the_202_body(
    client: httpx.AsyncClient, service: RecordingIngest
) -> None:
    service.result = WriteAccepted("D1", 3, NOW)
    response = await delete(client, "D1", "3")
    assert response.status_code == 202
    assert response.json() == {
        "request_id": response.headers["x-request-id"],
        "doc_id": "D1",
        "version": 3,
        "accepted_at": NOW.isoformat(),
    }
    assert service.deletes == [("D1", 3, response.json()["request_id"])]


async def test_the_largest_version_a_bigint_can_hold_is_accepted(
    client: httpx.AsyncClient, service: RecordingIngest
) -> None:
    assert (await delete(client, "D1", str(2**63 - 1))).status_code == 202
    assert service.deletes[0][1] == 2**63 - 1


@pytest.mark.parametrize(
    ("doc_id", "version", "fields"),
    [
        pytest.param("D1", None, {"version"}, id="version missing"),
        pytest.param("D1", "", {"version"}, id="version empty"),
        pytest.param("D1", "-1", {"version"}, id="negative"),
        pytest.param("D1", "1.5", {"version"}, id="decimal"),
        pytest.param("D1", "abc", {"version"}, id="not a number"),
        pytest.param("D1", "+5", {"version"}, id="explicit plus sign"),
        pytest.param("D1", " 5", {"version"}, id="leading space"),
        pytest.param("D1", "\u0665", {"version"}, id="an arabic indic digit"),
        pytest.param("D1", str(2**63), {"version"}, id="one past bigint"),
        pytest.param("D1", "9" * 25, {"version"}, id="far past bigint"),
        pytest.param("a" * 65, "1", {"doc_id"}, id="doc id too long"),
        pytest.param("has space", "1", {"doc_id"}, id="doc id with a space"),
        pytest.param("a" * 65, "x", {"doc_id", "version"}, id="both wrong"),
    ],
)
async def test_a_delete_with_bad_input_gets_a_422_and_never_reaches_the_service(
    client: httpx.AsyncClient,
    service: RecordingIngest,
    doc_id: str,
    version: str | None,
    fields: set[str],
) -> None:
    response = await delete(client, doc_id, version)
    assert response.status_code == 422
    assert response.json()["error"] == "validation_failed"
    assert {item["field"] for item in response.json()["errors"]} == fields
    assert service.deletes == []


async def test_a_refused_delete_is_a_409_like_a_refused_write() -> None:
    service = RecordingIngest(WriteRefused("stale_version", 9))
    async with await make_client(service) as client:
        response = await delete(client, "D1", "3")
    assert response.status_code == 409
    assert (response.json()["error"], response.json()["current_version"]) == ("stale_version", 9)


# STATUS


async def get_status(client: httpx.AsyncClient, doc_id: str = "D1") -> httpx.Response:
    return await client.get(f"/v1/documents/{doc_id}/status")


async def test_a_known_document_gets_its_three_versions(
    client: httpx.AsyncClient, service: RecordingIngest
) -> None:
    service.statuses["D1"] = DocumentStatus(
        latest_version=5, current_version=4, indexed_version=3, deleted=False
    )
    response = await get_status(client)
    assert response.status_code == 200
    assert response.json() == {
        "request_id": response.headers["x-request-id"],
        "doc_id": "D1",
        "latest_version": 5,
        "current_version": 4,
        "indexed_version": 3,
        "deleted": False,
    }
    assert service.status_lookups == ["D1"]


async def test_versions_the_worker_has_not_reached_are_null(
    client: httpx.AsyncClient, service: RecordingIngest
) -> None:
    service.statuses["D1"] = DocumentStatus(5, None, None, False)
    body = (await get_status(client)).json()
    assert (body["current_version"], body["indexed_version"]) == (None, None)


async def test_an_unknown_document_is_a_404_in_our_error_format(
    client: httpx.AsyncClient, service: RecordingIngest
) -> None:
    response = await get_status(client, "NOPE")
    assert response.status_code == 404
    body = response.json()
    assert (body["error"], body["request_id"]) == ("not_found", response.headers["x-request-id"])
    assert service.status_lookups == ["NOPE"]


@pytest.mark.parametrize(
    "doc_id",
    [pytest.param("a" * 65, id="too long"), pytest.param("has space", id="with a space")],
)
async def test_a_status_request_with_a_bad_id_is_a_422_and_never_reaches_the_service(
    client: httpx.AsyncClient, service: RecordingIngest, doc_id: str
) -> None:
    response = await get_status(client, doc_id)
    assert response.status_code == 422
    assert response.json()["error"] == "validation_failed"
    assert {item["field"] for item in response.json()["errors"]} == {"doc_id"}
    assert service.status_lookups == []


# STATUS BATCH


async def get_statuses(client: httpx.AsyncClient, doc_ids: Any) -> httpx.Response:
    return await client.post("/v1/document-statuses", json={"doc_ids": doc_ids})


async def test_a_batch_answers_for_the_known_documents_in_the_order_asked(
    client: httpx.AsyncClient, service: RecordingIngest
) -> None:
    service.statuses["D1"] = DocumentStatus(5, 4, 3, False)
    service.statuses["D3"] = DocumentStatus(2, None, None, False)
    response = await get_statuses(client, ["D3", "D2", "D1", "D3"])
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == response.headers["x-request-id"]
    # D2 is unknown and left out, D3 was asked twice and comes once
    assert [item["doc_id"] for item in body["statuses"]] == ["D3", "D1"]
    assert body["statuses"][1] == {
        "doc_id": "D1",
        "latest_version": 5,
        "current_version": 4,
        "indexed_version": 3,
        "deleted": False,
    }
    assert service.status_batches == [["D3", "D2", "D1", "D3"]]


@pytest.mark.parametrize(
    ("body", "field"),
    [
        pytest.param({"doc_ids": []}, "doc_ids", id="empty list"),
        pytest.param({"doc_ids": ["D1", "has space"]}, "doc_ids.1", id="a bad id"),
        pytest.param({"doc_ids": ["D1"], "extra": 1}, "extra", id="an unknown field"),
        pytest.param({}, "doc_ids", id="no list"),
        pytest.param({"doc_ids": "D1"}, "doc_ids", id="not a list"),
    ],
)
async def test_a_bad_batch_gets_a_422_and_never_reaches_the_service(
    client: httpx.AsyncClient, service: RecordingIngest, body: dict[str, Any], field: str
) -> None:
    response = await client.post("/v1/document-statuses", json=body)
    assert response.status_code == 422
    assert response.json()["error"] == "validation_failed"
    assert field in {item["field"] for item in response.json()["errors"]}
    assert service.status_batches == []


async def test_more_ids_than_the_limit_get_a_422(service: RecordingIngest) -> None:
    ids = [f"D{n}" for n in range(MAX_STATUS_BATCH + 1)]
    app = create_app(service, IngestSettings(max_body_bytes=500_000))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
        response = await get_statuses(client, ids)
    assert response.status_code == 422
    assert {item["field"] for item in response.json()["errors"]} == {"doc_ids"}
    assert service.status_batches == []


async def test_a_batch_that_is_not_json_is_a_422_invalid_json(
    client: httpx.AsyncClient, service: RecordingIngest
) -> None:
    response = await client.post("/v1/document-statuses", content=b"{not json")
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_json"
    assert service.status_batches == []


async def test_a_batch_body_over_the_size_limit_is_a_413(
    client: httpx.AsyncClient, service: RecordingIngest
) -> None:
    response = await get_statuses(client, [f"D{n}" for n in range(LIMIT)])
    assert response.status_code == 413
    assert service.status_batches == []


# the 503 answers: every way a dependency can say no

UNAVAILABLE = [
    pytest.param(
        StorageUnavailable("s3 down"), "storage_unavailable", "storage_unavailable", id="s3 down"
    ),
    pytest.param(Overloaded("full"), "storage_overloaded", "overloaded", id="s3 bulkhead full"),
    pytest.param(
        PoolTimeout("no connection"), "pool_exhausted", "database_busy", id="pool exhausted"
    ),
    pytest.param(LockNotAvailable("lock"), "lock_timeout", "database_busy", id="row lock timeout"),
    pytest.param(
        OperationalError("refused"), "database_unavailable", "database_busy", id="cannot connect"
    ),
    pytest.param(
        AdminShutdown("restart"), "database_unavailable", "database_busy", id="database restarting"
    ),
]


@pytest.mark.parametrize(("error", "outcome", "code"), UNAVAILABLE)
@pytest.mark.parametrize("route", ["post", "delete", "status", "statuses"])
async def test_a_failing_dependency_is_a_503_with_retry_after_and_the_request_id(
    logs: LogCapture, error: Exception, outcome: str, code: str, route: str
) -> None:
    service = RecordingIngest()
    service.error = error
    async with await make_client(service, retry_after_s=7) as client:
        if route == "post":
            response = await post(client, product())
        elif route == "delete":
            response = await delete(client, "D1", "3")
        elif route == "status":
            response = await get_status(client)
        else:
            response = await get_statuses(client, ["D1"])
    assert response.status_code == 503
    assert response.headers["retry-after"] == "7"
    body = response.json()
    assert (body["error"], body["request_id"]) == (code, response.headers["x-request-id"])
    (line,) = logs.lines()
    assert (line["status"], line["outcome"]) == (503, outcome)


# the outer admission limit


async def test_a_full_house_answers_503_at_once_and_frees_up_afterwards(
    logs: LogCapture,
) -> None:
    service = RecordingIngest()
    service.release.clear()  # the first request will sit inside the service
    async with await make_client(service, max_inflight_requests=1, admission_wait_s=0.01) as client:
        first = asyncio.create_task(post(client, product(doc_id="D1")))
        await asyncio.wait_for(service.entered.wait(), timeout=5)
        # bounded, so a missing limit fails the test instead of waiting for the blocked service
        shed = await asyncio.wait_for(post(client, product(doc_id="D2")), timeout=5)
        assert shed.status_code == 503
        assert shed.json()["error"] == "overloaded"
        assert shed.headers["retry-after"] == "1"
        # probes and unknown paths are not counted, they must answer even when the api is full
        assert (await client.get("/nothing-here")).status_code == 404
        service.release.set()
        assert (await first).status_code == 202
        assert (await post(client, product(doc_id="D3"))).status_code == 202
    assert [doc[0].doc_id for doc in service.calls] == ["D1", "D3"]  # the shed one never got in
    assert "admission_full" in [line["outcome"] for line in logs.lines()]


# anything we did not plan for


async def test_an_unexpected_error_is_a_500_that_tells_the_client_nothing(
    logs: LogCapture,
) -> None:
    service = RecordingIngest()
    service.error = RuntimeError("secret internal detail")
    async with await make_client(service) as client:
        response = await post(client, product())
    assert response.status_code == 500
    body = response.json()
    assert (body["error"], body["request_id"]) == (
        "internal_error",
        response.headers["x-request-id"],
    )
    assert "secret internal detail" not in response.text
    assert "Traceback" not in response.text
    error_line, access_line = logs.lines()
    assert error_line["level"] == "ERROR"
    assert "secret internal detail" in error_line["exception"]
    assert error_line["request_id"] == access_line["request_id"] == body["request_id"]
    assert (access_line["status"], access_line["outcome"]) == (500, "internal_error")


# text that is only whitespace is empty once stripped, the api refuses it with a 422 even though the
# schema in /openapi.json says length 1 is enough (the contract test replaces such strings)

BLANKS = [" ", "\t\n", "\x0b", "\u00a0", "\u2003"]


@pytest.mark.parametrize("blank", BLANKS)
@pytest.mark.parametrize("field", ["title", "brand", "color", "description"])
async def test_a_text_field_of_only_whitespace_is_a_422_and_reaches_nothing(
    client: httpx.AsyncClient, service: RecordingIngest, field: str, blank: str
) -> None:
    response = await post(client, product(**{field: blank}))
    assert response.status_code == 422
    assert response.json()["errors"][0]["field"] == field
    assert service.calls == []


@pytest.mark.parametrize("blank", BLANKS)
async def test_a_bullet_of_only_whitespace_is_a_422_and_names_its_place(
    client: httpx.AsyncClient, service: RecordingIngest, blank: str
) -> None:
    response = await post(client, product(bullets=["ok", blank]))
    assert response.status_code == 422
    assert response.json()["errors"][0]["field"] == "bullets.1"
    assert service.calls == []
