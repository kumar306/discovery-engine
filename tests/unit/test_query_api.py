"""Both flows through the real http app, with a fake encoder and a fake Qdrant search."""

import ast
import asyncio
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from prometheus_client import CollectorRegistry

from libs.observability.metrics import QueryMetrics
from services.query.app import create_app
from services.query.encoder import SparseQuery
from services.query.flow_chunks import ChunkFlow
from services.query.flow_search import SearchFlow
from services.query.primitives import Hit
from services.query.search import Filters
from services.query.settings import QuerySettings


def hit(doc_id: str, idx: int = 0, score: float = 1.0, **fields: Any) -> Hit:
    values: dict[str, Any] = {
        "point_id": f"{doc_id}-{idx}",
        "doc_id": doc_id,
        "chunk_idx": idx,
        "score": score,
        "text": f"{doc_id} chunk {idx} " + "word " * 40,
        "title": f"title {doc_id}",
        "brand": None,
        "color": None,
    }
    return Hit(**{**values, **fields})


class FakeEncoder:
    def __init__(self) -> None:
        self.dense_error: BaseException | None = None
        self.sparse_error: BaseException | None = None
        self.dense_queries: list[str] = []
        self.sparse_queries: list[str] = []

    async def encode_dense(self, query: str) -> list[float]:
        self.dense_queries.append(query)
        if self.dense_error is not None:
            raise self.dense_error
        return [1.0, 0.0]

    async def encode_sparse(self, query: str) -> SparseQuery:
        self.sparse_queries.append(query)
        if self.sparse_error is not None:
            raise self.sparse_error
        return SparseQuery([1], [1.0])


class FakeSearch:
    def __init__(self) -> None:
        self.grouped: dict[str, list[Hit]] = {"dense": [], "bm25": []}
        self.chunks: dict[str, list[Hit]] = {"dense": [], "bm25": []}
        # (shape, using) -> an error to raise, or seconds to sleep before answering
        self.errors: dict[tuple[str, str], BaseException] = {}
        self.delays: dict[tuple[str, str], float] = {}
        self.calls: list[tuple[str, str, Filters, int]] = []
        self.entered = asyncio.Event()
        self.hold: asyncio.Event | None = None

    async def _answer(
        self, shape: str, using: str, filters: Filters, limit: int, source: dict[str, list[Hit]]
    ) -> list[Hit]:
        self.calls.append((shape, using, filters, limit))
        self.entered.set()
        if self.hold is not None:
            await self.hold.wait()
        if (shape, using) in self.delays:
            await asyncio.sleep(self.delays[(shape, using)])
        if (shape, using) in self.errors:
            raise self.errors[(shape, using)]
        return list(source[using])

    async def grouped_search(
        self, using: str, vector: Any, filters: Filters, limit: int
    ) -> list[Hit]:
        return await self._answer("grouped", using, filters, limit, self.grouped)

    async def chunk_search(
        self, using: str, vector: Any, filters: Filters, limit: int
    ) -> list[Hit]:
        return await self._answer("chunks", using, filters, limit, self.chunks)


class Rig:
    def __init__(self, probes: dict[str, Any] | None = None, **settings: Any) -> None:
        self.settings = QuerySettings(**settings)
        self.encoder = FakeEncoder()
        self.search = FakeSearch()
        self.registry = CollectorRegistry()
        metrics = QueryMetrics(self.registry)
        self.app: FastAPI = create_app(
            SearchFlow(self.encoder, self.search, self.settings, metrics),  # type: ignore[arg-type]
            ChunkFlow(self.encoder, self.search, self.settings, metrics),  # type: ignore[arg-type]
            self.settings,
            metrics,
            self.registry,
            probes=probes,
        )

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://q")

    def degraded_count(self, flow: str, retriever: str) -> float:
        value = self.registry.get_sample_value(
            "query_degraded_total", {"flow": flow, "retriever": retriever}
        )
        return value or 0.0


@pytest.fixture
def rig() -> Rig:
    return Rig()


async def search(rig: Rig, body: Any, **headers: str) -> httpx.Response:
    async with rig.client() as client:
        return await client.post("/v1/search", json=body, headers=headers)


async def chunks(rig: Rig, body: Any) -> httpx.Response:
    async with rig.client() as client:
        return await client.post("/v1/chunks", json=body)


# ---------------- Flow S


def both_retrievers_answer(rig: Rig) -> None:
    rig.search.grouped["dense"] = [hit("a"), hit("b"), hit("c")]
    rig.search.grouped["bm25"] = [hit("b"), hit("a"), hit("d")]


async def test_flow_s_fuses_the_two_product_rankings(rig: Rig) -> None:
    both_retrievers_answer(rig)
    response = await search(rig, {"query": "red shoes"})
    assert response.status_code == 200
    body = response.json()
    assert [item["doc_id"] for item in body["results"]] == ["a", "b", "c", "d"]
    assert body["degraded"] == []
    assert body["request_id"] == response.headers["x-request-id"]
    assert set(body["timings_ms"]) == {"dense", "sparse", "fuse"}


async def test_explain_is_only_present_when_asked_for(rig: Rig) -> None:
    both_retrievers_answer(rig)
    plain = (await search(rig, {"query": "shoes"})).json()["results"][0]
    assert plain["explain"] is None
    asked = (await search(rig, {"query": "shoes", "explain": True})).json()["results"]
    assert asked[0]["explain"]["dense_rank"] == 1
    assert asked[0]["explain"]["sparse_rank"] == 2
    assert asked[0]["explain"]["rrf"] == pytest.approx(1 / 61 + 1 / 62)
    assert asked[3]["explain"]["dense_rank"] is None
    assert asked[3]["explain"]["sparse_rank"] == 3


async def test_a_result_shows_the_fields_of_its_best_chunk(rig: Rig) -> None:
    rig = Rig(snippet_chars=20)
    rig.search.grouped["dense"] = [hit("a", 2, brand="Nike", color="Red")]
    result = (await search(rig, {"query": "shoes"})).json()["results"][0]
    assert (result["doc_id"], result["title"], result["brand"], result["color"]) == (
        "a",
        "title a",
        "Nike",
        "Red",
    )
    assert result["snippet"] == hit("a", 2).text[:20]
    assert result["score"] == pytest.approx(1 / 61)


async def test_the_answer_has_at_most_k_products(rig: Rig) -> None:
    rig.search.grouped["dense"] = [hit(name) for name in "abcdef"]
    body = (await search(rig, {"query": "shoes", "k": 3})).json()
    assert [item["doc_id"] for item in body["results"]] == ["a", "b", "c"]
    default = (await search(rig, {"query": "shoes"})).json()
    assert len(default["results"]) == 6


async def test_both_retrievers_get_the_same_folded_filters_and_the_configured_depth(
    rig: Rig,
) -> None:
    both_retrievers_answer(rig)
    await search(
        rig, {"query": "shoes", "filters": {"brand": " NIKE ", "color": "Red", "locale": "us"}}
    )
    assert {call[1] for call in rig.search.calls} == {"dense", "bm25"}
    for shape, _, filters, limit in rig.search.calls:
        assert shape == "grouped"
        assert filters == Filters(brand_key="nike", color_key="red", locale="us")
        assert limit == rig.settings.candidates_per_retriever


async def test_no_filters_means_no_filter_conditions(rig: Rig) -> None:
    both_retrievers_answer(rig)
    await search(rig, {"query": "shoes"})
    assert all(call[2] == Filters() for call in rig.search.calls)


async def test_the_query_is_encoded_as_given_the_encoder_does_the_folding(rig: Rig) -> None:
    both_retrievers_answer(rig)
    await search(rig, {"query": "  Red   NIKE shoes "})
    assert rig.encoder.dense_queries == ["Red   NIKE shoes"]
    assert rig.encoder.sparse_queries == ["Red   NIKE shoes"]


@pytest.mark.parametrize(
    ("failing", "left"),
    [("dense", "bm25"), ("bm25", "dense")],
)
@pytest.mark.parametrize("how", ["encoder", "search", "timeout"])
async def test_one_retriever_failing_leaves_the_other_and_says_so(
    failing: str, left: str, how: str
) -> None:
    rig = Rig(retriever_timeout_s=0.05)
    rig.search.grouped["dense"] = [hit("d1"), hit("d2")]
    rig.search.grouped["bm25"] = [hit("s1"), hit("s2")]
    if how == "encoder":
        setattr(
            rig.encoder,
            "dense_error" if failing == "dense" else "sparse_error",
            RuntimeError("boom"),
        )
    elif how == "search":
        rig.search.errors[("grouped", failing)] = RuntimeError("qdrant said no")
    else:
        rig.search.delays[("grouped", failing)] = 1.0
    response = await search(rig, {"query": "shoes"})
    assert response.status_code == 200
    body = response.json()
    name = "dense" if failing == "dense" else "sparse"
    assert body["degraded"] == [name]
    prefix = "d" if left == "dense" else "s"
    assert [item["doc_id"] for item in body["results"]] == [f"{prefix}1", f"{prefix}2"]
    assert rig.degraded_count("search", name) == 1


async def test_a_failing_dense_encoder_never_reaches_the_dense_search() -> None:
    rig = Rig()
    rig.encoder.dense_error = RuntimeError("boom")
    rig.search.grouped["bm25"] = [hit("a")]
    await search(rig, {"query": "shoes"})
    assert [call[1] for call in rig.search.calls] == ["bm25"]


async def test_both_retrievers_failing_is_a_503_with_retry_after() -> None:
    rig = Rig(retry_after_s=4)
    rig.search.errors[("grouped", "dense")] = RuntimeError("down")
    rig.search.errors[("grouped", "bm25")] = RuntimeError("down")
    response = await search(rig, {"query": "shoes"})
    assert response.status_code == 503
    assert response.headers["retry-after"] == "4"
    body = response.json()
    assert (body["error"], body["request_id"]) == ("unavailable", response.headers["x-request-id"])


async def test_two_empty_but_working_retrievers_are_an_empty_200(rig: Rig) -> None:
    response = await search(rig, {"query": "shoes"})
    assert response.status_code == 200
    assert response.json()["results"] == []
    assert response.json()["degraded"] == []


BAD_SEARCHES = [
    pytest.param({"query": ""}, id="empty query"),
    pytest.param({"query": "   "}, id="blank query"),
    pytest.param({"query": "x" * 513}, id="query too long"),
    pytest.param({"query": 5}, id="query not a string"),
    pytest.param({}, id="no query"),
    pytest.param([], id="not an object"),
    pytest.param({"query": "a", "k": 0}, id="k zero"),
    pytest.param({"query": "a", "k": 101}, id="k too big"),
    pytest.param({"query": "a", "extra": 1}, id="unknown field"),
    pytest.param({"query": "a", "filters": {"locale": "fr"}}, id="unknown locale"),
    pytest.param({"query": "a", "filters": {"brand": ""}}, id="empty brand"),
    pytest.param({"query": "a", "filters": {"brand": "x" * 101}}, id="brand too long"),
]


@pytest.mark.parametrize("body", BAD_SEARCHES)
async def test_a_bad_search_is_a_422_in_our_format_and_reaches_nothing(rig: Rig, body: Any) -> None:
    response = await search(rig, body)
    assert response.status_code == 422
    answer = response.json()
    assert answer["error"] == "validation_failed"
    assert answer["request_id"] == response.headers["x-request-id"]
    assert answer["errors"]
    assert {"field", "message", "type"} <= set(answer["errors"][0])
    assert rig.encoder.dense_queries == []
    assert rig.search.calls == []


async def test_a_body_that_is_not_json_is_a_422_invalid_json(rig: Rig) -> None:
    async with rig.client() as client:
        response = await client.post(
            "/v1/search", content=b"{not json", headers={"content-type": "application/json"}
        )
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_json"


@pytest.mark.parametrize("route", ["/v1/search", "/v1/chunks"])
async def test_a_body_that_is_not_even_text_is_a_422_invalid_json(rig: Rig, route: str) -> None:
    async with rig.client() as client:
        response = await client.post(
            route, content=b"\xff\xfe\x00", headers={"content-type": "application/json"}
        )
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_json"
    assert response.json()["request_id"] == response.headers["x-request-id"]
    assert rig.search.calls == []


async def test_a_body_over_the_size_limit_is_a_413_before_anything_runs() -> None:
    rig = Rig(max_body_bytes=200)
    response = await search(rig, {"query": "x" * 400})
    assert response.status_code == 413
    assert response.json()["error"] == "payload_too_large"
    assert rig.search.calls == []


async def test_a_valid_request_id_is_kept_and_an_unsafe_one_is_replaced(rig: Rig) -> None:
    kept = await search(rig, {"query": "a"}, **{"x-request-id": "abc-123"})
    assert (kept.headers["x-request-id"], kept.json()["request_id"]) == ("abc-123", "abc-123")
    replaced = await search(rig, {"query": "a"}, **{"x-request-id": "bad id\nforged line"})
    assert re.fullmatch(r"[0-9a-f]{32}", replaced.headers["x-request-id"])


async def test_a_full_house_answers_503_at_once_and_frees_up_afterwards() -> None:
    rig = Rig(max_inflight_requests=1, admission_wait_s=0.01)
    rig.search.hold = asyncio.Event()
    rig.search.grouped["dense"] = [hit("a")]
    async with rig.client() as client:
        first = asyncio.create_task(client.post("/v1/search", json={"query": "a"}))
        await asyncio.wait_for(rig.search.entered.wait(), timeout=2)
        shed = await client.post("/v1/search", json={"query": "b"})
        rig.search.hold.set()
        assert (await first).status_code == 200
        again = await client.post("/v1/search", json={"query": "c"})
    assert shed.status_code == 503
    assert shed.json()["error"] == "overloaded"
    assert "retry-after" in shed.headers
    assert again.status_code == 200


# ---------------- Flow C


def chunk_rankings(rig: Rig) -> None:
    rig.search.chunks["dense"] = [hit("p1", 0), hit("p1", 1), hit("p2", 0)]
    rig.search.chunks["bm25"] = [hit("p1", 1), hit("p3", 0)]


async def test_flow_c_fuses_chunks_and_keeps_two_chunks_of_one_product_apart(rig: Rig) -> None:
    chunk_rankings(rig)
    response = await chunks(rig, {"query": "shoes"})
    assert response.status_code == 200
    body = response.json()
    assert [item["point_id"] for item in body["chunks"]] == ["p1-1", "p1-0", "p3-0", "p2-0"]
    assert [item["doc_id"] for item in body["chunks"]].count("p1") == 2
    best = body["chunks"][0]
    assert (best["dense_rank"], best["sparse_rank"]) == (2, 1)
    assert best["rrf"] == pytest.approx(1 / 62 + 1 / 61)
    assert (best["title"], best["chunk_idx"]) == ("title p1", 1)
    assert body["degraded"] == []


async def test_flow_c_asks_for_chunks_not_groups_with_the_product_list_and_its_own_depth(
    rig: Rig,
) -> None:
    chunk_rankings(rig)
    await chunks(rig, {"query": "shoes", "doc_ids": ["p1", "p2"], "per_retriever_limit": 7})
    assert {call[0] for call in rig.search.calls} == {"chunks"}
    for _, _, filters, limit in rig.search.calls:
        assert filters == Filters(doc_ids=("p1", "p2"))
        assert limit == 7


async def test_flow_c_defaults_to_the_configured_chunk_depth(rig: Rig) -> None:
    chunk_rankings(rig)
    await chunks(rig, {"query": "shoes"})
    assert {call[3] for call in rig.search.calls} == {rig.settings.chunks_per_retriever}


async def test_flow_c_degrades_like_flow_s_but_counts_under_its_own_name() -> None:
    rig = Rig()
    rig.search.chunks["dense"] = [hit("p1", 0)]
    rig.search.errors[("chunks", "bm25")] = RuntimeError("down")
    response = await chunks(rig, {"query": "shoes"})
    assert response.status_code == 200
    assert response.json()["degraded"] == ["sparse"]
    assert (rig.degraded_count("chunks", "sparse"), rig.degraded_count("search", "sparse")) == (
        1,
        0,
    )


async def test_flow_c_with_both_retrievers_failing_is_a_503() -> None:
    rig = Rig()
    rig.search.errors[("chunks", "dense")] = RuntimeError("down")
    rig.search.errors[("chunks", "bm25")] = RuntimeError("down")
    response = await chunks(rig, {"query": "shoes"})
    assert response.status_code == 503
    assert "retry-after" in response.headers


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"query": ""}, id="empty query"),
        pytest.param({"query": "a", "per_retriever_limit": 0}, id="limit zero"),
        pytest.param({"query": "a", "per_retriever_limit": 101}, id="limit too big"),
        pytest.param({"query": "a", "doc_ids": ["has space"]}, id="bad product id"),
        pytest.param({"query": "a", "doc_ids": ["p"] * 101}, id="too many product ids"),
        pytest.param({"query": "a", "filters": {}}, id="unknown field"),
    ],
)
async def test_a_bad_chunk_request_is_a_422_and_reaches_nothing(rig: Rig, body: Any) -> None:
    response = await chunks(rig, body)
    assert response.status_code == 422
    assert response.json()["error"] == "validation_failed"
    assert rig.search.calls == []


# ---------------- health and structure


async def test_readyz_follows_the_probes() -> None:
    async def down() -> None:
        raise RuntimeError("qdrant is gone")

    async def fine() -> None:
        return None

    for probe, expected in ((fine, 200), (down, 503)):
        async with Rig(probes={"qdrant": probe}).client() as client:
            assert (await client.get("/healthz")).status_code == 200
            assert (await client.get("/readyz")).status_code == expected


def imports_of(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found.add(node.module)
    return found


QUERY_PACKAGE = Path(__file__).parents[2] / "services" / "query"


def test_the_query_package_never_imports_a_database_or_the_worker() -> None:
    for path in QUERY_PACKAGE.glob("*.py"):
        for name in imports_of(path):
            assert not name.startswith(
                ("psycopg", "libs.db", "services.embed", "services.ingest")
            ), (
                path.name,
                name,
            )


def test_the_two_flows_share_no_handler_code() -> None:
    assert "services.query.flow_chunks" not in imports_of(QUERY_PACKAGE / "flow_search.py")
    assert "services.query.flow_search" not in imports_of(QUERY_PACKAGE / "flow_chunks.py")
