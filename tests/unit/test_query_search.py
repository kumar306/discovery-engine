"""What the query service sends to Qdrant: always the live condition, and only the filters."""

from typing import Any

import pytest
from qdrant_client import models
from qdrant_client.http.models import QueryResponse

from services.query.primitives import Hit
from services.query.search import Filters, QdrantSearch

LIVE = ("state", "live")
NO_FILTERS = Filters()
ALL_FILTERS = Filters(brand_key="nike", color_key="red", locale="us", doc_ids=("a", "b"))


def scored(
    point_id: int, doc_id: str, chunk_idx: int, score: float, **payload: Any
) -> models.ScoredPoint:
    return models.ScoredPoint(
        id=f"00000000-0000-0000-0000-{point_id:012d}",
        version=1,
        score=score,
        payload={
            "doc_id": doc_id,
            "chunk_idx": chunk_idx,
            "text": f"{doc_id} chunk {chunk_idx}",
            **payload,
        },
    )


class RecordingQdrant:
    """Stands in for the Qdrant client, keeps every call and returns what it was given."""

    def __init__(self, points: list[models.ScoredPoint] | None = None) -> None:
        self.points = points or []
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def query_points_groups(self, collection_name: str, **kwargs: Any) -> models.GroupsResult:
        self.calls.append(("groups", collection_name, kwargs))
        groups = [models.PointGroup(id=p.payload["doc_id"], hits=[p]) for p in self.points]  # type: ignore[index]
        return models.GroupsResult(groups=groups)

    async def query_points(self, collection_name: str, **kwargs: Any) -> QueryResponse:
        self.calls.append(("points", collection_name, kwargs))
        return QueryResponse(points=self.points)


def search_with(client: RecordingQdrant) -> QdrantSearch:
    return QdrantSearch(client, "chunks")  # type: ignore[arg-type]


def conditions_of(call: tuple[str, str, dict[str, Any]]) -> list[tuple[str, Any]]:
    query_filter = call[2]["query_filter"]
    assert isinstance(query_filter, models.Filter)
    assert query_filter.should is None
    assert query_filter.must_not is None
    found: list[tuple[str, Any]] = []
    for condition in query_filter.must or []:
        assert isinstance(condition, models.FieldCondition)
        match = condition.match
        if isinstance(match, models.MatchValue):
            found.append((condition.key, match.value))
        else:
            assert isinstance(match, models.MatchAny)
            found.append((condition.key, tuple(match.any)))
    return found


async def run(shape: str, using: str, filters: Filters, client: RecordingQdrant) -> list[Hit]:
    search = search_with(client)
    method = search.grouped_search if shape == "grouped" else search.chunk_search
    return await method(using, [1.0, 0.0], filters, 7)  # type: ignore[arg-type]


@pytest.mark.parametrize("shape", ["grouped", "chunks"])
@pytest.mark.parametrize("using", ["dense", "bm25"])
@pytest.mark.parametrize("filters", [NO_FILTERS, ALL_FILTERS], ids=["no filters", "all filters"])
async def test_every_request_carries_the_live_condition_first(
    shape: str, using: str, filters: Filters
) -> None:
    client = RecordingQdrant()
    await run(shape, using, filters, client)
    (call,) = client.calls
    assert conditions_of(call)[0] == LIVE
    assert call[2]["using"] == using
    assert call[2]["limit"] == 7
    assert call[1] == "chunks"


async def test_without_filters_the_live_condition_is_the_only_one() -> None:
    client = RecordingQdrant()
    await run("grouped", "dense", NO_FILTERS, client)
    assert conditions_of(client.calls[0]) == [LIVE]


async def test_the_filters_become_exact_conditions_and_nothing_else() -> None:
    client = RecordingQdrant()
    await run("chunks", "bm25", ALL_FILTERS, client)
    assert conditions_of(client.calls[0]) == [
        LIVE,
        ("brand_key", "nike"),
        ("color_key", "red"),
        ("locale", "us"),
        ("doc_id", ("a", "b")),
    ]


async def test_a_grouped_search_asks_for_one_chunk_per_product_by_doc_id() -> None:
    client = RecordingQdrant()
    await run("grouped", "dense", NO_FILTERS, client)
    kind, _, kwargs = client.calls[0]
    assert (kind, kwargs["group_by"], kwargs["group_size"]) == ("groups", "doc_id", 1)


async def test_a_chunk_search_does_not_group() -> None:
    client = RecordingQdrant()
    await run("chunks", "dense", NO_FILTERS, client)
    kind, _, kwargs = client.calls[0]
    assert kind == "points"
    assert "group_by" not in kwargs


async def test_a_sparse_query_vector_is_passed_through_as_it_is() -> None:
    client = RecordingQdrant()
    vector = models.SparseVector(indices=[2, 9], values=[1.0, 0.5])
    await search_with(client).grouped_search("bm25", vector, NO_FILTERS, 5)
    assert client.calls[0][2]["query"] == vector


async def test_points_become_hits_with_the_fields_a_result_needs() -> None:
    client = RecordingQdrant(
        [
            scored(1, "P1", 2, 0.9, title="Red Shoe", brand="Nike", color="Red"),
            scored(2, "P2", 0, 0.4),
        ]
    )
    hits = await run("chunks", "dense", NO_FILTERS, client)
    assert hits == [
        Hit(
            "00000000-0000-0000-0000-000000000001",
            "P1",
            2,
            0.9,
            "P1 chunk 2",
            "Red Shoe",
            "Nike",
            "Red",
        ),
        # a point written before titles existed has no title, brand or color
        Hit("00000000-0000-0000-0000-000000000002", "P2", 0, 0.4, "P2 chunk 0", "", None, None),
    ]


async def test_a_grouped_search_keeps_the_order_of_the_groups() -> None:
    client = RecordingQdrant([scored(1, "B", 0, 0.9), scored(2, "A", 0, 0.5)])
    hits = await run("grouped", "dense", NO_FILTERS, client)
    assert [hit.doc_id for hit in hits] == ["B", "A"]
