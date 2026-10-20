"""The query service's Qdrant adapter against the real server: only live points, grouped or not."""

from typing import Literal

import pytest
from qdrant_client import AsyncQdrantClient

from libs.embedding.collection import filter_key
from services.embed.qdrant import ChunkIndex, LiveChunk
from services.embed.settings import EmbedSettings
from services.query.encoder import SparseQuery
from services.query.search import Filters, QdrantSearch

from .test_qdrant_index import chunk, ident

# the dense query is close to every test chunk, the sparse query is on the term all of them share
DENSE_QUERY = [1.0, 1.0, 0.5, 0.25]
SPARSE_QUERY = SparseQuery(indices=[1], values=[1.0])

Using = Literal["dense", "bm25"]


async def load_every_state(qdrant: AsyncQdrantClient, collection: EmbedSettings) -> QdrantSearch:
    """P1 has two live chunks, P2 one; P3 is staging, P4 superseded, P5 deleted."""
    store = ChunkIndex(qdrant, collection.qdrant_collection)
    await store.stage(
        [
            chunk(1, "P1", idx=0, brand="Nike", color="Red"),
            chunk(2, "P1", idx=1, brand="Nike", color="Red"),
            chunk(3, "P2", brand="Puma", color="Blue"),
            chunk(4, "P3"),
            chunk(5, "P4"),
            chunk(6, "P5"),
        ]
    )
    await store.publish(
        1,
        [
            LiveChunk(ident(1), 0),
            LiveChunk(ident(2), 1),
            LiveChunk(ident(3), 0),
            LiveChunk(ident(5), 0),
        ],
        [],
    )
    await store.publish(2, [LiveChunk(ident(6), 0)], [ident(5)])
    await store.mark_deleted("P5")
    return QdrantSearch(qdrant, collection.qdrant_collection)


def vector_for(using: Using) -> list[float] | SparseQuery:
    return DENSE_QUERY if using == "dense" else SPARSE_QUERY


@pytest.mark.parametrize("using", ["dense", "bm25"])
async def test_grouped_search_returns_each_live_product_once(
    qdrant: AsyncQdrantClient, collection: EmbedSettings, using: Using
) -> None:
    search = await load_every_state(qdrant, collection)
    hits = await search.grouped_search(using, vector_for(using), Filters(), 10)
    assert sorted(hit.doc_id for hit in hits) == ["P1", "P2"]


@pytest.mark.parametrize("using", ["dense", "bm25"])
async def test_chunk_search_returns_every_live_chunk_so_a_product_may_appear_twice(
    qdrant: AsyncQdrantClient, collection: EmbedSettings, using: Using
) -> None:
    search = await load_every_state(qdrant, collection)
    hits = await search.chunk_search(using, vector_for(using), Filters(), 10)
    assert sorted(hit.point_id for hit in hits) == sorted(str(ident(n)) for n in (1, 2, 3))
    assert sorted(hit.doc_id for hit in hits) == ["P1", "P1", "P2"]


async def test_staging_superseded_and_deleted_points_are_never_returned(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    search = await load_every_state(qdrant, collection)
    for using in ("dense", "bm25"):
        grouped = await search.grouped_search(using, vector_for(using), Filters(), 10)  # type: ignore[arg-type]
        chunks = await search.chunk_search(using, vector_for(using), Filters(), 10)  # type: ignore[arg-type]
        assert {hit.doc_id for hit in grouped + chunks}.isdisjoint({"P3", "P4", "P5"})


async def test_the_brand_filter_is_exact_on_the_folded_key(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    search = await load_every_state(qdrant, collection)
    nike = await search.grouped_search(
        "dense", DENSE_QUERY, Filters(brand_key=filter_key("NIKE ")), 10
    )
    assert [hit.doc_id for hit in nike] == ["P1"]
    assert nike[0].brand == "Nike"  # the original is what a result shows
    # the unfolded value is a different string for qdrant, folding is the caller's job
    assert await search.grouped_search("dense", DENSE_QUERY, Filters(brand_key="Nike"), 10) == []


async def test_color_and_a_product_list_narrow_the_search_together(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    search = await load_every_state(qdrant, collection)
    both = Filters(color_key="red", doc_ids=("P1", "P2"))
    assert {hit.doc_id for hit in await search.chunk_search("bm25", SPARSE_QUERY, both, 10)} == {
        "P1"
    }
    only_p2 = Filters(doc_ids=("P2",))
    assert {hit.doc_id for hit in await search.chunk_search("bm25", SPARSE_QUERY, only_p2, 10)} == {
        "P2"
    }


async def test_limit_counts_products_when_grouped_and_chunks_when_not(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    search = await load_every_state(qdrant, collection)
    assert len(await search.grouped_search("dense", DENSE_QUERY, Filters(), 1)) == 1
    assert len(await search.chunk_search("dense", DENSE_QUERY, Filters(), 2)) == 2


async def test_the_hit_carries_the_fields_a_result_shows(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    search = await load_every_state(qdrant, collection)
    (hit,) = await search.grouped_search(
        "dense", DENSE_QUERY, Filters(locale="us", brand_key="puma"), 5
    )
    assert (hit.doc_id, hit.title, hit.brand, hit.color) == (
        "P2",
        "title of chunk 3",
        "Puma",
        "Blue",
    )
    assert hit.text == "text of chunk 3"
