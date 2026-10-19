"""The qdrant adapter against the real server: the collection, staging, publishing, deleting."""

import time
import uuid
from typing import Any

import pytest
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from libs.embedding.collection import CollectionMismatch, verify_collection
from services.embed.qdrant import (
    ChunkIndex,
    ChunkPayload,
    IndexedChunk,
    LiveChunk,
    ensure_collection,
)
from services.embed.settings import EmbedSettings

from .conftest import TEST_DIM, read_points

CHUNK_HASH = "a" * 64


def ident(number: int) -> uuid.UUID:
    return uuid.UUID(int=number)


def chunk(
    number: int, doc_id: str = "P1", version: int = 1, idx: int = 0, **fields: Any
) -> IndexedChunk:
    payload = ChunkPayload(
        doc_id=doc_id,
        version=version,
        chunk_idx=idx,
        content_hash=CHUNK_HASH,
        model_version=EmbedSettings().model_version,
        locale="us",
        text=f"text of chunk {number}",
        **fields,
    )
    dense = [float(number), 1.0, 0.5, 0.25][:TEST_DIM]
    return IndexedChunk(ident(number), dense, [1, number + 10], [1.0, 2.0], payload)


def index(settings: EmbedSettings, client: AsyncQdrantClient) -> ChunkIndex:
    return ChunkIndex(client, settings.qdrant_collection)


def state_filter(state: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="state", match=models.MatchValue(value=state))]
    )


async def hits(client: AsyncQdrantClient, name: str, state: str) -> list[uuid.UUID]:
    # both kinds of search, as the query service will run them, always with a state filter
    dense = await client.query_points(
        name, query=[1.0, 1.0, 0.5, 0.25], using="dense", query_filter=state_filter(state)
    )
    sparse = await client.query_points(
        name,
        query=models.SparseVector(indices=[1], values=[1.0]),
        using="bm25",
        query_filter=state_filter(state),
    )
    return sorted({uuid.UUID(str(p.id)) for p in dense.points + sparse.points})


# the collection: created once, checked every time, never altered


async def test_ensure_creates_the_collection_with_the_agreed_shape_and_repeats_cleanly(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    info = await qdrant.get_collection(collection.qdrant_collection)
    dense = info.config.params.vectors
    assert isinstance(dense, dict)
    assert (dense["dense"].size, dense["dense"].distance) == (TEST_DIM, models.Distance.COSINE)
    assert info.config.params.sparse_vectors is not None
    assert info.config.params.sparse_vectors["bm25"].modifier == models.Modifier.IDF
    assert info.config.metadata == {"model_version": collection.model_version}
    assert {k: v.data_type.value for k, v in info.payload_schema.items()} == {
        "state": "keyword",
        "doc_id": "keyword",
        "locale": "keyword",
        "version": "integer",
    }
    # the fixture already created it, so a second call finds it right and changes nothing
    assert await ensure_collection(qdrant, collection) is False


async def test_a_new_collection_is_reported_as_created(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    fresh = collection.model_copy(update={"qdrant_collection": collection.qdrant_collection + "x"})
    try:
        assert await ensure_collection(qdrant, fresh) is True
    finally:
        await qdrant.delete_collection(fresh.qdrant_collection)


async def test_verify_accepts_a_collection_built_by_ensure(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    await verify_collection(qdrant, collection)


async def test_a_missing_collection_is_refused_with_the_fix_in_the_message(
    qdrant: AsyncQdrantClient,
) -> None:
    settings = EmbedSettings(qdrant_collection="search_test_nothing_here", dense_dim=TEST_DIM)
    with pytest.raises(CollectionMismatch, match="make init"):
        await verify_collection(qdrant, settings)


@pytest.mark.parametrize(
    ("change", "complaint"),
    [
        pytest.param({"dense_dim": TEST_DIM * 2}, "dense size is 4, expected 8", id="size"),
        pytest.param({"model_revision": 2}, "model_version is", id="model version"),
    ],
)
async def test_a_collection_for_another_model_is_refused_and_left_alone(
    qdrant: AsyncQdrantClient,
    collection: EmbedSettings,
    change: dict[str, Any],
    complaint: str,
) -> None:
    other = collection.model_copy(update=change)
    with pytest.raises(CollectionMismatch, match=complaint):
        await verify_collection(qdrant, other)
    with pytest.raises(CollectionMismatch, match=complaint):
        await ensure_collection(qdrant, other)
    # ensure only checks an existing collection, it never repairs or recreates it
    await verify_collection(qdrant, collection)


@pytest.mark.parametrize(
    ("vectors", "sparse", "complaint"),
    [
        pytest.param(
            {"dense": models.VectorParams(size=TEST_DIM, distance=models.Distance.DOT)},
            {"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            "dense distance",
            id="wrong distance",
        ),
        pytest.param(
            {"dense": models.VectorParams(size=TEST_DIM, distance=models.Distance.COSINE)},
            {"bm25": models.SparseVectorParams()},
            "no idf modifier",
            id="sparse without idf",
        ),
        pytest.param(
            {"dense": models.VectorParams(size=TEST_DIM, distance=models.Distance.COSINE)},
            None,
            "sparse vector is missing",
            id="no sparse vector",
        ),
        pytest.param(
            {"other": models.VectorParams(size=TEST_DIM, distance=models.Distance.COSINE)},
            {"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            "dense vector is missing",
            id="no dense vector",
        ),
    ],
)
async def test_a_collection_with_the_wrong_vector_setup_is_refused(
    qdrant: AsyncQdrantClient,
    vectors: dict[str, models.VectorParams],
    sparse: dict[str, models.SparseVectorParams] | None,
    complaint: str,
) -> None:
    settings = EmbedSettings(
        qdrant_collection=f"search_test_{uuid.uuid4().hex[:10]}", dense_dim=TEST_DIM
    )
    await qdrant.create_collection(
        settings.qdrant_collection,
        vectors_config=vectors,
        sparse_vectors_config=sparse,
        metadata={"model_version": settings.model_version},
    )
    try:
        for field, schema in (
            ("state", models.PayloadSchemaType.KEYWORD),
            ("doc_id", models.PayloadSchemaType.KEYWORD),
            ("locale", models.PayloadSchemaType.KEYWORD),
            ("version", models.PayloadSchemaType.INTEGER),
        ):
            await qdrant.create_payload_index(settings.qdrant_collection, field, schema)
        with pytest.raises(CollectionMismatch, match=complaint):
            await verify_collection(qdrant, settings)
    finally:
        await qdrant.delete_collection(settings.qdrant_collection)


async def test_a_collection_missing_a_payload_index_is_refused(qdrant: AsyncQdrantClient) -> None:
    settings = EmbedSettings(
        qdrant_collection=f"search_test_{uuid.uuid4().hex[:10]}", dense_dim=TEST_DIM
    )
    await qdrant.create_collection(
        settings.qdrant_collection,
        vectors_config={
            "dense": models.VectorParams(size=TEST_DIM, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        metadata={"model_version": settings.model_version},
    )
    try:
        with pytest.raises(CollectionMismatch, match="payload index on state"):
            await verify_collection(qdrant, settings)
    finally:
        await qdrant.delete_collection(settings.qdrant_collection)


# staging: points exist but no reader can see them


async def test_staged_points_hold_both_vectors_and_the_payload_but_no_search_sees_them(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    name = collection.qdrant_collection
    await index(collection, qdrant).stage([chunk(1, brand="Acme", color="black"), chunk(2)])
    stored = await read_points(qdrant, name)
    assert stored[ident(1)] == {
        "doc_id": "P1",
        "version": 1,
        "chunk_idx": 0,
        "content_hash": CHUNK_HASH,
        "model_version": collection.model_version,
        "locale": "us",
        "text": "text of chunk 1",
        "state": "staging",
        "brand": "Acme",
        "color": "black",
    }
    # a missing brand or color is not stored at all
    assert "brand" not in stored[ident(2)]
    assert await hits(qdrant, name, "live") == []
    assert await hits(qdrant, name, "staging") == [ident(1), ident(2)]
    vectors = (await qdrant.retrieve(name, ids=[str(ident(1))], with_vectors=True))[0].vector
    assert isinstance(vectors, dict)
    assert set(vectors) == {"dense", "bm25"}


async def test_staging_again_never_demotes_a_point_that_went_live(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    name = collection.qdrant_collection
    store = index(collection, qdrant)
    await store.stage([chunk(1)])
    await store.publish(1, [LiveChunk(ident(1), 0)], [])
    # the message is delivered again after the point went live, and one new chunk comes with it
    await store.stage([chunk(1), chunk(2)])
    stored = await read_points(qdrant, name)
    assert stored[ident(1)]["state"] == "live"
    assert stored[ident(2)]["state"] == "staging"


async def test_staging_nothing_sends_nothing(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    await index(collection, qdrant).stage([])
    assert await read_points(qdrant, collection.qdrant_collection) == {}


async def test_a_text_with_no_indexable_terms_is_staged_with_an_empty_sparse_vector(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    empty = IndexedChunk(ident(1), [1.0, 0.0, 0.0, 0.0], [], [], chunk(1).payload)
    await index(collection, qdrant).stage([empty])
    assert list(await read_points(qdrant, collection.qdrant_collection)) == [ident(1)]


async def test_a_product_with_many_chunks_is_staged_in_one_call(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    await index(collection, qdrant).stage([chunk(n, idx=n) for n in range(1, 201)])
    assert len(await read_points(qdrant, collection.qdrant_collection, "P1")) == 200


# publishing: live first, then superseded, and safe to run again


async def test_publish_makes_the_new_version_live_and_hides_the_old_one(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    name = collection.qdrant_collection
    store = index(collection, qdrant)
    await store.stage([chunk(1), chunk(2, idx=1)])
    await store.publish(1, [LiveChunk(ident(1), 0), LiveChunk(ident(2), 1)], [])
    # version 2 keeps chunk 1, moves chunk 2 to position 5 and adds chunk 3, chunk 2 is dropped
    await store.stage([chunk(3, version=2, idx=1)])
    await store.publish(2, [LiveChunk(ident(1), 0), LiveChunk(ident(3), 1)], [ident(2)])
    stored = await read_points(qdrant, name, "P1")
    assert {k: (v["state"], v["version"], v["chunk_idx"]) for k, v in stored.items()} == {
        ident(1): ("live", 2, 0),
        ident(2): ("superseded", 1, 1),
        ident(3): ("live", 2, 1),
    }
    assert await hits(qdrant, name, "live") == [ident(1), ident(3)]
    # the text and the product fields are not touched by a publish
    assert stored[ident(1)]["text"] == "text of chunk 1"


async def test_publishing_twice_leaves_the_same_payloads(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    store = index(collection, qdrant)
    await store.stage([chunk(1), chunk(2, idx=1)])
    live = [LiveChunk(ident(1), 0), LiveChunk(ident(2), 1)]
    await store.publish(1, live, [ident(9)])
    once = await read_points(qdrant, collection.qdrant_collection)
    await store.publish(1, live, [ident(9)])
    assert await read_points(qdrant, collection.qdrant_collection) == once


async def test_publish_sets_the_position_of_a_chunk_that_is_already_live(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    # after a crash the postgres rows already hold the new position, so no diff shows the move
    # and the publish itself must carry it
    store = index(collection, qdrant)
    await store.stage([chunk(1, idx=3)])
    await store.publish(1, [LiveChunk(ident(1), 3)], [])
    await store.publish(1, [LiveChunk(ident(1), 0)], [])
    stored = await read_points(qdrant, collection.qdrant_collection)
    assert stored[ident(1)]["chunk_idx"] == 0


async def test_publish_ignores_points_that_no_longer_exist(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    # the compactor may have deleted a superseded point before the message came back
    store = index(collection, qdrant)
    await store.stage([chunk(1)])
    await store.publish(1, [LiveChunk(ident(1), 0), LiveChunk(ident(77), 1)], [ident(88)])
    stored = await read_points(qdrant, collection.qdrant_collection)
    assert list(stored) == [ident(1)]
    assert stored[ident(1)]["state"] == "live"


async def test_publishing_one_product_never_touches_another(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    store = index(collection, qdrant)
    await store.stage([chunk(1, doc_id="P1"), chunk(2, doc_id="P2")])
    await store.publish(1, [LiveChunk(ident(1), 0)], [])
    stored = await read_points(qdrant, collection.qdrant_collection)
    assert (stored[ident(1)]["state"], stored[ident(2)]["state"]) == ("live", "staging")


# deleting: every point of the product goes invisible at once


async def test_a_delete_hides_every_point_of_the_product_in_any_state(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    name = collection.qdrant_collection
    store = index(collection, qdrant)
    await store.stage([chunk(1), chunk(2, idx=1), chunk(3, version=2), chunk(9, doc_id="P2")])
    await store.publish(1, [LiveChunk(ident(1), 0)], [])
    await store.publish(1, [LiveChunk(ident(2), 1)], [ident(1)])  # chunk 1 superseded, 3 staged
    await store.mark_deleted("P1")
    stored = await read_points(qdrant, name)
    assert {k: v["state"] for k, v in stored.items()} == {
        ident(1): "deleted",
        ident(2): "deleted",
        ident(3): "deleted",
        ident(9): "staging",
    }
    assert await hits(qdrant, name, "live") == []
    # a second delete, and a delete of a product that was never indexed, are both fine
    await store.mark_deleted("P1")
    await store.mark_deleted("NEVER")


# failures: a slow or missing server must surface as an error, not as a hang or a silent success


async def test_an_unreachable_server_fails_within_the_timeout(
    collection: EmbedSettings,
) -> None:
    dead = AsyncQdrantClient(url="http://127.0.0.1:9", timeout=1)
    started = time.monotonic()
    with pytest.raises(ResponseHandlingException):
        await index(collection, dead).stage([chunk(1)])
    assert time.monotonic() - started < 5
    await dead.close()


async def test_staging_into_a_missing_collection_is_an_error_not_a_new_collection(
    qdrant: AsyncQdrantClient,
) -> None:
    store = ChunkIndex(qdrant, "search_test_nothing_here")
    with pytest.raises(UnexpectedResponse):
        await store.stage([chunk(1)])
    assert not await qdrant.collection_exists("search_test_nothing_here")
