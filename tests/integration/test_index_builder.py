"""The index builder against real Qdrant, compared with the real worker pipeline.

The builder is a second way to fill a collection (no Kafka, Postgres or S3). These tests are what
keeps it honest: the same products go through the worker and through the builder, and the two
collections must hold the same points with the same payloads and the same vectors.
"""

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import AsyncQdrantClient, models

from eval.dataset import JudgementRow, Product, QueryRow, Selection, SelectionConfig, write_dataset
from eval.index_builder import build_index
from eval.loader import product_to_request
from services.embed.settings import EmbedSettings
from tests.support.hash_embedder import HashEmbedder

from .conftest import TEST_DIM, Rig, words

REVISION = 1


def sample_products() -> list[Product]:
    return [
        # two chunks: the bullets, then the description
        Product(
            "P0",
            "us",
            "Acme mouse",
            description=words(30, "d"),
            bullet_point=words(20, "a") + "\n" + words(20, "b"),
            brand="Acme",
            color="Black",
        ),
        Product("P1", "us", "Plain pen", None, None, None, None),
        Product("P2", "us", "Desk  lamp", words(10, "x"), None, "Lumen", None),
        Product("P3", "us", "Large gadget", words(60, "g"), words(25, "h"), None, "Red"),
        Product("P4", "us", "Accents café", None, "naïve bullet", None, None),
    ]


def write_sample(directory: Path, items: list[Product]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    selection = Selection(
        (QueryRow(1, "mouse", "validation"),),
        tuple(JudgementRow(1, item.product_id, "E") for item in items),
        tuple(sorted(item.product_id for item in items)),
    )
    write_dataset(directory, selection, items, SelectionConfig(seed=1), REVISION)


async def everything(client: AsyncQdrantClient, name: str) -> dict[uuid.UUID, dict[str, Any]]:
    """Payload and both vectors of every point of the collection, keyed by id."""
    found: dict[uuid.UUID, dict[str, Any]] = {}
    offset = None
    while True:
        points, offset = await client.scroll(
            name, limit=100, offset=offset, with_payload=True, with_vectors=True
        )
        for point in points:
            found[uuid.UUID(str(point.id))] = {"payload": point.payload, "vector": point.vector}
        if offset is None:
            return found


@pytest.fixture
async def builder_settings(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> AsyncIterator[EmbedSettings]:
    settings = collection.model_copy(
        update={"qdrant_collection": f"search_test_built_{uuid.uuid4().hex[:8]}"}
    )
    try:
        yield settings
    finally:
        await qdrant.delete_collection(settings.qdrant_collection)


async def test_the_builder_makes_the_same_collection_as_the_worker(
    rig: Rig, builder_settings: EmbedSettings, tmp_path: Path
) -> None:
    items = sample_products()
    write_sample(tmp_path / "set", items)
    # the worker's way: through the real api, the outbox event, and the pipeline
    for item in items:
        body = product_to_request(item, REVISION)
        fields = {k: v for k, v in body.items() if k not in {"corpus", "doc_id", "version"}}
        event = await rig.ingest(item.product_id, REVISION, **fields)
        await rig.pipeline.handle(event)
    # the builder's way, into a second collection, with the same stub model
    report = await build_index(tmp_path / "set", builder_settings, rig.qdrant_client, rig.embedder)
    worker = await everything(rig.qdrant_client, rig.embed_settings.qdrant_collection)
    built = await everything(rig.qdrant_client, builder_settings.qdrant_collection)
    assert report.chunks == len(worker) == len(built)
    assert set(built) == set(worker)
    for point_id, expected in worker.items():
        assert built[point_id]["payload"] == expected["payload"], point_id
        assert built[point_id]["vector"] == expected["vector"], point_id


async def test_building_twice_leaves_the_collection_as_it_was(
    rig: Rig, builder_settings: EmbedSettings, tmp_path: Path
) -> None:
    write_sample(tmp_path / "set", sample_products())
    await build_index(tmp_path / "set", builder_settings, rig.qdrant_client, rig.embedder)
    first = await everything(rig.qdrant_client, builder_settings.qdrant_collection)
    await build_index(tmp_path / "set", builder_settings, rig.qdrant_client, rig.embedder)
    assert await everything(rig.qdrant_client, builder_settings.qdrant_collection) == first


async def test_a_model_with_another_vector_size_gets_its_own_collection(
    rig: Rig, builder_settings: EmbedSettings, tmp_path: Path
) -> None:
    write_sample(tmp_path / "set", sample_products())
    wide = builder_settings.model_copy(update={"dense_dim": 8})
    await build_index(
        tmp_path / "set", wide, rig.qdrant_client, HashEmbedder(dense_dim=8, max_tokens=64)
    )
    points = await everything(rig.qdrant_client, wide.qdrant_collection)
    assert points
    assert {len(p["vector"]["dense"]) for p in points.values()} == {8}
    assert TEST_DIM != 8


async def test_every_point_is_live_after_a_build(
    rig: Rig, builder_settings: EmbedSettings, tmp_path: Path
) -> None:
    write_sample(tmp_path / "set", sample_products())
    await build_index(tmp_path / "set", builder_settings, rig.qdrant_client, rig.embedder)
    live = models.Filter(
        must=[models.FieldCondition(key="state", match=models.MatchValue(value="live"))]
    )
    everyone = await rig.qdrant_client.count(builder_settings.qdrant_collection, exact=True)
    visible = await rig.qdrant_client.count(
        builder_settings.qdrant_collection, count_filter=live, exact=True
    )
    assert everyone.count == visible.count > 0
