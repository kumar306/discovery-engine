"""The worker process on the real stack: it does the job, and it refuses to start when unfit."""

import asyncio
import uuid

import pytest

from services.embed.embedder import ModelMismatch
from services.embed.qdrant import CollectionMismatch
from services.embed.settings import EmbedSettings
from services.embed.worker import serve
from tests.support.hash_embedder import HashEmbedder

from .conftest import Database, Rig, committed_offsets, eventually, produce


def settings_of(
    rig: Rig, db: Database, bucket: str, topic: str, dlq_topic: str, **changes: object
) -> EmbedSettings:
    return rig.embed_settings.model_copy(
        update={
            "database_dsn": db.pooled_dsn,
            "s3_bucket": bucket,
            "topic_events": topic,
            "topic_dlq": dlq_topic,
            "consumer_group": f"test-{uuid.uuid4().hex[:10]}",
            "consumer_poll_ms": 100,
            **changes,
        }
    )


@pytest.fixture(autouse=True)
def moto_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    # moto accepts any credentials, in a pod these come from the role
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")


class WrongSizeEmbedder(HashEmbedder):
    async def verify(self) -> None:
        raise ModelMismatch("the model returns 768 floats, expected 4")


async def test_the_worker_applies_a_message_and_stops_cleanly_when_told(
    rig: Rig, db: Database, bucket: str, topic: str, dlq_topic: str
) -> None:
    event = await rig.ingest("W1", 1)
    await produce(topic, event.model_dump_json().encode(), b"W1", partition=0)
    settings = settings_of(rig, db, bucket, topic, dlq_topic)
    stop = asyncio.Event()
    task = asyncio.create_task(serve(settings, stop, HashEmbedder(dense_dim=4, max_tokens=64)))
    try:

        async def indexed() -> bool:
            return rig.row("W1")["indexed"] == 1

        await eventually(indexed)
        assert {p["state"] for p in (await rig.points("W1")).values()} == {"live"}
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=15)  # no error, so everything it opened was closed
    assert await committed_offsets(settings.consumer_group) == {0: 1}


async def test_a_collection_built_for_another_model_stops_the_worker_before_it_reads_anything(
    rig: Rig, db: Database, bucket: str, topic: str, dlq_topic: str
) -> None:
    event = await rig.ingest("W1", 1)
    await produce(topic, event.model_dump_json().encode(), b"W1", partition=0)
    settings = settings_of(rig, db, bucket, topic, dlq_topic, model_revision=2)
    with pytest.raises(CollectionMismatch, match="model_version"):
        await asyncio.wait_for(serve(settings, asyncio.Event(), HashEmbedder()), timeout=20)
    assert rig.row("W1")["current"] is None
    assert await committed_offsets(settings.consumer_group) == {}  # not one message was taken


async def test_a_missing_collection_stops_the_worker_and_says_how_to_fix_it(
    rig: Rig, db: Database, bucket: str, topic: str, dlq_topic: str
) -> None:
    settings = settings_of(
        rig, db, bucket, topic, dlq_topic, qdrant_collection="search_test_nothing_here"
    )
    with pytest.raises(CollectionMismatch, match="make init"):
        await asyncio.wait_for(serve(settings, asyncio.Event(), HashEmbedder()), timeout=20)


async def test_a_model_of_the_wrong_size_stops_the_worker_before_it_reads_anything(
    rig: Rig, db: Database, bucket: str, topic: str, dlq_topic: str
) -> None:
    event = await rig.ingest("W1", 1)
    await produce(topic, event.model_dump_json().encode(), b"W1", partition=0)
    settings = settings_of(rig, db, bucket, topic, dlq_topic)
    with pytest.raises(ModelMismatch):
        await asyncio.wait_for(serve(settings, asyncio.Event(), WrongSizeEmbedder()), timeout=20)
    assert await committed_offsets(settings.consumer_group) == {}
