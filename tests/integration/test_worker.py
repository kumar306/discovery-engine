"""The worker process on the real stack: it does the job, and it refuses to start when unfit."""

import asyncio
import time
import uuid
from collections.abc import Sequence

import httpx
import pytest

from services.embed.embedder import Embedding, EmbedTransient, ModelMismatch
from services.embed.qdrant import CollectionMismatch
from services.embed.settings import EmbedSettings
from services.embed.worker import serve
from tests.support.hash_embedder import HashEmbedder

from .conftest import Database, Rig, committed_offsets, eventually, free_port, produce


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
            # every worker has its own health port, so tests never meet on one
            "health_port": free_port(),
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


# the health endpoints and the metrics of a running worker


async def get(settings: EmbedSettings, path: str) -> httpx.Response:
    async with httpx.AsyncClient(timeout=5) as client:
        return await client.get(f"http://127.0.0.1:{settings.health_port}{path}")


async def ready(settings: EmbedSettings) -> bool:
    try:
        return (await get(settings, "/readyz")).status_code == 200
    except httpx.TransportError:  # the server is not up yet
        return False


async def metric(settings: EmbedSettings, line: str) -> bool:
    try:
        return line in (await get(settings, "/metrics")).text
    except httpx.TransportError:  # the server is not up yet
        return False


class DownEmbedder(HashEmbedder):
    """The model is up but the step that uses it keeps failing: a transient error, every time."""

    async def embed(self, texts: Sequence[str], timeout_s: float) -> list[Embedding]:
        raise EmbedTransient("the embedder is not answering")


class SlowEmbedder(HashEmbedder):
    def __init__(self, seconds: float | None) -> None:
        super().__init__(dense_dim=4, max_tokens=64)
        self.started = asyncio.Event()
        self._seconds = seconds  # None: never returns

    async def embed(self, texts: Sequence[str], timeout_s: float) -> list[Embedding]:
        self.started.set()
        if self._seconds is None:
            await asyncio.Event().wait()
        else:
            await asyncio.sleep(self._seconds)
        return await super().embed(texts, timeout_s)


async def test_the_worker_becomes_ready_and_counts_what_it_does(
    rig: Rig, db: Database, bucket: str, topic: str, dlq_topic: str
) -> None:
    event = await rig.ingest("H1", 1)
    await produce(topic, event.model_dump_json().encode(), b"H1", partition=0)
    settings = settings_of(rig, db, bucket, topic, dlq_topic)
    stop = asyncio.Event()
    task = asyncio.create_task(serve(settings, stop, HashEmbedder(dense_dim=4, max_tokens=64)))
    try:
        await eventually(lambda: ready(settings))
        body = (await get(settings, "/readyz")).json()
        assert body == {
            "ready": True,
            "checks": {"postgres": "ok", "qdrant": "ok", "kafka": "ok"},
        }
        assert (await get(settings, "/healthz")).status_code == 200

        async def applied() -> bool:
            return await metric(settings, 'embed_events_total{result="applied"} 1.0')

        await eventually(applied)
        assert await metric(settings, "embed_held_partitions 0.0")
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=15)


async def test_an_idle_worker_keeps_beating_so_it_stays_alive_under_a_tight_limit(
    rig: Rig, db: Database, bucket: str, topic: str, dlq_topic: str
) -> None:
    # the limit is three seconds, the default five minutes would hide a heartbeat nobody beats
    settings = settings_of(rig, db, bucket, topic, dlq_topic, liveness_max_age_s=3.0)
    stop = asyncio.Event()
    task = asyncio.create_task(serve(settings, stop, HashEmbedder(dense_dim=4, max_tokens=64)))
    try:
        await eventually(lambda: ready(settings))
        for _ in range(9):  # four and a half seconds, longer than the limit
            assert (await get(settings, "/healthz")).status_code == 200
            await asyncio.sleep(0.5)
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=15)


async def test_a_held_partition_keeps_the_worker_ready_and_raises_the_gauges(
    rig: Rig, db: Database, bucket: str, topic: str, dlq_topic: str
) -> None:
    event = await rig.ingest("H2", 1)
    await produce(topic, event.model_dump_json().encode(), b"H2", partition=0)
    settings = settings_of(rig, db, bucket, topic, dlq_topic)
    stop = asyncio.Event()
    task = asyncio.create_task(serve(settings, stop, DownEmbedder(dense_dim=4, max_tokens=64)))
    try:

        async def held() -> bool:
            return await metric(settings, "embed_held_partitions 1.0")

        await eventually(held)
        # restarting the worker would not fix a failing step, so it stays ready and alive
        assert (await get(settings, "/readyz")).status_code == 200
        assert (await get(settings, "/healthz")).status_code == 200
        text = (await get(settings, "/metrics")).text
        assert "embed_oldest_hold_age_seconds 0.0" not in text  # the hold has an age
        assert 'embed_events_total{result="held"}' in text
        assert rig.row("H2")["current"] is None  # and nothing was committed
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=15)
    assert await committed_offsets(settings.consumer_group) == {}


async def test_a_stop_during_a_message_lets_it_finish_and_commits_it(
    rig: Rig, db: Database, bucket: str, topic: str, dlq_topic: str
) -> None:
    event = await rig.ingest("S1", 1)
    await produce(topic, event.model_dump_json().encode(), b"S1", partition=0)
    settings = settings_of(rig, db, bucket, topic, dlq_topic, shutdown_grace_s=15)
    embedder = SlowEmbedder(seconds=1.0)
    stop = asyncio.Event()
    task = asyncio.create_task(serve(settings, stop, embedder))
    await asyncio.wait_for(embedder.started.wait(), timeout=20)
    stop.set()  # told to stop while the message is being embedded
    await asyncio.wait_for(task, timeout=15)
    assert rig.row("S1")["indexed"] == 1  # it was finished, not abandoned
    assert await committed_offsets(settings.consumer_group) == {0: 1}


async def test_a_stop_during_a_message_that_never_finishes_ends_inside_the_grace_period(
    rig: Rig, db: Database, bucket: str, topic: str, dlq_topic: str
) -> None:
    event = await rig.ingest("S2", 1)
    await produce(topic, event.model_dump_json().encode(), b"S2", partition=0)
    settings = settings_of(rig, db, bucket, topic, dlq_topic, shutdown_grace_s=0.5)
    embedder = SlowEmbedder(seconds=None)
    stop = asyncio.Event()
    task = asyncio.create_task(serve(settings, stop, embedder))
    await asyncio.wait_for(embedder.started.wait(), timeout=20)
    started = time.monotonic()
    stop.set()
    await asyncio.wait_for(task, timeout=15)  # no error: the clients were closed in order
    assert time.monotonic() - started < 5
    # the message was abandoned, so it is still on the topic for the next worker
    assert await committed_offsets(settings.consumer_group) == {}
    assert rig.row("S2")["current"] is None
