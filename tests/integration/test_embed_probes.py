"""The readiness probes against the real stores: healthy when they answer, an error when not."""

import asyncio

import pytest
from psycopg_pool import AsyncConnectionPool, PoolTimeout
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from libs.db.pool import create_pool
from services.embed.consumer import kafka_consumer
from services.embed.probes import KafkaAssignmentProbe, postgres_probe, qdrant_probe
from services.embed.settings import EmbedSettings

from .conftest import Database


async def test_postgres_probe_passes_on_a_pool_that_reaches_the_database(
    pool: AsyncConnectionPool,
) -> None:
    await postgres_probe(pool)()


async def test_postgres_probe_fails_when_nothing_listens() -> None:
    dead = create_pool("postgresql://search:x@127.0.0.1:9/search", max_size=1, timeout=0.5)
    await dead.open(wait=False)
    try:
        with pytest.raises(PoolTimeout):
            await asyncio.wait_for(postgres_probe(dead)(), timeout=5)
    finally:
        await dead.close()


async def test_qdrant_probe_passes_for_the_collection_of_the_worker(
    qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> None:
    await qdrant_probe(qdrant, collection.qdrant_collection)()


async def test_qdrant_probe_fails_for_a_collection_that_is_not_there(
    qdrant: AsyncQdrantClient,
) -> None:
    with pytest.raises(UnexpectedResponse, match="Not found"):
        await qdrant_probe(qdrant, "no_such_collection_for_the_probe_test")()


async def test_qdrant_probe_fails_when_nothing_listens() -> None:
    dead = AsyncQdrantClient(url="http://127.0.0.1:9", timeout=1)
    try:
        with pytest.raises(ResponseHandlingException):
            await asyncio.wait_for(qdrant_probe(dead, "chunks_esci")(), timeout=5)
    finally:
        await dead.close()


async def test_kafka_probe_passes_once_the_group_has_given_partitions(
    topic: str, dlq_topic: str, db: Database
) -> None:
    settings = EmbedSettings(
        topic_events=topic, topic_dlq=dlq_topic, consumer_group=f"probe-{topic[-8:]}"
    )
    probe = KafkaAssignmentProbe()
    with pytest.raises(RuntimeError, match="not started"):
        await probe()
    async with kafka_consumer(settings) as consumer:
        # starting the consumer joins the group, so the partitions are there as soon as it is up
        probe.attach(consumer)
        await probe()
