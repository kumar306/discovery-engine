"""The api readiness probes against the real stores: healthy when they answer, an error if not."""

import asyncio

import pytest
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from libs.db.pool import create_pool
from libs.observability.probes import postgres_probe
from libs.resilience.retry import PermanentError, TransientError
from services.ingest.probes import s3_probe

from .conftest import make_store, moto_client


async def test_the_s3_probe_passes_for_a_bucket_that_exists(bucket: str) -> None:
    store, _ = make_store(moto_client(), bucket)
    try:
        await s3_probe(store)()
    finally:
        store.close()


async def test_the_s3_probe_fails_for_a_bucket_that_does_not_exist() -> None:
    store, _ = make_store(moto_client(), "searchtest-no-such-bucket")
    try:
        with pytest.raises(PermanentError, match="404"):
            await s3_probe(store)()
    finally:
        store.close()


async def test_the_s3_probe_fails_when_nothing_listens() -> None:
    store, _ = make_store(moto_client("http://127.0.0.1:9", timeout=0.5), "anything")
    try:
        with pytest.raises(TransientError, match="connection problem"):
            await asyncio.wait_for(s3_probe(store)(), timeout=10)
    finally:
        store.close()


async def test_the_postgres_probe_passes_on_a_working_pool(pool: AsyncConnectionPool) -> None:
    await postgres_probe(pool)()


async def test_the_postgres_probe_fails_when_nothing_listens() -> None:
    dead = create_pool("postgresql://search:x@127.0.0.1:9/search", max_size=1, timeout=0.5)
    await dead.open(wait=False)
    try:
        with pytest.raises(PoolTimeout):
            await asyncio.wait_for(postgres_probe(dead)(), timeout=5)
    finally:
        await dead.close()
