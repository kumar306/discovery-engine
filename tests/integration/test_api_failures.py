"""The ingest api when things go wrong: every dependency that can say no, and clients racing."""

import asyncio
import io
import json
import logging
import time
from typing import Any

import psycopg
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from libs.observability.logging import JsonFormatter
from services.ingest.repository import DocumentRepository
from services.ingest.settings import IngestSettings

from .conftest import Api, Database, document, make_store, moto_client, outbox, serving
from .test_storage_moto import BlockingClient


class FailingAccept:
    """The real repository for reads, with a compare and set that fails the way the pool would."""

    def __init__(self, real: DocumentRepository, error: BaseException) -> None:
        self._real, self._error = real, error

    async def known(self, corpus: str, doc_id: str) -> Any:
        return await self._real.known(corpus, doc_id)

    async def accept(self, *args: Any) -> Any:
        raise self._error


# s3 and postgres failing


async def test_s3_down_is_a_503_and_postgres_is_untouched(
    pool: AsyncConnectionPool, bucket: str, db: Database
) -> None:
    nothing_listens_here = make_store(moto_client("http://127.0.0.1:9", timeout=0.3), bucket)[0]
    async with serving(pool, bucket, db, store=nothing_listens_here) as api:
        response = await api.post("D1", 1)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    assert response.json()["error"] == "storage_unavailable"
    assert document(db.direct_dsn, "D1") is None
    assert outbox(db.direct_dsn) == []


async def test_postgres_failing_after_the_s3_write_leaves_a_harmless_object_and_a_retry_works(
    pool: AsyncConnectionPool, bucket: str, db: Database
) -> None:
    broken = FailingAccept(DocumentRepository(pool), PoolTimeout("no connection in time"))
    async with serving(pool, bucket, db, repository=broken) as api:
        failed = await api.post("D1", 1)
        assert failed.status_code == 503
        assert failed.json()["error"] == "database_busy"
        assert api.object_exists("esci/D1/v1.json")  # the stray object
        assert document(db.direct_dsn, "D1") is None
        assert outbox(db.direct_dsn) == []
    async with serving(pool, bucket, db) as api:
        assert (await api.post("D1", 1)).status_code == 202  # the retry finishes the job
        assert len(api.events("D1")) == 1


async def test_a_stuck_row_lock_becomes_a_503_within_the_timeout_and_the_retry_works(
    pool: AsyncConnectionPool, bucket: str, db: Database
) -> None:
    settings = IngestSettings(db_lock_timeout_s=0.3)
    async with serving(pool, bucket, db, settings=settings) as api:
        assert (await api.post("D1", 1)).status_code == 202
        stuck = await psycopg.AsyncConnection.connect(db.direct_dsn)
        try:
            await stuck.execute(
                "update documents set latest_version = 2 where corpus = 'esci' and doc_id = 'D1'"
            )
            started = time.monotonic()
            response = await asyncio.wait_for(api.post("D1", 3), timeout=10)
            assert time.monotonic() - started < 3
            assert response.status_code == 503
            assert response.json()["error"] == "database_busy"
            assert response.headers["retry-after"] == "1"
        finally:
            await stuck.rollback()
            await stuck.close()
        assert (await api.post("D1", 3)).status_code == 202


async def test_a_full_s3_bulkhead_sheds_the_next_write_at_once(
    pool: AsyncConnectionPool, bucket: str, db: Database
) -> None:
    blocking = BlockingClient(moto_client())
    store = make_store(blocking, bucket, limit=1)[0]
    async with serving(pool, bucket, db, store=store) as api:
        slow = asyncio.create_task(api.post("SLOW", 1))
        await asyncio.to_thread(blocking.entered.wait, 5)
        # bounded, so a missing bulkhead fails the test instead of waiting for the blocked write
        shed = await asyncio.wait_for(api.post("OTHER", 1), timeout=5)
        assert shed.status_code == 503
        assert shed.json()["error"] == "overloaded"
        assert shed.headers["retry-after"] == "1"
        blocking.release.set()
        assert (await slow).status_code == 202
        assert document(db.direct_dsn, "OTHER") is None  # the shed write left nothing behind


async def test_the_admission_limit_sheds_while_s3_and_postgres_still_have_room(
    pool: AsyncConnectionPool, bucket: str, db: Database
) -> None:
    # the s3 bulkhead has its default permits free and the pool has free connections
    # so when the second request is turned away only the outer admission limit can have done it
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("ingest.request")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    blocking = BlockingClient(moto_client())
    store = make_store(blocking, bucket)[0]
    settings = IngestSettings(max_inflight_requests=1, admission_wait_s=0.01)
    try:
        async with serving(pool, bucket, db, store=store, settings=settings) as api:
            slow = asyncio.create_task(api.post("SLOW", 1))
            await asyncio.to_thread(blocking.entered.wait, 5)
            # bounded, so a missing limit fails the test instead of waiting on the blocked write
            shed = await asyncio.wait_for(api.post("OTHER", 1), timeout=5)
            assert shed.status_code == 503
            assert shed.json()["error"] == "overloaded"
            assert shed.headers["retry-after"] == "1"
            blocking.release.set()
            assert (await slow).status_code == 202
            assert document(db.direct_dsn, "OTHER") is None
    finally:
        blocking.release.set()
        logger.removeHandler(handler)
    outcomes = [json.loads(line)["outcome"] for line in stream.getvalue().splitlines()]
    assert "admission_full" in outcomes  # not storage_overloaded, the s3 bulkhead never filled
    assert "storage_overloaded" not in outcomes


# clients racing, never more than the pool size


async def test_racing_versions_end_at_the_newest_and_every_accepted_write_has_its_event(
    api: Api,
) -> None:
    for round_number in range(5):
        doc_id = f"RACE{round_number}"
        versions = [6, 7, 5]
        responses = await asyncio.gather(
            *(api.post(doc_id, v, title=f"Title {v}") for v in versions)
        )
        by_version = dict(zip(versions, responses, strict=True))
        assert all(r.status_code in {202, 409} for r in responses)  # never a 5xx
        assert by_version[7].status_code == 202  # the newest cannot lose
        row = document(api.database.direct_dsn, doc_id)
        assert row is not None
        assert row["latest"] == 7
        accepted = sorted(v for v, r in by_version.items() if r.status_code == 202)
        assert [event.version for event in api.events(doc_id)] == accepted


async def test_two_writers_of_one_version_with_different_content_get_one_202_and_one_409(
    api: Api,
) -> None:
    for round_number in range(5):
        doc_id = f"SPLIT{round_number}"
        first, second = await asyncio.gather(
            api.post(doc_id, 6, title="Title A"), api.post(doc_id, 6, title="Title B")
        )
        assert sorted([first.status_code, second.status_code]) == [202, 409]
        winner = "Title A" if first.status_code == 202 else "Title B"
        assert winner.encode() in api.object_bytes(f"esci/{doc_id}/v6.json")
        assert len(api.events(doc_id)) == 1


async def test_two_identical_writers_both_get_a_202_and_exactly_one_event_is_queued(
    api: Api,
) -> None:
    for round_number in range(5):
        doc_id = f"TWIN{round_number}"
        first, second = await asyncio.gather(api.post(doc_id, 6), api.post(doc_id, 6))
        assert (first.status_code, second.status_code) == (202, 202)
        assert len(api.events(doc_id)) == 1
