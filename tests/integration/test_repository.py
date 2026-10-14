"""The documents repository: the compare and set that picks the winning write, and its outbox."""

import asyncio
import json
import random
import time

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from libs.db.pool import create_pool
from services.ingest.repository import AcceptResult, DocumentRepository, Outcome, Outgoing
from services.ingest.rules import KnownDocument

from .conftest import APP_POOL_SIZE, Database, document, outbox, wait_until_a_writer_is_blocked

SHA_A = "a" * 64
SHA_B = "b" * 64
ACCEPTED, DUPLICATE, CONFLICT = Outcome.ACCEPTED, Outcome.DUPLICATE, Outcome.CONFLICT


def outgoing(doc_id: str, version: int) -> Outgoing:
    payload = json.dumps({"doc_id": doc_id, "version": version})
    return Outgoing(request_id=f"req-{version}", event_schema="doc_event_v1", payload=payload)


async def accept(
    pool: AsyncConnectionPool,
    doc_id: str,
    version: int,
    sha: str | None = SHA_A,
    corpus: str = "esci",
) -> AcceptResult:
    return await DocumentRepository(pool).accept(
        corpus, doc_id, version, sha, outgoing(doc_id, version)
    )


# the compare and set: first write wins, the rest are judged against the committed row


async def test_an_unknown_document_is_accepted_and_then_known(
    pool: AsyncConnectionPool, db: Database
) -> None:
    repo = DocumentRepository(pool)
    assert await repo.known("esci", "D1") is None
    assert await accept(pool, "D1", 3) == AcceptResult(ACCEPTED, 3)
    assert await repo.known("esci", "D1") == KnownDocument(3, SHA_A)
    row = document(db.direct_dsn, "D1")
    assert row is not None
    assert (row["latest"], row["sha"], row["current"], row["indexed"]) == (3, SHA_A, None, None)


@pytest.mark.parametrize(
    ("existing", "incoming", "outcome", "latest"),
    [
        pytest.param(None, (0, SHA_A), ACCEPTED, 0, id="new document, version zero"),
        pytest.param(None, (7, SHA_A), ACCEPTED, 7, id="new document, any version"),
        pytest.param((5, SHA_A), (6, SHA_B), ACCEPTED, 6, id="higher version, other content"),
        pytest.param((5, SHA_A), (6, SHA_A), ACCEPTED, 6, id="higher version, same content"),
        pytest.param((5, SHA_A), (4, SHA_B), CONFLICT, 5, id="lower version"),
        pytest.param((5, SHA_A), (0, SHA_A), CONFLICT, 5, id="lower version, same content"),
        pytest.param((5, SHA_A), (5, SHA_A), DUPLICATE, 5, id="equal version, same content"),
        pytest.param((5, SHA_A), (5, SHA_B), CONFLICT, 5, id="equal version, other content"),
        pytest.param((5, SHA_A), (6, None), ACCEPTED, 6, id="delete over an upsert"),
        pytest.param((5, None), (5, None), DUPLICATE, 5, id="delete retried"),
        pytest.param((5, None), (5, SHA_A), CONFLICT, 5, id="upsert at a delete's version"),
        pytest.param((5, SHA_A), (5, None), CONFLICT, 5, id="delete at an upsert's version"),
        pytest.param((5, None), (6, SHA_A), ACCEPTED, 6, id="upsert after a delete"),
        pytest.param((5, None), (4, SHA_A), CONFLICT, 5, id="older upsert after a delete"),
    ],
)
async def test_the_write_is_judged_against_what_is_stored(
    pool: AsyncConnectionPool,
    db: Database,
    existing: tuple[int, str | None] | None,
    incoming: tuple[int, str | None],
    outcome: Outcome,
    latest: int,
) -> None:
    if existing is not None:
        await accept(pool, "D1", existing[0], existing[1])
    seeded = len(outbox(db.direct_dsn))
    before = document(db.direct_dsn, "D1")
    result = await accept(pool, "D1", incoming[0], incoming[1])
    assert result == AcceptResult(outcome, latest)
    after = document(db.direct_dsn, "D1")
    assert after is not None
    expected_sha = incoming[1] if outcome is ACCEPTED else existing[1] if existing else None
    assert (after["latest"], after["sha"]) == (latest, expected_sha)
    if outcome is ACCEPTED:
        assert len(outbox(db.direct_dsn)) == seeded + 1
    else:
        # a refused or repeated write leaves no trace: same row, same updated_at, no new event
        assert after == before
        assert len(outbox(db.direct_dsn)) == seeded


async def test_a_delete_of_an_unknown_document_leaves_a_tombstone_without_a_sha(
    pool: AsyncConnectionPool, db: Database
) -> None:
    assert await accept(pool, "D1", 4, None) == AcceptResult(ACCEPTED, 4)
    row = document(db.direct_dsn, "D1")
    assert row is not None
    assert (row["latest"], row["sha"]) == (4, None)
    # ingest only records the newest accepted version, marking it deleted is the worker's job
    assert row["deleted"] is False
    assert await DocumentRepository(pool).known("esci", "D1") == KnownDocument(4, None)


async def test_the_same_doc_id_in_another_corpus_is_another_document(
    pool: AsyncConnectionPool,
) -> None:
    await accept(pool, "D1", 5, corpus="esci")
    assert await accept(pool, "D1", 1, corpus="other") == AcceptResult(ACCEPTED, 1)
    assert await DocumentRepository(pool).known("esci", "D1") == KnownDocument(5, SHA_A)


async def test_accepting_moves_updated_at_but_never_created_at(
    pool: AsyncConnectionPool, db: Database
) -> None:
    await accept(pool, "D1", 1)
    first = document(db.direct_dsn, "D1")
    await accept(pool, "D1", 2)
    second = document(db.direct_dsn, "D1")
    assert first is not None
    assert second is not None
    assert second["created_at"] == first["created_at"]
    assert second["updated_at"] > first["updated_at"]


# the outbox row is written in the same transaction


async def test_the_outbox_row_carries_the_event_and_starts_unpublished(
    pool: AsyncConnectionPool, db: Database
) -> None:
    await accept(pool, "D1", 3)
    ((_, doc_id, request_id, schema, payload, published_at),) = outbox(db.direct_dsn)
    assert (doc_id, request_id, schema) == ("D1", "req-3", "doc_event_v1")
    assert payload == {"doc_id": "D1", "version": 3}
    assert published_at is None


@pytest.mark.parametrize("existing", [False, True], ids=["new document", "existing document"])
async def test_a_failing_outbox_insert_rolls_the_document_write_back(
    pool: AsyncConnectionPool, db: Database, existing: bool
) -> None:
    if existing:
        await accept(pool, "D1", 5)
    seeded = len(outbox(db.direct_dsn))
    before = document(db.direct_dsn, "D1")
    broken = Outgoing(request_id="r", event_schema="doc_event_v1", payload="this is not json")
    with pytest.raises(psycopg.errors.InvalidTextRepresentation):
        await DocumentRepository(pool).accept("esci", "D1", 6, SHA_B, broken)
    assert document(db.direct_dsn, "D1") == before
    assert len(outbox(db.direct_dsn)) == seeded


# concurrency: the row lock decides, whatever the interleaving


async def test_writers_up_to_the_pool_size_lose_nothing_and_never_go_backwards(
    pool: AsyncConnectionPool, db: Database
) -> None:
    rng = random.Random(11)
    for round_number in range(5):
        doc_id = f"RACE{round_number}"
        versions = list(range(1, APP_POOL_SIZE + 1))
        rng.shuffle(versions)
        results = await asyncio.gather(*(accept(pool, doc_id, v) for v in versions))
        by_version = dict(zip(versions, results, strict=True))
        assert all(r.outcome in {ACCEPTED, CONFLICT} for r in results)
        assert by_version[max(versions)].outcome is ACCEPTED  # the newest can never lose
        assert all(r.latest_version >= v for v, r in by_version.items() if r.outcome is CONFLICT)
        row = document(db.direct_dsn, doc_id)
        assert row is not None
        assert row["latest"] == max(versions)
        # events are written under the row lock, so their ids follow the order the writes won in
        events = [payload["version"] for *_, payload, _ in outbox(db.direct_dsn, doc_id)]
        assert events == sorted(events)
        assert len(set(events)) == len(events)
        assert len(events) == sum(r.outcome is ACCEPTED for r in results)


async def test_two_writers_of_one_version_with_different_content_get_one_winner(
    pool: AsyncConnectionPool, db: Database
) -> None:
    for round_number in range(10):
        doc_id = f"SPLIT{round_number}"
        a, b = await asyncio.gather(accept(pool, doc_id, 6, SHA_A), accept(pool, doc_id, 6, SHA_B))
        assert sorted([a.outcome, b.outcome]) == sorted([ACCEPTED, CONFLICT])
        winner_sha = SHA_A if a.outcome is ACCEPTED else SHA_B
        row = document(db.direct_dsn, doc_id)
        assert row is not None
        assert row["sha"] == winner_sha
        assert len(outbox(db.direct_dsn, doc_id)) == 1


async def test_two_identical_writers_get_one_event_and_both_succeed(
    pool: AsyncConnectionPool, db: Database
) -> None:
    for round_number in range(10):
        doc_id = f"TWIN{round_number}"
        a, b = await asyncio.gather(accept(pool, doc_id, 6), accept(pool, doc_id, 6))
        assert sorted([a.outcome, b.outcome]) == sorted([ACCEPTED, DUPLICATE])
        assert len(outbox(db.direct_dsn, doc_id)) == 1


@pytest.mark.parametrize(
    ("existing", "second", "outcome", "latest"),
    [
        pytest.param(True, (6, SHA_A), DUPLICATE, 6, id="existing row, same version and content"),
        pytest.param(True, (6, SHA_B), CONFLICT, 6, id="existing row, same version other content"),
        pytest.param(True, (5, SHA_B), CONFLICT, 6, id="existing row, a lower version"),
        pytest.param(True, (7, SHA_B), ACCEPTED, 7, id="existing row, a higher version"),
        pytest.param(False, (6, SHA_A), DUPLICATE, 6, id="new row, same version and content"),
        pytest.param(False, (6, SHA_B), CONFLICT, 6, id="new row, same version other content"),
    ],
)
async def test_a_writer_waits_for_the_first_one_then_rereads_the_committed_row(
    pool: AsyncConnectionPool,
    db: Database,
    existing: bool,
    second: tuple[int, str],
    outcome: Outcome,
    latest: int,
) -> None:
    # a raw connection plays the first writer and keeps its transaction open on purpose
    if existing:
        await accept(pool, "D1", 5, SHA_B)
    first = await psycopg.AsyncConnection.connect(db.direct_dsn)
    try:
        if existing:
            await first.execute(
                "update documents set latest_version = 6, latest_content_sha = %s "
                "where corpus = 'esci' and doc_id = 'D1'",
                (SHA_A,),
            )
        else:
            await first.execute(
                "insert into documents (corpus, doc_id, latest_version, latest_content_sha) "
                "values ('esci', 'D1', 6, %s)",
                (SHA_A,),
            )
        task = asyncio.create_task(accept(pool, "D1", second[0], second[1]))
        await wait_until_a_writer_is_blocked(db.direct_dsn)
        assert not task.done()  # it is really waiting, not racing
        await first.commit()
    finally:
        await first.close()
    assert await asyncio.wait_for(task, timeout=5) == AcceptResult(outcome, latest)


# the pool: callers that cannot get a connection fail fast, so the api can answer 503


async def test_a_caller_fails_fast_when_every_connection_is_busy(db: Database) -> None:
    tiny = create_pool(db.pooled_dsn, max_size=1, timeout=0.2)
    await tiny.open(wait=True, timeout=10)
    try:
        async with tiny.connection():
            with pytest.raises(PoolTimeout):
                await accept(tiny, "D1", 1)
    finally:
        await tiny.close()


# the lock timeout: a writer stuck behind a row lock gives up instead of hanging


async def test_a_writer_stuck_behind_a_row_lock_gives_up_after_the_timeout(
    pool: AsyncConnectionPool, db: Database
) -> None:
    await accept(pool, "D1", 5)
    first = await psycopg.AsyncConnection.connect(db.direct_dsn)  # holds the row and never commits
    try:
        await first.execute(
            "update documents set latest_version = 6 where corpus = 'esci' and doc_id = 'D1'"
        )
        repo = DocumentRepository(pool, lock_timeout_s=0.3)
        started = time.monotonic()
        # the outer limit keeps a missing timeout from hanging the whole test run
        with pytest.raises(psycopg.errors.LockNotAvailable):
            async with asyncio.timeout(10):
                await repo.accept("esci", "D1", 7, SHA_B, outgoing("D1", 7))
        waited = time.monotonic() - started
        assert 0.25 <= waited < 3  # it did wait for the timeout, and no longer than that
    finally:
        await first.rollback()
        await first.close()
    # nothing was left half done, the same write goes through once the row is free
    assert await accept(pool, "D1", 7, SHA_B) == AcceptResult(ACCEPTED, 7)


async def test_the_lock_timeout_covers_one_transaction_and_is_not_left_on_the_connection(
    db: Database,
) -> None:
    # straight to postgres on purpose: behind pgbouncer the next statement may run elsewhere
    single = create_pool(db.direct_dsn, max_size=1, timeout=2.0)
    await single.open(wait=True, timeout=10)
    try:
        await DocumentRepository(single, lock_timeout_s=0.3).accept(
            "esci", "D1", 1, SHA_A, outgoing("D1", 1)
        )
        async with single.connection() as conn:
            row = await (await conn.execute("show lock_timeout")).fetchone()
        assert row == ("0",)  # the default, so a later transaction never inherits our 300 ms
    finally:
        await single.close()
