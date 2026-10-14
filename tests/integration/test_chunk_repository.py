"""The worker's Postgres side: the gate under a row lock, replacing chunks, the stale queue."""

import asyncio
import time
import uuid
from typing import Any

import psycopg
import pytest
from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from services.embed.gate import GateState
from services.embed.repository import (
    ChunkRepository,
    CommitOutcome,
    MissingDocument,
    NewChunk,
)

from .conftest import Database, document, wait_until_a_writer_is_blocked

COMMITTED, STALE = CommitOutcome.COMMITTED, CommitOutcome.STALE


def chunk(number: int, *positions: int) -> NewChunk:
    # the number makes a readable, repeatable chunk id and content hash
    return NewChunk(uuid.UUID(int=number), f"{number:064x}", positions or (number,))


def seed(dsn: str, doc_id: str = "D1", **values: Any) -> None:
    row: dict[str, Any] = {"corpus": "esci", "doc_id": doc_id, "latest_version": 9}
    row.update(values)
    statement = sql.SQL("insert into documents ({}) values ({})").format(
        sql.SQL(", ").join(sql.Identifier(name) for name in row),
        sql.SQL(", ").join(sql.Placeholder() for _ in row),
    )
    with psycopg.connect(dsn) as conn:
        conn.execute(statement, list(row.values()))


def stored(dsn: str, doc_id: str = "D1") -> list[tuple[uuid.UUID, str, tuple[int, ...]]]:
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "select chunk_id, content_hash, positions from chunks "
            "where corpus = 'esci' and doc_id = %s order by positions[1], chunk_id",
            (doc_id,),
        ).fetchall()
    return [
        (chunk_id, content_hash, tuple(positions)) for chunk_id, content_hash, positions in rows
    ]


def stale(dsn: str, doc_id: str = "D1") -> set[uuid.UUID]:
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "select chunk_id from stale_chunks where corpus = 'esci' and doc_id = %s", (doc_id,)
        ).fetchall()
    return {chunk_id for (chunk_id,) in rows}


def ids(*numbers: int) -> set[uuid.UUID]:
    return {uuid.UUID(int=number) for number in numbers}


def repo(pool: AsyncConnectionPool, lock_timeout_s: float = 2.0) -> ChunkRepository:
    return ChunkRepository(pool, lock_timeout_s)


# what the gate is shown before any work starts


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        pytest.param(None, None, id="no such document"),
        pytest.param({}, None, id="accepted by the api, nothing committed"),
        pytest.param({"current_version": 5}, GateState(5, False), id="committed, not live yet"),
        pytest.param(
            {"current_version": 5, "indexed_version": 4}, GateState(5, False), id="live is behind"
        ),
        pytest.param(
            {"current_version": 5, "indexed_version": 5}, GateState(5, True), id="live is current"
        ),
    ],
)
async def test_the_gate_state_tells_how_far_the_document_got(
    pool: AsyncConnectionPool, db: Database, values: dict[str, int] | None, expected: GateState
) -> None:
    if values is not None:
        seed(db.direct_dsn, **values)
    assert await repo(pool).gate_state("esci", "D1") == expected


async def test_the_snapshot_is_empty_for_a_document_without_chunks(
    pool: AsyncConnectionPool, db: Database
) -> None:
    seed(db.direct_dsn)
    snapshot = await repo(pool).snapshot("esci", "D1")
    assert snapshot.current == {}
    assert snapshot.stale == frozenset()


async def test_the_snapshot_returns_every_position_of_a_chunk_and_the_stale_ids(
    pool: AsyncConnectionPool, db: Database
) -> None:
    seed(db.direct_dsn)
    await repo(pool).commit_version("esci", "D1", 1, [chunk(1, 0, 2), chunk(2, 1)])
    await repo(pool).commit_version("esci", "D1", 2, [chunk(1, 0, 2), chunk(3, 1)])
    snapshot = await repo(pool).snapshot("esci", "D1")
    # a repeated text is one chunk at several positions, the diff compares these tuples exactly
    assert snapshot.current == {uuid.UUID(int=1): (0, 2), uuid.UUID(int=3): (1,)}
    assert snapshot.stale == frozenset(ids(2))


# committing a version


async def test_the_first_commit_stores_the_chunks_and_moves_only_the_current_version(
    pool: AsyncConnectionPool, db: Database
) -> None:
    seed(db.direct_dsn)
    assert (
        await repo(pool).commit_version("esci", "D1", 3, [chunk(1, 0), chunk(2, 1, 4)]) == COMMITTED
    )
    assert stored(db.direct_dsn) == [
        (uuid.UUID(int=1), f"{1:064x}", (0,)),
        (uuid.UUID(int=2), f"{2:064x}", (1, 4)),
    ]
    row = document(db.direct_dsn, "D1")
    assert row is not None
    assert (row["current"], row["indexed"], row["deleted"]) == (3, None, False)
    assert stale(db.direct_dsn) == set()


async def test_an_update_replaces_the_chunks_and_queues_the_ones_it_dropped(
    pool: AsyncConnectionPool, db: Database
) -> None:
    seed(db.direct_dsn)
    await repo(pool).commit_version("esci", "D1", 1, [chunk(1, 0), chunk(2, 1), chunk(3, 2)])
    # chunk 1 stays, chunk 2 is dropped, chunk 3 moves, chunk 4 is new
    await repo(pool).commit_version("esci", "D1", 2, [chunk(1, 0), chunk(4, 1), chunk(3, 2, 3)])
    assert [(c[0], c[2]) for c in stored(db.direct_dsn)] == [
        (uuid.UUID(int=1), (0,)),
        (uuid.UUID(int=4), (1,)),
        (uuid.UUID(int=3), (2, 3)),
    ]
    assert stale(db.direct_dsn) == ids(2)
    row = document(db.direct_dsn, "D1")
    assert row is not None
    assert row["current"] == 2


async def test_a_chunk_that_comes_back_leaves_the_stale_queue(
    pool: AsyncConnectionPool, db: Database
) -> None:
    seed(db.direct_dsn)
    await repo(pool).commit_version("esci", "D1", 1, [chunk(1, 0), chunk(2, 1)])
    await repo(pool).commit_version("esci", "D1", 2, [chunk(1, 0)])
    assert stale(db.direct_dsn) == ids(2)
    await repo(pool).commit_version("esci", "D1", 3, [chunk(1, 0), chunk(2, 1)])
    assert stale(db.direct_dsn) == set()
    assert {c[0] for c in stored(db.direct_dsn)} == ids(1, 2)


async def test_committing_the_same_version_again_changes_nothing(
    pool: AsyncConnectionPool, db: Database
) -> None:
    # a crash after the commit and before the offset: the message comes back, the gate lets it in
    seed(db.direct_dsn)
    await repo(pool).commit_version("esci", "D1", 1, [chunk(1, 0), chunk(2, 1)])
    await repo(pool).commit_version("esci", "D1", 2, [chunk(1, 0), chunk(3, 1)])
    before = (stored(db.direct_dsn), stale(db.direct_dsn))
    assert await repo(pool).commit_version("esci", "D1", 2, [chunk(1, 0), chunk(3, 1)]) == COMMITTED
    assert (stored(db.direct_dsn), stale(db.direct_dsn)) == before


@pytest.mark.parametrize(
    ("current", "indexed", "version", "expected"),
    [
        pytest.param(None, None, 1, COMMITTED, id="never committed"),
        pytest.param(5, 5, 6, COMMITTED, id="newer than live"),
        pytest.param(5, None, 5, COMMITTED, id="same version, committed but not live yet"),
        pytest.param(5, 4, 5, COMMITTED, id="same version, live is behind"),
        pytest.param(5, 5, 5, STALE, id="same version, already live"),
        pytest.param(5, 5, 4, STALE, id="older than live"),
        pytest.param(5, None, 4, STALE, id="older than committed"),
    ],
)
async def test_the_gate_is_judged_again_under_the_lock(
    pool: AsyncConnectionPool,
    db: Database,
    current: int | None,
    indexed: int | None,
    version: int,
    expected: CommitOutcome,
) -> None:
    seed(db.direct_dsn, current_version=current, indexed_version=indexed)
    outcome = await repo(pool).commit_version("esci", "D1", version, [chunk(7, 0)])
    assert outcome == expected
    row = document(db.direct_dsn, "D1")
    assert row is not None
    if expected is STALE:
        # a stale event leaves no trace at all
        assert stored(db.direct_dsn) == []
        assert (row["current"], row["indexed"]) == (current, indexed)
    else:
        assert row["current"] == version
        assert [c[0] for c in stored(db.direct_dsn)] == [uuid.UUID(int=7)]


async def test_a_document_the_api_never_created_is_an_error_and_writes_nothing(
    pool: AsyncConnectionPool, db: Database
) -> None:
    with pytest.raises(MissingDocument, match="esci/GHOST"):
        await repo(pool).commit_version("esci", "GHOST", 1, [chunk(1, 0)])
    assert stored(db.direct_dsn, "GHOST") == []


async def test_a_failure_in_the_middle_of_a_commit_rolls_everything_back(
    pool: AsyncConnectionPool, db: Database
) -> None:
    seed(db.direct_dsn)
    await repo(pool).commit_version("esci", "D1", 1, [chunk(1, 0), chunk(2, 1)])
    before = (stored(db.direct_dsn), stale(db.direct_dsn), document(db.direct_dsn, "D1"))
    # the second chunk has no position, the database refuses it after the first statements ran
    with pytest.raises(psycopg.errors.CheckViolation):
        await repo(pool).commit_version(
            "esci", "D1", 2, [chunk(3, 0), NewChunk(chunk(4).chunk_id, chunk(4).content_hash, ())]
        )
    assert (stored(db.direct_dsn), stale(db.direct_dsn), document(db.direct_dsn, "D1")) == before


# two workers, or a worker and the api, on one document


@pytest.mark.parametrize(
    ("holder_sql", "expected", "chunks_after"),
    [
        pytest.param(
            "update documents set current_version = 3, indexed_version = 3 "
            "where corpus = 'esci' and doc_id = 'D1'",
            STALE,
            0,
            id="the holder commits a newer version, so we are stale",
        ),
        pytest.param(
            "update documents set latest_version = 10 where corpus = 'esci' and doc_id = 'D1'",
            COMMITTED,
            1,
            id="the holder only accepts a newer upload, we still go ahead",
        ),
    ],
)
async def test_a_commit_that_waited_for_the_lock_judges_the_row_the_holder_committed(
    pool: AsyncConnectionPool,
    db: Database,
    holder_sql: str,
    expected: CommitOutcome,
    chunks_after: int,
) -> None:
    seed(db.direct_dsn)
    holder = await psycopg.AsyncConnection.connect(db.direct_dsn)
    try:
        await holder.execute(holder_sql)
        task = asyncio.create_task(
            repo(pool, lock_timeout_s=5).commit_version("esci", "D1", 2, [chunk(1, 0)])
        )
        await wait_until_a_writer_is_blocked(db.direct_dsn)
        assert not task.done()  # it is really waiting, not racing
        await holder.commit()
    finally:
        await holder.close()
    assert await asyncio.wait_for(task, timeout=5) == expected
    assert len(stored(db.direct_dsn)) == chunks_after


async def test_a_commit_stuck_behind_a_row_lock_gives_up_after_the_timeout(
    pool: AsyncConnectionPool, db: Database
) -> None:
    seed(db.direct_dsn)
    holder = await psycopg.AsyncConnection.connect(db.direct_dsn)  # holds the row, never commits
    try:
        await holder.execute("select 1 from documents where doc_id = 'D1' for update")
        started = time.monotonic()
        with pytest.raises(psycopg.errors.LockNotAvailable):
            async with asyncio.timeout(10):
                await repo(pool, lock_timeout_s=0.3).commit_version("esci", "D1", 1, [chunk(1, 0)])
        assert 0.25 <= time.monotonic() - started < 3
    finally:
        await holder.rollback()
        await holder.close()
    assert stored(db.direct_dsn) == []
    # nothing was left half done, the same commit goes through once the row is free
    assert await repo(pool).commit_version("esci", "D1", 1, [chunk(1, 0)]) == COMMITTED


async def test_workers_on_different_documents_never_wait_for_each_other(
    pool: AsyncConnectionPool, db: Database
) -> None:
    seed(db.direct_dsn, "D1")
    seed(db.direct_dsn, "D2")
    holder = await psycopg.AsyncConnection.connect(db.direct_dsn)
    try:
        await holder.execute("select 1 from documents where doc_id = 'D1' for update")
        outcome = await asyncio.wait_for(
            repo(pool, lock_timeout_s=5).commit_version("esci", "D2", 1, [chunk(1, 0)]), timeout=3
        )
    finally:
        await holder.rollback()
        await holder.close()
    assert outcome == COMMITTED


# deletes


async def test_a_delete_queues_every_chunk_and_leaves_a_tombstone(
    pool: AsyncConnectionPool, db: Database
) -> None:
    seed(db.direct_dsn)
    await repo(pool).commit_version("esci", "D1", 1, [chunk(1, 0), chunk(2, 1)])
    assert await repo(pool).commit_delete("esci", "D1", 2) == COMMITTED
    assert stored(db.direct_dsn) == []
    assert stale(db.direct_dsn) == ids(1, 2)
    row = document(db.direct_dsn, "D1")
    assert row is not None
    assert (row["current"], row["indexed"], row["deleted"]) == (2, None, True)


async def test_a_delete_of_a_document_that_was_never_indexed_still_leaves_the_tombstone(
    pool: AsyncConnectionPool, db: Database
) -> None:
    seed(db.direct_dsn, latest_version=4)
    assert await repo(pool).commit_delete("esci", "D1", 4) == COMMITTED
    row = document(db.direct_dsn, "D1")
    assert row is not None
    assert (row["current"], row["deleted"]) == (4, True)
    assert stale(db.direct_dsn) == set()


async def test_a_delayed_older_upsert_cannot_bring_a_deleted_document_back(
    pool: AsyncConnectionPool, db: Database
) -> None:
    seed(db.direct_dsn)
    await repo(pool).commit_version("esci", "D1", 1, [chunk(1, 0)])
    await repo(pool).commit_delete("esci", "D1", 3)
    assert await repo(pool).commit_version("esci", "D1", 2, [chunk(9, 0)]) == STALE
    assert stored(db.direct_dsn) == []
    row = document(db.direct_dsn, "D1")
    assert row is not None
    assert (row["current"], row["deleted"]) == (3, True)


async def test_a_newer_upload_after_a_delete_brings_the_document_back(
    pool: AsyncConnectionPool, db: Database
) -> None:
    seed(db.direct_dsn)
    await repo(pool).commit_version("esci", "D1", 1, [chunk(1, 0)])
    await repo(pool).commit_delete("esci", "D1", 2)
    assert await repo(pool).commit_version("esci", "D1", 3, [chunk(1, 0), chunk(2, 1)]) == COMMITTED
    row = document(db.direct_dsn, "D1")
    assert row is not None
    assert (row["current"], row["deleted"]) == (3, False)
    assert {c[0] for c in stored(db.direct_dsn)} == ids(1, 2)
    assert stale(db.direct_dsn) == set()  # chunk 1 was revived


@pytest.mark.parametrize(
    ("indexed", "expected"),
    [
        pytest.param(None, COMMITTED, id="the delete was committed but not published"),
        pytest.param(2, STALE, id="the delete is already live"),
    ],
)
async def test_the_same_delete_delivered_again_follows_the_gate(
    pool: AsyncConnectionPool, db: Database, indexed: int | None, expected: CommitOutcome
) -> None:
    seed(db.direct_dsn, current_version=2, indexed_version=indexed, deleted=True)
    assert await repo(pool).commit_delete("esci", "D1", 2) == expected


# marking a version live


async def test_marking_indexed_follows_the_commit_and_a_repeat_does_nothing(
    pool: AsyncConnectionPool, db: Database
) -> None:
    seed(db.direct_dsn)
    await repo(pool).commit_version("esci", "D1", 3, [chunk(1, 0)])
    assert await repo(pool).mark_indexed("esci", "D1", 3) is True
    assert await repo(pool).mark_indexed("esci", "D1", 3) is False
    row = document(db.direct_dsn, "D1")
    assert row is not None
    assert (row["current"], row["indexed"]) == (3, 3)


@pytest.mark.parametrize(
    ("current", "indexed", "version"),
    [
        pytest.param(5, 5, 4, id="never goes backwards"),
        pytest.param(3, None, 4, id="never goes past what was committed"),
        pytest.param(None, None, 1, id="needs a commit first"),
    ],
)
async def test_marking_indexed_refuses_what_would_break_the_ordering(
    pool: AsyncConnectionPool, db: Database, current: int | None, indexed: int | None, version: int
) -> None:
    seed(db.direct_dsn, current_version=current, indexed_version=indexed)
    assert await repo(pool).mark_indexed("esci", "D1", version) is False
    row = document(db.direct_dsn, "D1")
    assert row is not None
    assert (row["current"], row["indexed"]) == (current, indexed)
