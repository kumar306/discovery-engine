from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from services.embed.diff import Positions
from services.embed.gate import GateState, VersionGateDecision, gate

# the document row is the lock for everything the worker changes about one document
# a second worker, or the ingest api writing a newer version, waits here until we commit
_STATE = "select current_version, indexed_version from documents where corpus = %s and doc_id = %s"

_LOCK = _STATE + " for update"

_SET_LOCK_TIMEOUT = "select set_config('lock_timeout', %s, true)"

_CHUNKS = "select chunk_id, positions from chunks where corpus = %s and doc_id = %s"

# a stale row with a deletion_started_at is claimed, a compactor is deleting its point right now
_STALE = """
select chunk_id, deletion_started_at is not null from stale_chunks
 where corpus = %s and doc_id = %s
"""

# the stale rows of the chunks the new version needs, locked, so the check and the revive that
# follows it are one step: a compactor claiming a row waits for this lock or skips the row, and a
# claim that was made before is seen here
_STALE_OF_NEW_CHUNKS = """
select chunk_id, deletion_started_at is not null from stale_chunks
 where corpus = %s and doc_id = %s and chunk_id = any(%s::uuid[])
   for update
"""

# one statement, so the grace period, the claim timeout and the claim itself cannot be separated
# by an ingester that brings the chunk back in between, a revive deletes the row and a claim
# updates it, and whoever locks the row first wins
# the not exists is a second line of defence, a chunk that is current is never deleted
_CLAIM = """
with picked as (
    select s.corpus, s.doc_id, s.chunk_id
      from stale_chunks s
     where s.stale_at < now() - make_interval(secs => %s)
       and (s.deletion_started_at is null
            or s.deletion_started_at < now() - make_interval(secs => %s))
       and not exists (
            select 1 from chunks c
             where c.corpus = s.corpus and c.doc_id = s.doc_id and c.chunk_id = s.chunk_id)
     order by s.stale_at
     limit %s
       for update of s skip locked
)
update stale_chunks s set deletion_started_at = now()
  from picked p
 where s.corpus = p.corpus and s.doc_id = p.doc_id and s.chunk_id = p.chunk_id
returning s.corpus, s.doc_id, s.chunk_id
"""

_FORGET = """
delete from stale_chunks s
  using unnest(%s::text[], %s::text[], %s::uuid[]) as t(corpus, doc_id, chunk_id)
 where s.corpus = t.corpus and s.doc_id = t.doc_id and s.chunk_id = t.chunk_id
   and s.deletion_started_at is not null
"""

_OLDEST_STALE = "select coalesce(extract(epoch from now() - min(stale_at)), 0) from stale_chunks"

# chunks the new version no longer has wait in the stale queue
# the cast matters, an empty python list reaches postgres as an untyped value
_QUEUE_REMOVED = """
insert into stale_chunks (corpus, doc_id, chunk_id)
select corpus, doc_id, chunk_id from chunks
 where corpus = %s and doc_id = %s and chunk_id <> all(%s::uuid[])
on conflict do nothing
"""

_QUEUE_ALL = """
insert into stale_chunks (corpus, doc_id, chunk_id)
select corpus, doc_id, chunk_id from chunks where corpus = %s and doc_id = %s
on conflict do nothing
"""

_DELETE_CHUNKS = "delete from chunks where corpus = %s and doc_id = %s"

_INSERT_CHUNK = """
insert into chunks (corpus, doc_id, chunk_id, content_hash, positions)
values (%s, %s, %s, %s, %s::int[])
"""

# a chunk that came back is live again, so it leaves the stale queue
_REVIVE = """
delete from stale_chunks where corpus = %s and doc_id = %s and chunk_id = any(%s::uuid[])
"""

_SET_COMMITTED = """
update documents set current_version = %s, deleted = false, updated_at = now()
 where corpus = %s and doc_id = %s
"""

_SET_DELETED = """
update documents set current_version = %s, deleted = true, updated_at = now()
 where corpus = %s and doc_id = %s
"""

# indexed only moves forward, and never past what was committed
_MARK_INDEXED = """
update documents set indexed_version = %s, updated_at = now()
 where corpus = %s and doc_id = %s
   and current_version >= %s
   and (indexed_version is null or indexed_version < %s)
returning indexed_version
"""


class PgCommitResult(StrEnum):
    PG_COMMIT_APPLIED = "committed"
    PG_COMMIT_SKIPPED_STALE = "stale"
    # a chunk the version needs is being deleted by the compactor, or one that was counted as
    # coming back has been deleted already, so the diff that led here is no longer true
    PG_COMMIT_CHUNKS_CHANGED = "chunks_changed"


class MissingDocument(RuntimeError):
    """The ingest api creates the document row before any event exists, so it must be there."""


@dataclass(frozen=True)
class NewChunk:
    chunk_id: UUID
    content_hash: str
    positions: Positions


@dataclass(frozen=True)
class Snapshot:
    current: dict[UUID, Positions]
    stale: frozenset[UUID]
    # the stale chunks a compactor has claimed, a subset of stale
    claimed: frozenset[UUID] = frozenset()

    def claimed_among(self, needed: Iterable[UUID]) -> frozenset[UUID]:
        """Which of the chunks a version needs a compactor is about to delete the point of."""
        return self.claimed.intersection(needed)


@dataclass(frozen=True)
class ClaimedChunk:
    corpus: str
    doc_id: str
    chunk_id: UUID


def _gate_state(current: int | None, indexed: int | None) -> GateState | None:
    # a document that was never committed has no state, the gate lets any version through
    if current is None:
        return None
    return GateState(current, indexed is not None and indexed >= current)


class ChunkRepository:
    def __init__(self, pg_pool: AsyncConnectionPool, lock_timeout_s: float = 2.0) -> None:
        self._pg_pool = pg_pool
        self._lock_timeout_ms = int(lock_timeout_s * 1000)

    async def gate_state(self, corpus: str, doc_id: str) -> GateState | None:
        async with self._pg_pool.connection() as conn:
            row = await (await conn.execute(_STATE, (corpus, doc_id))).fetchone()
        return None if row is None else _gate_state(row[0], row[1])

    async def snapshot(self, corpus: str, doc_id: str) -> Snapshot:
        async with self._pg_pool.connection() as conn:
            chunks = await (await conn.execute(_CHUNKS, (corpus, doc_id))).fetchall()
            stale = await (await conn.execute(_STALE, (corpus, doc_id))).fetchall()
        current = {chunk_id: tuple(positions) for chunk_id, positions in chunks}
        return Snapshot(
            current,
            frozenset(chunk_id for chunk_id, _ in stale),
            frozenset(chunk_id for chunk_id, claimed in stale if claimed),
        )

    async def commit_version(
        self,
        corpus: str,
        doc_id: str,
        version: int,
        chunks: Sequence[NewChunk],
        revived: Sequence[UUID] = (),
    ) -> PgCommitResult:
        ids = [chunk.chunk_id for chunk in chunks]
        async with self._pg_pool.connection() as conn, conn.transaction():
            if not await self._lock_and_judge(conn, corpus, doc_id, version):
                return PgCommitResult.PG_COMMIT_SKIPPED_STALE
            # the race between the ingester and the compactor ends here
            # the diff was made before this lock, then came the s3 read, the embedding and the
            # staging, and the compactor works without the document lock, so it can claim a stale
            # chunk or finish deleting one at any moment in between
            # this check is the one that counts, the one before the embedding only saves work
            # nothing has been changed yet, so refusing is just a return, the transaction ends
            # empty and releases its locks, and the ingester backs off and tries the event again
            # no stamp of ours could save a chunk that is already claimed, its delete is in
            # flight at qdrant, a chunk that is not claimed is saved by the revive below, which
            # deletes its stale row under the same lock this check took
            if not await self._stale_rows_agree(conn, corpus, doc_id, ids, revived):
                return PgCommitResult.PG_COMMIT_CHUNKS_CHANGED
            # why replace all, not a delta computed in python from diff_chunks
            # that delta was computed before this lock was held, so it could describe rows
            # that changed while we waited, here the sql reads the rows it has locked, so the
            # removed and the revived chunks always match the table
            # a delta needs the same round trips (delete removed, insert added, update moved)
            # and more code to defend, a product has few chunks, so rewriting them all is cheap
            # and a replay of the same version rewrites identical rows
            # the stale queue keeps the points no version needs, so a crash before the
            # compactor loses nothing
            await conn.execute(_QUEUE_REMOVED, (corpus, doc_id, ids))
            await conn.execute(_DELETE_CHUNKS, (corpus, doc_id))
            # executemany sends one insert per chunk through a pipeline, and one bad chunk fails
            # them all, postgres skips the rest of the pipeline after the error and psycopg raises
            # probed with a bad second row, the first and the third row were both absent after
            # that, so no chunk is ever stored without the others
            # the error leaves this transaction aborted, nothing more can run in it, and leaving
            # the block rolls back the queue, the delete and the version as well
            async with conn.cursor() as cursor:
                await cursor.executemany(
                    _INSERT_CHUNK,
                    [
                        (corpus, doc_id, c.chunk_id, c.content_hash, list(c.positions))
                        for c in chunks
                    ],
                )
            await conn.execute(_REVIVE, (corpus, doc_id, ids))
            await conn.execute(_SET_COMMITTED, (version, corpus, doc_id))
        return PgCommitResult.PG_COMMIT_APPLIED

    async def commit_delete(self, corpus: str, doc_id: str, version: int) -> PgCommitResult:
        async with self._pg_pool.connection() as conn, conn.transaction():
            if not await self._lock_and_judge(conn, corpus, doc_id, version):
                return PgCommitResult.PG_COMMIT_SKIPPED_STALE
            await conn.execute(_QUEUE_ALL, (corpus, doc_id))
            await conn.execute(_DELETE_CHUNKS, (corpus, doc_id))
            await conn.execute(_SET_DELETED, (version, corpus, doc_id))
        return PgCommitResult.PG_COMMIT_APPLIED

    async def claim_stale(
        self, *, grace_s: float, claim_timeout_s: float, limit: int
    ) -> list[ClaimedChunk]:
        """Take the stale rows that are due, then delete their points, then forget the rows."""
        async with self._pg_pool.connection() as conn:
            rows = await (await conn.execute(_CLAIM, (grace_s, claim_timeout_s, limit))).fetchall()
        return [ClaimedChunk(corpus, doc_id, chunk_id) for corpus, doc_id, chunk_id in rows]

    async def forget_claimed(self, claimed: Sequence[ClaimedChunk]) -> None:
        """Remove claimed rows once their points are gone, a row nobody claimed is left alone."""
        if not claimed:
            return
        params = (
            [c.corpus for c in claimed],
            [c.doc_id for c in claimed],
            [c.chunk_id for c in claimed],
        )
        async with self._pg_pool.connection() as conn:
            await conn.execute(_FORGET, params)

    async def oldest_stale_age_s(self) -> float:
        async with self._pg_pool.connection() as conn:
            row = await (await conn.execute(_OLDEST_STALE)).fetchone()
        return float(row[0]) if row else 0.0

    async def mark_indexed(self, corpus: str, doc_id: str, version: int) -> bool:
        async with self._pg_pool.connection() as conn:
            row = await (
                await conn.execute(_MARK_INDEXED, (version, corpus, doc_id, version, version))
            ).fetchone()
        return row is not None

    async def _stale_rows_agree(
        self,
        conn: AsyncConnection,
        corpus: str,
        doc_id: str,
        ids: Sequence[UUID],
        revived: Sequence[UUID],
    ) -> bool:
        # the rows are locked here, so a compactor that is claiming one of them at this moment
        # either committed its claim first, and then this select sees it, or it skips the row
        # and finds it deleted by the revive after this commit, either way it cannot slip in
        # between the check and the revive
        rows = await (
            await conn.execute(_STALE_OF_NEW_CHUNKS, (corpus, doc_id, list(ids)))
        ).fetchall()
        # case one: a compactor claimed a chunk this version needs, after the early check
        # its point is about to be deleted, so this version cannot use it
        if any(claimed for _, claimed in rows):
            return False
        waiting = {chunk_id for chunk_id, _ in rows}
        # case two: a chunk the diff counted as coming back has no stale row any more, so a
        # compactor claimed, deleted and forgot it after the diff, and its point is gone
        return set(revived) <= waiting

    async def _lock_and_judge(
        self, conn: AsyncConnection, corpus: str, doc_id: str, version: int
    ) -> bool:
        # a waiter gives up after the timeout instead of hanging, and set_config with true
        # lasts only for this transaction so it never leaks onto a pgbouncer server connection
        await conn.execute(_SET_LOCK_TIMEOUT, (f"{self._lock_timeout_ms}ms",))
        row = await (await conn.execute(_LOCK, (corpus, doc_id))).fetchone()
        if row is None:
            raise MissingDocument(f"document {corpus}/{doc_id} does not exist")
        # the same pure gate as before the work started, judged again now that we hold the lock
        # a worker that waited here sees the committed row of whoever held it before
        return (
            gate(version, _gate_state(row[0], row[1])) is VersionGateDecision.VERSION_GATE_PROCEED
        )
