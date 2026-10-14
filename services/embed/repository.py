from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from services.embed.diff import Positions
from services.embed.gate import GateDecision, GateState, gate

# the document row is the lock for everything the worker changes about one document
# a second worker, or the ingest api writing a newer version, waits here until we commit
_STATE = "select current_version, indexed_version from documents where corpus = %s and doc_id = %s"

_LOCK = _STATE + " for update"

_SET_LOCK_TIMEOUT = "select set_config('lock_timeout', %s, true)"

_CHUNKS = "select chunk_id, positions from chunks where corpus = %s and doc_id = %s"

_STALE = "select chunk_id from stale_chunks where corpus = %s and doc_id = %s"

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


class CommitOutcome(StrEnum):
    COMMITTED = "committed"
    STALE = "stale"


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


def _gate_state(current: int | None, indexed: int | None) -> GateState | None:
    # a document that was never committed has no state, the gate lets any version through
    if current is None:
        return None
    return GateState(current, indexed is not None and indexed >= current)


class ChunkRepository:
    def __init__(self, pool: AsyncConnectionPool, lock_timeout_s: float = 2.0) -> None:
        self._pool = pool
        self._lock_timeout_ms = int(lock_timeout_s * 1000)

    async def gate_state(self, corpus: str, doc_id: str) -> GateState | None:
        async with self._pool.connection() as conn:
            row = await (await conn.execute(_STATE, (corpus, doc_id))).fetchone()
        return None if row is None else _gate_state(row[0], row[1])

    async def snapshot(self, corpus: str, doc_id: str) -> Snapshot:
        async with self._pool.connection() as conn:
            chunks = await (await conn.execute(_CHUNKS, (corpus, doc_id))).fetchall()
            stale = await (await conn.execute(_STALE, (corpus, doc_id))).fetchall()
        current = {chunk_id: tuple(positions) for chunk_id, positions in chunks}
        return Snapshot(current, frozenset(chunk_id for (chunk_id,) in stale))

    async def commit_version(
        self, corpus: str, doc_id: str, version: int, chunks: Sequence[NewChunk]
    ) -> CommitOutcome:
        ids = [chunk.chunk_id for chunk in chunks]
        async with self._pool.connection() as conn, conn.transaction():
            if not await self._lock_and_judge(conn, corpus, doc_id, version):
                return CommitOutcome.STALE
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
        return CommitOutcome.COMMITTED

    async def commit_delete(self, corpus: str, doc_id: str, version: int) -> CommitOutcome:
        async with self._pool.connection() as conn, conn.transaction():
            if not await self._lock_and_judge(conn, corpus, doc_id, version):
                return CommitOutcome.STALE
            await conn.execute(_QUEUE_ALL, (corpus, doc_id))
            await conn.execute(_DELETE_CHUNKS, (corpus, doc_id))
            await conn.execute(_SET_DELETED, (version, corpus, doc_id))
        return CommitOutcome.COMMITTED

    async def mark_indexed(self, corpus: str, doc_id: str, version: int) -> bool:
        async with self._pool.connection() as conn:
            row = await (
                await conn.execute(_MARK_INDEXED, (version, corpus, doc_id, version, version))
            ).fetchone()
        return row is not None

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
        return gate(version, _gate_state(row[0], row[1])) is GateDecision.PROCEED
