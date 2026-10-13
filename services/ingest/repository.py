from dataclasses import dataclass
from enum import StrEnum

from psycopg_pool import AsyncConnectionPool

from services.ingest.rules import KnownDocument

# strict less than: an equal version never updates the row, so only a newer version can win
# example with two producers sending the same document at once
# producer a updates the row first, postgres stamps that row version with a's transaction id in xmax
# the xmax stamp is the row level lock, a has not committed yet so nobody else can change the row
# producer b reaches the same row and waits on a's transaction id
# when a commits, b rereads the committed row and runs this where against it
# a new document works the same way, b waits on a's uncommitted insert instead of an xmax stamp
_ACCEPT = """
insert into documents (corpus, doc_id, latest_version, latest_content_sha)
values (%s, %s, %s, %s)
on conflict (corpus, doc_id) do update
   set latest_version = excluded.latest_version,
       latest_content_sha = excluded.latest_content_sha,
       updated_at = now()
 where documents.latest_version < excluded.latest_version
returning latest_version
"""

_ADD_TO_OUTBOX = """
insert into outbox (doc_id, request_id, event_schema, payload)
values (%s, %s, %s, %s::jsonb)
"""

_SET_LOCK_TIMEOUT = "select set_config('lock_timeout', %s, true)"

_KNOWN = (
    "select latest_version, latest_content_sha from documents where corpus = %s and doc_id = %s"
)


class Outcome(StrEnum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class Outgoing:
    request_id: str
    event_schema: str
    payload: str


@dataclass(frozen=True)
class AcceptResult:
    outcome: Outcome
    latest_version: int


class DocumentRepository:
    def __init__(self, pool: AsyncConnectionPool, lock_timeout_s: float = 2.0) -> None:
        self._pool = pool
        self._lock_timeout_ms = int(lock_timeout_s * 1000)

    async def known(self, corpus: str, doc_id: str) -> KnownDocument | None:
        async with self._pool.connection() as conn:
            row = await (await conn.execute(_KNOWN, (corpus, doc_id))).fetchone()
        return None if row is None else KnownDocument(row[0], row[1])

    async def accept(
        self,
        corpus: str,
        doc_id: str,
        version: int,
        content_sha: str | None,
        outgoing: Outgoing,
    ) -> AcceptResult:
        async with self._pool.connection() as conn:
            # the document row and its outbox row commit together or not at all
            async with conn.transaction():
                # a writer that waits on a row lock gives up after the timeout instead of hanging
                # set_config with true lasts only for this transaction, a session setting would
                # stay on a pgbouncer server connection and leak into whoever borrows it next
                await conn.execute(_SET_LOCK_TIMEOUT, (f"{self._lock_timeout_ms}ms",))
                # execute sends the query and waits for the reply, the only network trip here
                # psycopg pulls the whole result into this process, so the second await is no trip
                # fetchone only takes the first row from that local copy
                won = await (
                    await conn.execute(_ACCEPT, (corpus, doc_id, version, content_sha))
                ).fetchone()
                # won holds a row only when the upsert changed something
                # a return inside the block commits first and then leaves the whole function
                # an exception inside the block would roll the document and the outbox row back
                if won is not None:
                    await conn.execute(
                        _ADD_TO_OUTBOX,
                        (doc_id, outgoing.request_id, outgoing.event_schema, outgoing.payload),
                    )
                    return AcceptResult(Outcome.ACCEPTED, version)
            # won was empty so the block ended normally and committed a transaction with no changes
            # the where matched no row, so no document write and no outbox row exist
            # the winner's commit is visible now, a plain read tells us why we lost
            # only the losing path pays for this extra read
            row = await (await conn.execute(_KNOWN, (corpus, doc_id))).fetchone()
        if row is None:
            # the row we lost to must exist, nothing ever deletes documents
            raise RuntimeError(f"document {corpus}/{doc_id} vanished after a lost write")
        latest_version, latest_sha = row
        same = latest_version == version and latest_sha == content_sha
        return AcceptResult(Outcome.DUPLICATE if same else Outcome.CONFLICT, latest_version)
