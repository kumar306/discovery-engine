from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from libs.schemas import UpsertRequest
from services.ingest.events import EVENT_SCHEMA, prepare_delete, prepare_upsert
from services.ingest.repository import (
    AcceptResult,
    DocumentRepository,
    DocumentStatus,
    Outgoing,
    PgWriteResult,
)
from services.ingest.rules import KnownDocument, WritePrecheck, precheck
from services.ingest.storage import DocumentStore, S3PutResult

CORPUS: Literal["esci"] = "esci"


@dataclass(frozen=True)
class WriteAccepted:
    doc_id: str
    version: int
    accepted_at: datetime


@dataclass(frozen=True)
class WriteRefused:
    error: Literal["stale_version", "content_conflict"]
    # the newest version we hold, or None when only a leftover s3 object stands in the way
    current_version: int | None


class IngestService:
    def __init__(
        self,
        pg_documents: DocumentRepository,
        s3_documents: DocumentStore,
        clock: Callable[[], datetime],
    ) -> None:
        self._pg_documents = pg_documents
        self._s3_documents = s3_documents
        self._clock = clock

    async def upsert(self, request: UpsertRequest, request_id: str) -> WriteAccepted | WriteRefused:
        prepared = prepare_upsert(request, produced_at=self._clock())
        known = await self._pg_documents.known(request.corpus, request.doc_id)
        # a cheap early answer, the compare and set in accept is the real guard
        early = self._early_answer(
            precheck(request.version, prepared.content_sha, known),
            known,
            request.doc_id,
            request.version,
        )
        if early is not None:
            return early
        # s3 first, the object is immutable and a leftover from a crash is harmless
        put = await self._s3_documents.put_if_absent(
            prepared.key, prepared.data, prepared.content_sha
        )
        if put is S3PutResult.S3_PUT_CONFLICT:
            return WriteRefused("content_conflict", known.latest_version if known else None)
        outgoing = Outgoing(request_id, EVENT_SCHEMA, prepared.event_json)
        result = await self._pg_documents.accept(
            request.corpus, request.doc_id, request.version, prepared.content_sha, outgoing
        )
        return self._after_accept(result, request.doc_id, request.version)

    async def delete(
        self, doc_id: str, version: int, request_id: str
    ) -> WriteAccepted | WriteRefused:
        prepared = prepare_delete(CORPUS, doc_id, version, produced_at=self._clock())
        known = await self._pg_documents.known(CORPUS, doc_id)
        # a delete has no content, so its sha is None and nothing goes to s3
        early = self._early_answer(precheck(version, None, known), known, doc_id, version)
        if early is not None:
            return early
        outgoing = Outgoing(request_id, EVENT_SCHEMA, prepared.event_json)
        result = await self._pg_documents.accept(CORPUS, doc_id, version, None, outgoing)
        return self._after_accept(result, doc_id, version)

    async def status(self, doc_id: str) -> DocumentStatus | None:
        return await self._pg_documents.status(CORPUS, doc_id)

    async def status_batch(self, doc_ids: Sequence[str]) -> dict[str, DocumentStatus]:
        return await self._pg_documents.status_batch(CORPUS, doc_ids)

    # the four answers the precheck can give, shared by upsert and delete so they cannot drift
    # returns None when the write may go on
    def _early_answer(
        self, outcome: WritePrecheck, known: KnownDocument | None, doc_id: str, version: int
    ) -> WriteAccepted | WriteRefused | None:
        held = known.latest_version if known else None
        match outcome:
            case WritePrecheck.WRITE_PRECHECK_REJECT_STALE:
                return WriteRefused("stale_version", held)
            case WritePrecheck.WRITE_PRECHECK_REJECT_CONFLICT:
                return WriteRefused("content_conflict", held)
            case WritePrecheck.WRITE_PRECHECK_ACCEPT_DUPLICATE:
                return WriteAccepted(doc_id, version, self._clock())
            case WritePrecheck.WRITE_PRECHECK_PROCEED:
                return None

    # the compare and set can still lose after the precheck passed
    # this turns its result into an answer
    def _after_accept(
        self, result: AcceptResult, doc_id: str, version: int
    ) -> WriteAccepted | WriteRefused:
        if result.outcome is PgWriteResult.PG_WRITE_REJECTED:
            stale = result.latest_version > version
            return WriteRefused(
                "stale_version" if stale else "content_conflict", result.latest_version
            )
        # accepted, or a duplicate that raced past the precheck: both mean the write is durable
        return WriteAccepted(doc_id, version, self._clock())
