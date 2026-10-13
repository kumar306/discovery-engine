from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from libs.schemas import UpsertRequest
from services.ingest.events import EVENT_SCHEMA, prepare_delete, prepare_upsert
from services.ingest.repository import AcceptResult, DocumentRepository, Outcome, Outgoing
from services.ingest.rules import KnownDocument, PrecheckOutcome, precheck
from services.ingest.storage import DocumentStore, PutResult

CORPUS: Literal["esci"] = "esci"


@dataclass(frozen=True)
class Accepted:
    doc_id: str
    version: int
    accepted_at: datetime


@dataclass(frozen=True)
class Refused:
    error: Literal["stale_version", "content_conflict"]
    # the newest version we hold, or None when only a leftover s3 object stands in the way
    current_version: int | None


class IngestService:
    def __init__(
        self, repository: DocumentRepository, store: DocumentStore, clock: Callable[[], datetime]
    ) -> None:
        self._repository = repository
        self._store = store
        self._clock = clock

    async def upsert(self, request: UpsertRequest, request_id: str) -> Accepted | Refused:
        prepared = prepare_upsert(request, produced_at=self._clock())
        known = await self._repository.known(request.corpus, request.doc_id)
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
        put = await self._store.put_if_absent(prepared.key, prepared.data, prepared.content_sha)
        if put is PutResult.CONFLICT:
            return Refused("content_conflict", known.latest_version if known else None)
        outgoing = Outgoing(request_id, EVENT_SCHEMA, prepared.event_json)
        result = await self._repository.accept(
            request.corpus, request.doc_id, request.version, prepared.content_sha, outgoing
        )
        return self._after_accept(result, request.doc_id, request.version)

    async def delete(self, doc_id: str, version: int, request_id: str) -> Accepted | Refused:
        prepared = prepare_delete(CORPUS, doc_id, version, produced_at=self._clock())
        known = await self._repository.known(CORPUS, doc_id)
        # a delete has no content, so its sha is None and nothing goes to s3
        early = self._early_answer(precheck(version, None, known), known, doc_id, version)
        if early is not None:
            return early
        outgoing = Outgoing(request_id, EVENT_SCHEMA, prepared.event_json)
        result = await self._repository.accept(CORPUS, doc_id, version, None, outgoing)
        return self._after_accept(result, doc_id, version)

    # the four answers the precheck can give, shared by upsert and delete so they cannot drift
    # returns None when the write may go on
    def _early_answer(
        self, outcome: PrecheckOutcome, known: KnownDocument | None, doc_id: str, version: int
    ) -> Accepted | Refused | None:
        held = known.latest_version if known else None
        match outcome:
            case PrecheckOutcome.REJECT_STALE:
                return Refused("stale_version", held)
            case PrecheckOutcome.REJECT_CONFLICT:
                return Refused("content_conflict", held)
            case PrecheckOutcome.ACCEPT_DUPLICATE:
                return Accepted(doc_id, version, self._clock())
            case PrecheckOutcome.PROCEED:
                return None

    # the compare and set can still lose after the precheck passed
    # this turns its result into an answer
    def _after_accept(self, result: AcceptResult, doc_id: str, version: int) -> Accepted | Refused:
        if result.outcome is Outcome.CONFLICT:
            stale = result.latest_version > version
            return Refused("stale_version" if stale else "content_conflict", result.latest_version)
        # accepted, or a duplicate that raced past the precheck: both mean the write is durable
        return Accepted(doc_id, version, self._clock())
