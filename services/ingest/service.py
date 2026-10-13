from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from libs.schemas import UpsertRequest
from services.ingest.events import EVENT_SCHEMA, prepare_upsert
from services.ingest.repository import DocumentRepository, Outcome, Outgoing
from services.ingest.rules import PrecheckOutcome, precheck
from services.ingest.storage import DocumentStore, PutResult


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
        match precheck(request.version, prepared.content_sha, known):
            case PrecheckOutcome.REJECT_STALE:
                return Refused("stale_version", known.latest_version if known else None)
            case PrecheckOutcome.REJECT_CONFLICT:
                return Refused("content_conflict", known.latest_version if known else None)
            case PrecheckOutcome.ACCEPT_DUPLICATE:
                return Accepted(request.doc_id, request.version, self._clock())
            case PrecheckOutcome.PROCEED:
                pass
        # s3 first, the object is immutable and a leftover from a crash is harmless
        put = await self._store.put_if_absent(prepared.key, prepared.data, prepared.content_sha)
        if put is PutResult.CONFLICT:
            return Refused("content_conflict", known.latest_version if known else None)
        outgoing = Outgoing(request_id, EVENT_SCHEMA, prepared.event_json)
        result = await self._repository.accept(
            request.corpus, request.doc_id, request.version, prepared.content_sha, outgoing
        )
        if result.outcome is Outcome.CONFLICT:
            stale = result.latest_version > request.version
            return Refused("stale_version" if stale else "content_conflict", result.latest_version)
        # accepted, or a duplicate that raced past the precheck: both mean the write is durable
        return Accepted(request.doc_id, request.version, self._clock())
