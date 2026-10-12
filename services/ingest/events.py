from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from libs.schemas import DeleteEvent, UpsertEvent, UpsertRequest, doc_event_adapter
from libs.schemas.hashing import canonical_bytes, content_sha

# written to the outbox next to every event and sent as the schema header on the kafka message
EVENT_SCHEMA = "doc_event_v1"


def s3_key(corpus: str, doc_id: str, version: int) -> str:
    return f"{corpus}/{doc_id}/v{version}.json"


@dataclass(frozen=True)
class PreparedUpsert:
    key: str
    data: bytes
    content_sha: str
    event_json: str


@dataclass(frozen=True)
class PreparedDelete:
    event_json: str


def prepare_upsert(request: UpsertRequest, produced_at: datetime) -> PreparedUpsert:
    data = canonical_bytes(request)
    sha = content_sha(data)
    key = s3_key(request.corpus, request.doc_id, request.version)
    event = UpsertEvent(
        schema_version=1,
        corpus=request.corpus,
        doc_id=request.doc_id,
        version=request.version,
        produced_at=produced_at,
        op="upsert",
        s3_key=key,
        content_sha=sha,
        size_bytes=len(data),
    )
    return PreparedUpsert(key, data, sha, _event_json(event))


def prepare_delete(
    corpus: Literal["esci"], doc_id: str, version: int, produced_at: datetime
) -> PreparedDelete:
    event = DeleteEvent(
        schema_version=1,
        corpus=corpus,
        doc_id=doc_id,
        version=version,
        produced_at=produced_at,
        op="delete",
    )
    return PreparedDelete(_event_json(event))


def _event_json(event: UpsertEvent | DeleteEvent) -> str:
    # dump_json validates the event and gives back utf-8 bytes
    # decode turns those bytes into a json string, the outbox payload column takes text
    return doc_event_adapter.dump_json(event).decode("utf-8")
