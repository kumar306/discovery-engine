"""Events for tests whose handler never reads the document, so no s3 object is needed."""

from datetime import UTC, datetime

from libs.schemas import UpsertEvent

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def upsert_event_bytes(doc_id: str, version: int) -> bytes:
    event = UpsertEvent(
        schema_version=1,
        corpus="esci",
        doc_id=doc_id,
        version=version,
        produced_at=NOW,
        op="upsert",
        s3_key=f"esci/{doc_id}/v{version}.json",
        content_sha="a" * 64,
        size_bytes=1,
    )
    return event.model_dump_json().encode()
