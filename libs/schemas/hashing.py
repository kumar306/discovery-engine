import hashlib
import json
import uuid

from libs.schemas.documents import UpsertRequest

CHUNK_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "search-platform.chunks")


def canonical_bytes(request: UpsertRequest) -> bytes:
    document = request.model_dump(mode="json")
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def content_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chunk_id(doc_id: str, model_version: str, content_hash: str) -> uuid.UUID:
    doc_namespace = uuid.uuid5(CHUNK_NAMESPACE, doc_id)
    model_namespace = uuid.uuid5(doc_namespace, model_version)
    return uuid.uuid5(model_namespace, content_hash)
