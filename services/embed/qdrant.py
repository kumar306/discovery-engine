from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from qdrant_client import AsyncQdrantClient, models

from services.embed.settings import EmbedSettings

DENSE = "dense"
SPARSE = "bm25"

# the payload fields a search or a flip filters on get an index
# without one qdrant scans the payload of every point to find the state, doc_id or version
# we ask for, with one it jumps to the matching points, so moving staging to live never
# means a full scan, a missing index only makes the filters slow
_PAYLOAD_INDEXES = {
    "state": models.PayloadSchemaType.KEYWORD,
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "locale": models.PayloadSchemaType.KEYWORD,
    "version": models.PayloadSchemaType.INTEGER,
}


class QdrantPointState(StrEnum):
    # every search filters on live, so the other three are invisible to readers
    QDRANT_POINT_STAGING = "staging"
    QDRANT_POINT_LIVE = "live"
    QDRANT_POINT_SUPERSEDED = "superseded"
    QDRANT_POINT_DELETED = "deleted"


class CollectionMismatch(RuntimeError):
    """The collection is missing or was built for a different model, the worker must not run."""


@dataclass(frozen=True)
class ChunkPayload:
    doc_id: str
    version: int
    chunk_idx: int
    content_hash: str
    model_version: str
    locale: str
    text: str
    brand: str | None = None
    color: str | None = None


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: UUID
    dense: Sequence[float]
    sparse_indices: Sequence[int]
    sparse_values: Sequence[float]
    payload: ChunkPayload


@dataclass(frozen=True)
class LiveChunk:
    chunk_id: UUID
    chunk_idx: int


def _ids(chunk_ids: Sequence[UUID]) -> models.Filter:
    # match the chunk ids with a filter, not with a request by id
    # a request by id answers 404 when any point is missing, one the compactor already deleted
    # for example, while a filter answers 200 ok and matches only what exists, which we can
    # handle cleanly
    return models.Filter(must=[models.HasIdCondition(has_id=[str(i) for i in chunk_ids])])


def _set(payload: dict[str, object], where: models.Filter) -> models.SetPayloadOperation:
    return models.SetPayloadOperation(set_payload=models.SetPayload(payload=payload, filter=where))


def publish_operations(
    version: int, live: Sequence[LiveChunk], superseded: Sequence[UUID]
) -> list[models.UpdateOperation]:
    # the new version goes live first and the old one is superseded after it, so a product
    # that was searchable never has zero live chunks, even if we stop between the two
    # every live chunk gets state, version and position, not only the ones that moved, because
    # after a crash the postgres rows already hold the new positions and no diff would show the move
    operations: list[models.UpdateOperation] = [
        _set(
            {
                "state": QdrantPointState.QDRANT_POINT_LIVE.value,
                "version": version,
                "chunk_idx": chunk.chunk_idx,
            },
            _ids([chunk.chunk_id]),
        )
        for chunk in live
    ]
    if superseded:
        operations.append(
            _set({"state": QdrantPointState.QDRANT_POINT_SUPERSEDED.value}, _ids(superseded))
        )
    return operations


class ChunkIndex:
    def __init__(self, qdrant_client: AsyncQdrantClient, qdrant_collection: str) -> None:
        self._qdrant_client = qdrant_client
        self._qdrant_collection = qdrant_collection

    async def stage(self, chunks: Sequence[IndexedChunk]) -> None:
        if not chunks:
            return
        points = [
            models.PointStruct(
                id=str(chunk.chunk_id),
                vector={
                    DENSE: list(chunk.dense),
                    SPARSE: models.SparseVector(
                        indices=list(chunk.sparse_indices), values=list(chunk.sparse_values)
                    ),
                },
                payload=_payload(chunk.payload),
            )
            for chunk in chunks
        ]
        # insert only: qdrant matches a point by its id, and a duplicate chunk has the same id
        # because the id is made from the document, the model version and the chunk content
        # so the existing point is left alone and no new point is added for the duplicate
        # a redelivered message can never demote a point that went live back to staging
        await self._qdrant_client.upsert(
            self._qdrant_collection, points, wait=True, update_mode=models.UpdateMode.INSERT_ONLY
        )

    async def publish(
        self, version: int, live: Sequence[LiveChunk], superseded: Sequence[UUID]
    ) -> None:
        if not live:
            # superseding the old chunks with nothing live in their place would hide the product
            raise ValueError("a version with no live chunks cannot be published")
        # one request, run in order, and running it twice leaves the same payloads
        await self._qdrant_client.batch_update_points(
            self._qdrant_collection,
            update_operations=publish_operations(version, live, superseded),
            wait=True,
        )

    async def mark_deleted(self, doc_id: str) -> None:
        # by the product id, so points in any state go invisible at once, whatever version made them
        where = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )
        await self._qdrant_client.set_payload(
            self._qdrant_collection,
            {"state": QdrantPointState.QDRANT_POINT_DELETED.value},
            points=where,
            wait=True,
        )


def _payload(payload: ChunkPayload) -> dict[str, object]:
    fields: dict[str, object] = {
        "doc_id": payload.doc_id,
        "version": payload.version,
        "chunk_idx": payload.chunk_idx,
        "content_hash": payload.content_hash,
        "model_version": payload.model_version,
        "locale": payload.locale,
        "text": payload.text,
        "state": QdrantPointState.QDRANT_POINT_STAGING.value,
    }
    # a missing brand or color is left out, a filter on it simply does not match
    if payload.brand is not None:
        fields["brand"] = payload.brand
    if payload.color is not None:
        fields["color"] = payload.color
    return fields


def new_client(settings: EmbedSettings) -> AsyncQdrantClient:
    return AsyncQdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_s)


async def ensure_collection(client: AsyncQdrantClient, settings: EmbedSettings) -> bool:
    """Create the collection when it is missing, return False when it was already right."""
    if not await client.collection_exists(settings.qdrant_collection):
        await client.create_collection(
            settings.qdrant_collection,
            vectors_config={
                DENSE: models.VectorParams(size=settings.dense_dim, distance=models.Distance.COSINE)
            },
            # the idf modifier makes the server weigh rare terms higher at query time
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            # the model version lives in the collection, so a worker for another model refuses it
            metadata={"model_version": settings.model_version},
        )
        for field, schema in _PAYLOAD_INDEXES.items():
            await client.create_payload_index(settings.qdrant_collection, field, schema)
        return True
    # an existing collection is never altered, it is checked, and a wrong one stops the caller
    await verify_collection(client, settings)
    return False


# at worker startup: the worker has its model version baked into the settings property, built
# from the model name, the bm25 length and the revision, and this compares it with the one
# stored in the collection, so a worker for another model refuses to start
# the collection may not exist yet, that is flagged here too, with the fix in the message
async def verify_collection(client: AsyncQdrantClient, settings: EmbedSettings) -> None:
    name = settings.qdrant_collection
    if not await client.collection_exists(name):
        raise CollectionMismatch(f"collection {name} does not exist, run make init")
    info = await client.get_collection(name)
    problems: list[str] = []
    dense = (info.config.params.vectors or {}).get(DENSE)  # type: ignore[union-attr]
    if not isinstance(dense, models.VectorParams):
        problems.append("the dense vector is missing")
    else:
        if dense.size != settings.dense_dim:
            problems.append(f"dense size is {dense.size}, expected {settings.dense_dim}")
        if dense.distance != models.Distance.COSINE:
            problems.append(f"dense distance is {dense.distance}, expected cosine")
    sparse = (info.config.params.sparse_vectors or {}).get(SPARSE)
    if sparse is None:
        problems.append("the bm25 sparse vector is missing")
    elif sparse.modifier != models.Modifier.IDF:
        problems.append("the bm25 sparse vector has no idf modifier")
    stored = (info.config.metadata or {}).get("model_version")
    if stored != settings.model_version:
        problems.append(f"model_version is {stored!r}, expected {settings.model_version!r}")
    for field, schema in _PAYLOAD_INDEXES.items():
        index = info.payload_schema.get(field)
        if index is None or index.data_type != schema:
            problems.append(f"payload index on {field} is missing or has the wrong type")
    if problems:
        raise CollectionMismatch(f"collection {name}: " + "; ".join(problems))
