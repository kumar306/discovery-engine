from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from qdrant_client import AsyncQdrantClient, models

from libs.embedding.collection import (
    DENSE,
    PAYLOAD_INDEXES,
    SPARSE,
    QdrantPointState,
    verify_collection,
)
from libs.settings import QdrantSettings
from services.embed.settings import EmbedSettings


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


# qdrant cannot flip all the points of a product at once, so for a moment a search can meet a
# product with some chunks at version n and some at n plus 1
# nothing here or in qdrant picks between them, qdrant returns every live chunk that matches
# our query service does, in s6: of the live chunks of one product it keeps only those with the
# highest version, so a changed chunk and its replacement are never shown together
# what the rule needs from this function: new chunks go live first (never zero live chunks), and
# every live chunk gets its version, so the rule can tell the new chunks from the old
# no lock or lease on purpose, see docs/learning/reader_consistency.md
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

    async def delete(self, chunk_ids: Sequence[UUID]) -> None:
        if not chunk_ids:
            return
        # a filter on ids, like the flips, so a point that is already gone is not an error and
        # a pass of the compactor that is repeated after a crash simply finds nothing to delete
        await self._qdrant_client.delete(
            self._qdrant_collection,
            points_selector=models.FilterSelector(filter=_ids(chunk_ids)),
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


def new_client(settings: QdrantSettings) -> AsyncQdrantClient:
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
        for field, schema in PAYLOAD_INDEXES.items():
            await client.create_payload_index(settings.qdrant_collection, field, schema)
        return True
    # an existing collection is never altered, it is checked, and a wrong one stops the caller
    await verify_collection(client, settings)
    return False
