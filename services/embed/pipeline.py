from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from prometheus_client import CollectorRegistry
from pydantic import ValidationError

from libs.embedding.collection import normalise
from libs.embedding.model import Embedder
from libs.observability.metrics import EmbedMetrics
from libs.resilience.retry import PermanentError
from libs.schemas import DocEvent, UpsertEvent, UpsertRequest
from libs.schemas.hashing import chunk_id, content_sha
from services.embed.compose import Chunk, compose
from services.embed.diff import diff_chunks
from services.embed.gate import GateState, VersionGateDecision, gate
from services.embed.qdrant import ChunkPayload, IndexedChunk, LiveChunk
from services.embed.repository import NewChunk, PgCommitResult, Snapshot
from services.embed.settings import EmbedSettings


class ChunkBeingDeleted(RuntimeError):
    """A chunk this version needs is being deleted by the compactor, so the event waits.

    It is an ordinary error to the retry policy, so it is tried again and then held, and once
    the compactor has finished the chunk is a plain miss that is embedded and staged afresh.
    """


class KafkaEventResult(StrEnum):
    # the event was applied, and the document is live in the version it carried
    KAFKA_EVENT_APPLIED = "done"
    # the event was stale or a duplicate, nothing was changed, and that is a normal answer
    KAFKA_EVENT_SKIPPED = "dropped"


# the four things the pipeline talks to, as the small surfaces it uses, so a test can put a
# recorder between the pipeline and the real thing and see every call in order
class PgChunkRepo(Protocol):
    async def gate_state(self, corpus: str, doc_id: str) -> GateState | None: ...
    async def snapshot(self, corpus: str, doc_id: str) -> Snapshot: ...
    async def commit_version(
        self,
        corpus: str,
        doc_id: str,
        version: int,
        chunks: Sequence[NewChunk],
        revived: Sequence[UUID] = (),
    ) -> PgCommitResult: ...
    async def commit_delete(self, corpus: str, doc_id: str, version: int) -> PgCommitResult: ...
    async def mark_indexed(self, corpus: str, doc_id: str, version: int) -> bool: ...


class QdrantIndex(Protocol):
    async def stage(self, chunks: Sequence[IndexedChunk]) -> None: ...
    async def publish(
        self, version: int, live: Sequence[LiveChunk], superseded: Sequence[UUID]
    ) -> None: ...
    async def mark_deleted(self, doc_id: str) -> None: ...


class S3DocumentReader(Protocol):
    async def get(self, key: str) -> bytes: ...


class Pipeline:
    def __init__(
        self,
        pg_chunk_repo: PgChunkRepo,
        qdrant_index: QdrantIndex,
        s3_documents: S3DocumentReader,
        embedder: Embedder,
        embed_settings: EmbedSettings,
        metrics: EmbedMetrics | None = None,
    ) -> None:
        self._metrics = metrics or EmbedMetrics(CollectorRegistry())
        self._pg_chunk_repo = pg_chunk_repo
        self._qdrant_index = qdrant_index
        self._s3_documents = s3_documents
        self._embedder = embedder
        self._embed_settings = embed_settings

    async def handle(self, event: DocEvent) -> KafkaEventResult:
        """Apply one event, every step is safe to run again, an error means nothing committed."""
        if isinstance(event, UpsertEvent):
            return await self._upsert(event)
        return await self._delete(event.corpus, event.doc_id, event.version)

    async def _upsert(self, event: UpsertEvent) -> KafkaEventResult:
        # the gate, nothing is read or computed for an event that is stale or already applied
        if await self._is_stale(event.corpus, event.doc_id, event.version):
            return KafkaEventResult.KAFKA_EVENT_SKIPPED
        # read the document and turn it into chunks, still nothing has changed anywhere
        product = await self._read(event)
        chunks = compose(product, self._embedder.tokenizer, self._embedder.limits)
        by_id = {
            chunk_id(event.doc_id, self._embed_settings.model_version, chunk.content_hash): chunk
            for chunk in chunks
        }
        # what changed against the committed chunks decides what is embedded, and only that
        snapshot = await self._pg_chunk_repo.snapshot(event.corpus, event.doc_id)
        # a chunk the compactor has claimed loses its point any moment, so it is neither reused
        # nor staged again until the compactor is done, nothing has been embedded yet
        # this early stop only saves the embedding, a claim can still land later, during the s3
        # read or the staging, and the commit checks again under the lock, that check decides
        being_deleted = snapshot.claimed_among(by_id)
        if being_deleted:
            self._metrics.chunk_being_deleted.inc()
            raise ChunkBeingDeleted(
                f"{event.doc_id}: {len(being_deleted)} chunk(s) of this version are being deleted"
            )
        diff = diff_chunks(
            {cid: chunk.positions for cid, chunk in by_id.items()},
            snapshot.current,
            snapshot.stale,
        )
        # staged points are invisible to every search, so a crash from here to the commit
        # leaves the old version fully visible
        await self._stage(event, product, [(cid, by_id[cid]) for cid in diff.misses])
        # the commit, the moment the new version officially exists, judged again under the lock
        committed = await self._pg_chunk_repo.commit_version(
            event.corpus,
            event.doc_id,
            event.version,
            [NewChunk(cid, chunk.content_hash, chunk.positions) for cid, chunk in by_id.items()],
            diff.revived,
        )
        if committed is PgCommitResult.PG_COMMIT_SKIPPED_STALE:
            return KafkaEventResult.KAFKA_EVENT_SKIPPED
        if committed is PgCommitResult.PG_COMMIT_CHUNKS_CHANGED:
            # the compactor claimed or finished a chunk after the diff, so the diff is out of date
            # nothing was committed, the staged points are invisible, so the event is tried again
            # and the diff of the next attempt sees the chunk as a miss and embeds it afresh
            self._metrics.chunk_being_deleted.inc()
            raise ChunkBeingDeleted(f"{event.doc_id}: a chunk changed under the commit")
        await self._publish(event.corpus, event.doc_id, event.version)
        return KafkaEventResult.KAFKA_EVENT_APPLIED

    async def _delete(self, corpus: str, doc_id: str, version: int) -> KafkaEventResult:
        if await self._is_stale(corpus, doc_id, version):
            return KafkaEventResult.KAFKA_EVENT_SKIPPED
        if (
            await self._pg_chunk_repo.commit_delete(corpus, doc_id, version)
            is PgCommitResult.PG_COMMIT_SKIPPED_STALE
        ):
            return KafkaEventResult.KAFKA_EVENT_SKIPPED
        # the tombstone is committed, now every point of the product goes invisible at once
        await self._qdrant_index.mark_deleted(doc_id)
        await self._pg_chunk_repo.mark_indexed(corpus, doc_id, version)
        return KafkaEventResult.KAFKA_EVENT_APPLIED

    async def _is_stale(self, corpus: str, doc_id: str, version: int) -> bool:
        state = await self._pg_chunk_repo.gate_state(corpus, doc_id)
        return gate(version, state) is VersionGateDecision.VERSION_GATE_DROP

    async def _read(self, event: UpsertEvent) -> UpsertRequest:
        data = await self._s3_documents.get(event.s3_key)
        if content_sha(data) != event.content_sha:
            raise PermanentError(f"{event.s3_key} does not match the sha in its event")
        try:
            product = UpsertRequest.model_validate_json(data)
        except ValidationError as error:
            # the message is not echoed, a stored document may hold anything
            raise PermanentError(
                f"{event.s3_key} is not a valid document, {error.error_count()} problem(s)"
            ) from error
        if (product.corpus, product.doc_id, product.version) != (
            event.corpus,
            event.doc_id,
            event.version,
        ):
            raise PermanentError(f"{event.s3_key} holds another document than its event names")
        return product

    async def _stage(
        self, event: UpsertEvent, product: UpsertRequest, misses: list[tuple[UUID, Chunk]]
    ) -> None:
        if not misses:
            return
        embeddings = await self._embedder.embed(
            [chunk.text for _, chunk in misses], self._embed_settings.embed_timeout_s
        )
        await self._qdrant_index.stage(
            [
                IndexedChunk(
                    cid,
                    embedding.dense,
                    embedding.sparse_indices,
                    embedding.sparse_values,
                    ChunkPayload(
                        doc_id=event.doc_id,
                        version=event.version,
                        chunk_idx=chunk.positions[0],
                        content_hash=chunk.content_hash,
                        model_version=self._embed_settings.model_version,
                        locale=product.locale,
                        text=chunk.text,
                        title=normalise(product.title),
                        brand=product.brand,
                        color=product.color,
                    ),
                )
                for (cid, chunk), embedding in zip(misses, embeddings, strict=True)
            ]
        )

    async def _publish(self, corpus: str, doc_id: str, version: int) -> None:
        # what goes live is read back from postgres, the record of what was committed, and not
        # taken from memory, so a replay after a crash publishes exactly the committed chunks
        snapshot = await self._pg_chunk_repo.snapshot(corpus, doc_id)
        live = [
            LiveChunk(cid, positions[0])
            for cid, positions in sorted(snapshot.current.items(), key=lambda item: item[1][0])
        ]
        await self._qdrant_index.publish(version, live, sorted(snapshot.stale))
        await self._pg_chunk_repo.mark_indexed(corpus, doc_id, version)
