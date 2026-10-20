"""Flow C: raw chunks for the shopping assistant. Several chunks of one product are allowed.

Two retrievers run at the same time, each "encode the query, then search Qdrant" with its own
timeout, and neither groups: a ranking here is a list of chunks. The two chunk rankings are fused
with reciprocal rank fusion, keyed by the chunk, so two chunks of one product stay two separate
results. An optional list of product ids restricts the search to products already shown. This flow
shares no handler code with Flow S, only the primitives.
"""

import asyncio
import logging
import time
from typing import Literal

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from libs.observability.metrics import QueryMetrics
from libs.schemas.documents import DocId
from services.query.api_models import ErrorBody, ValidationFailedBody
from services.query.encoder import QueryEncoder
from services.query.primitives import Hit, QueryText, Retrieval, Unavailable, rrf
from services.query.search import Filters, QdrantSearch
from services.query.settings import QuerySettings

log = logging.getLogger("query.chunks")


class ChunkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: QueryText
    # chunks asked of each retriever, the configured default when absent
    per_retriever_limit: int | None = Field(default=None, ge=1, le=100)
    # only these products, for a follow up question about products already shown
    doc_ids: list[DocId] = Field(default_factory=list, max_length=100)


class ChunkResult(BaseModel):
    point_id: str
    doc_id: str
    chunk_idx: int
    text: str
    title: str
    brand: str | None
    color: str | None
    rrf: float
    # 1-based rank in each retriever's list, None when that retriever did not return the chunk
    dense_rank: int | None
    sparse_rank: int | None


class ChunkResponse(BaseModel):
    request_id: str
    chunks: list[ChunkResult]
    # the retrievers left out of this answer: dense, sparse
    degraded: list[str]
    timings_ms: dict[str, float]


class ChunkFlow:
    def __init__(
        self,
        encoder: QueryEncoder,
        search: QdrantSearch,
        settings: QuerySettings,
        metrics: QueryMetrics,
    ) -> None:
        self._encoder = encoder
        self._search = search
        self._settings = settings
        self._metrics = metrics

    async def run(self, request: ChunkRequest, request_id: str) -> ChunkResponse:
        filters = Filters(doc_ids=tuple(request.doc_ids))
        limit = request.per_retriever_limit or self._settings.chunks_per_retriever
        # both retrievers at once, neither can raise (a failure comes back as hits=None), so one
        # failing never cancels the other
        async with asyncio.TaskGroup() as group:
            dense = group.create_task(self._retrieve("dense", request.query, filters, limit))
            sparse = group.create_task(self._retrieve("sparse", request.query, filters, limit))
        retrievals = [dense.result(), sparse.result()]
        timings = {r.name: r.seconds * 1000 for r in retrievals}
        for r in retrievals:
            self._metrics.stage_seconds.labels(flow="chunks", stage=r.name).observe(r.seconds)
        rankings = {r.name: r.hits for r in retrievals if r.hits is not None}
        degraded = [r.name for r in retrievals if r.hits is None]
        for name in degraded:
            self._metrics.degraded.labels(flow="chunks", retriever=name).inc()
        if not rankings:
            raise Unavailable("neither retriever answered")

        started = time.perf_counter()
        # the key is the chunk, not the product
        fused = rrf(
            {name: [hit.point_id for hit in hits] for name, hits in rankings.items()},
            k=self._settings.rrf_k,
        )
        by_point = {hit.point_id: hit for hits in rankings.values() for hit in hits}
        chunks = [
            ChunkResult(
                point_id=item.key,
                doc_id=by_point[item.key].doc_id,
                chunk_idx=by_point[item.key].chunk_idx,
                text=by_point[item.key].text,
                title=by_point[item.key].title,
                brand=by_point[item.key].brand,
                color=by_point[item.key].color,
                rrf=item.score,
                dense_rank=item.ranks.get("dense"),
                sparse_rank=item.ranks.get("sparse"),
            )
            for item in fused
        ]
        fuse_seconds = time.perf_counter() - started
        self._metrics.stage_seconds.labels(flow="chunks", stage="fuse").observe(fuse_seconds)
        timings["fuse"] = fuse_seconds * 1000
        return ChunkResponse(
            request_id=request_id, chunks=chunks, degraded=degraded, timings_ms=timings
        )

    async def _retrieve(
        self, name: Literal["dense", "sparse"], query: str, filters: Filters, limit: int
    ) -> Retrieval:
        started = time.perf_counter()
        hits: list[Hit] | None = None
        try:
            async with asyncio.timeout(self._settings.retriever_timeout_s):
                if name == "dense":
                    vector = await self._encoder.encode_dense(query)
                    hits = await self._search.chunk_search("dense", vector, filters, limit)
                else:
                    terms = await self._encoder.encode_sparse(query)
                    hits = await self._search.chunk_search("bm25", terms, filters, limit)
        except Exception as error:
            # the type only, an error message may hold the query or a payload
            log.warning(
                "retriever failed",
                extra={"fields": {"retriever": name, "error_type": type(error).__name__}},
            )
        return Retrieval(name, hits, time.perf_counter() - started)


def register_chunk_route(app: FastAPI, flow: ChunkFlow) -> None:
    @app.post(
        "/v1/chunks",
        responses={
            413: {"model": ErrorBody, "description": "The body is bigger than the limit."},
            422: {"model": ValidationFailedBody, "description": "A field breaks the rules."},
            503: {"model": ErrorBody, "description": "No retriever could answer, or overloaded."},
        },
    )
    async def chunks(body: ChunkRequest, request: Request) -> ChunkResponse:
        response = await flow.run(body, request.state.request_id)
        request.state.outcome = "degraded" if response.degraded else "ok"
        return response
