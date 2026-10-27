"""Flow S: search and recommendations. One result per product.

Two retrievers run at the same time. Each is "encode the query, then search Qdrant", and each has
its own timeout. Qdrant itself collapses the chunks of a product to the best one (group_by
doc_id), so a ranking here is a list of products. The two product rankings are fused with
reciprocal rank fusion. If one retriever fails, the answer comes from the other and says so.
"""

import asyncio
import logging
import time
from typing import Annotated, Literal

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from libs.embedding.collection import filter_key
from libs.observability.metrics import QueryMetrics
from libs.schemas.documents import Locale
from services.query.api_models import ErrorBody, ValidationFailedBody
from services.query.encoder import QueryEncoder
from services.query.primitives import Hit, QueryText, Retrieval, Unavailable, rrf
from services.query.search import Filters, QdrantSearch
from services.query.settings import QuerySettings

log = logging.getLogger("query.search")

MAX_K = 100

FilterText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchFilters(_Strict):
    brand: FilterText | None = None
    color: FilterText | None = None
    locale: Locale | None = None


class SearchRequest(_Strict):
    query: QueryText
    k: int = Field(default=10, ge=1, le=MAX_K)
    filters: SearchFilters = SearchFilters()
    explain: bool = False


class Explain(BaseModel):
    # 1-based rank in each retriever's list, None when that retriever did not return the product
    dense_rank: int | None
    sparse_rank: int | None
    rrf: float


class ResultItem(BaseModel):
    doc_id: str
    title: str
    brand: str | None
    color: str | None
    # the fusion score, only meaningful for comparing results of this one answer
    score: float
    snippet: str
    explain: Explain | None


class SearchResponse(BaseModel):
    request_id: str
    results: list[ResultItem]
    # the retrievers left out of this answer: dense, sparse
    degraded: list[str]
    timings_ms: dict[str, float]


class SearchFlow:
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

    async def run(self, request: SearchRequest, request_id: str) -> SearchResponse:
        filters = Filters(
            brand_key=None if request.filters.brand is None else filter_key(request.filters.brand),
            color_key=None if request.filters.color is None else filter_key(request.filters.color),
            locale=request.filters.locale,
        )
        # both retrievers at once, neither can raise (a failure comes back as hits=None), so one
        # failing never cancels the other
        async with asyncio.TaskGroup() as group:
            dense = group.create_task(self._retrieve("dense", request.query, filters))
            sparse = group.create_task(self._retrieve("sparse", request.query, filters))
        retrievals = [dense.result(), sparse.result()]
        timings = {r.name: r.seconds * 1000 for r in retrievals}
        for r in retrievals:
            self._metrics.stage_seconds.labels(flow="search", stage=r.name).observe(r.seconds)
            if r.encode_seconds is not None:
                timings[f"encode_{r.name}"] = r.encode_seconds * 1000
                self._metrics.stage_seconds.labels(flow="search", stage=f"encode_{r.name}").observe(
                    r.encode_seconds
                )
            if r.search_seconds is not None:
                timings[f"search_{r.name}"] = r.search_seconds * 1000
                self._metrics.stage_seconds.labels(flow="search", stage=f"search_{r.name}").observe(
                    r.search_seconds
                )
        rankings = {r.name: r.hits for r in retrievals if r.hits is not None}
        degraded = [r.name for r in retrievals if r.hits is None]
        for name in degraded:
            self._metrics.degraded.labels(flow="search", retriever=name).inc()
        if not rankings:
            raise Unavailable("neither retriever answered")

        started = time.perf_counter()
        fused = rrf(
            {name: [hit.doc_id for hit in hits] for name, hits in rankings.items()},
            k=self._settings.rrf_k,
        )
        # the chunk shown for a product is the one from the retriever that ranked it higher,
        # the dense one when both ranked it the same
        shown: dict[str, tuple[int, Hit]] = {}
        for hits in rankings.values():
            for position, hit in enumerate(hits, start=1):
                if hit.doc_id not in shown or position < shown[hit.doc_id][0]:
                    shown[hit.doc_id] = (position, hit)
        results = []
        for item in fused[: request.k]:
            best = shown[item.key][1]
            results.append(
                ResultItem(
                    doc_id=item.key,
                    title=best.title,
                    brand=best.brand,
                    color=best.color,
                    score=item.score,
                    snippet=best.text[: self._settings.snippet_chars],
                    explain=Explain(
                        dense_rank=item.ranks.get("dense"),
                        sparse_rank=item.ranks.get("sparse"),
                        rrf=item.score,
                    )
                    if request.explain
                    else None,
                )
            )
        fuse_seconds = time.perf_counter() - started
        self._metrics.stage_seconds.labels(flow="search", stage="fuse").observe(fuse_seconds)
        timings["fuse"] = fuse_seconds * 1000
        return SearchResponse(
            request_id=request_id, results=results, degraded=degraded, timings_ms=timings
        )

    async def _retrieve(
        self, name: Literal["dense", "sparse"], query: str, filters: Filters
    ) -> Retrieval:
        started = time.perf_counter()
        hits: list[Hit] | None = None
        encode_seconds: float | None = None
        search_seconds: float | None = None
        limit = self._settings.candidates_per_retriever
        try:
            async with asyncio.timeout(self._settings.retriever_timeout_s):
                step_started = time.perf_counter()
                if name == "dense":
                    vector = await self._encoder.encode_dense(query)
                    encode_seconds = time.perf_counter() - step_started
                    step_started = time.perf_counter()
                    hits = await self._search.grouped_search("dense", vector, filters, limit)
                else:
                    terms = await self._encoder.encode_sparse(query)
                    encode_seconds = time.perf_counter() - step_started
                    step_started = time.perf_counter()
                    hits = await self._search.grouped_search("bm25", terms, filters, limit)
                search_seconds = time.perf_counter() - step_started
        except Exception as error:
            # the type only, an error message may hold the query or a payload
            log.warning(
                "retriever failed",
                extra={"fields": {"retriever": name, "error_type": type(error).__name__}},
            )
        return Retrieval(name, hits, time.perf_counter() - started, encode_seconds, search_seconds)


def register_search_route(app: FastAPI, flow: SearchFlow) -> None:
    @app.post(
        "/v1/search",
        responses={
            413: {"model": ErrorBody, "description": "The body is bigger than the limit."},
            422: {"model": ValidationFailedBody, "description": "A field breaks the rules."},
            503: {"model": ErrorBody, "description": "No retriever could answer, or overloaded."},
        },
    )
    async def search(body: SearchRequest, request: Request) -> SearchResponse:
        response = await flow.run(body, request.state.request_id)
        request.state.outcome = "degraded" if response.degraded else "ok"
        return response
