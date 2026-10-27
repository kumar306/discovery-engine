from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from prometheus_client import CollectorRegistry
from qdrant_client import AsyncQdrantClient

from libs.embedding.collection import qdrant_probe, verify_collection
from libs.embedding.model import build_onnx_embedder
from libs.observability.logging import configure_logging
from libs.observability.metrics import QueryMetrics
from services.query.app import create_app
from services.query.encoder import OnnxQueryEncoder
from services.query.flow_chunks import ChunkFlow
from services.query.flow_search import SearchFlow
from services.query.search import QdrantSearch
from services.query.settings import QuerySettings


def build(settings: QuerySettings | None = None) -> FastAPI:
    settings = settings or QuerySettings()
    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_s)
    registry = CollectorRegistry()
    metrics = QueryMetrics(registry)
    embedder = build_onnx_embedder(
        settings,
        workers=settings.encode_workers,
        wait_observer=metrics.encode_wait_seconds.observe,
    )
    # read from the embedder at the moment of a scrape
    metrics.encode_queued.set_function(lambda: embedder.queued_jobs)
    encoder = OnnxQueryEncoder(embedder, settings.encode_timeout_s)
    search = QdrantSearch(qdrant_client, settings.qdrant_collection)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # a missing collection, or one built for another model than this process loaded, stops
        # the process here, before it answers a query with vectors that cannot be compared
        await verify_collection(qdrant_client, settings)
        # a model of the wrong size stops it here too
        await embedder.verify()
        try:
            yield
        finally:
            embedder.close()
            await qdrant_client.close()

    # readiness asks the one thing the service cannot work without, and never Postgres:
    # the query path does not depend on the metadata database
    return create_app(
        SearchFlow(encoder, search, settings, metrics),
        ChunkFlow(encoder, search, settings, metrics),
        settings,
        metrics,
        registry,
        lifespan=lifespan,
        probes={"qdrant": qdrant_probe(qdrant_client, settings.qdrant_collection)},
    )


if __name__ == "__main__":
    configure_logging()
    config = QuerySettings()
    uvicorn.run(build(config), host=config.host, port=config.port)
