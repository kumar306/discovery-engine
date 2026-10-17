import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import uvicorn
from fastapi import FastAPI
from prometheus_client import CollectorRegistry

from libs.db.pool import create_pool
from libs.observability.logging import configure_logging
from libs.observability.probes import postgres_probe
from services.ingest.app import create_app
from services.ingest.probes import s3_probe
from services.ingest.repository import DocumentRepository
from services.ingest.service import IngestService
from services.ingest.settings import IngestSettings
from services.ingest.storage import build_store


def build(settings: IngestSettings | None = None) -> FastAPI:
    settings = settings or IngestSettings()
    pool = create_pool(settings.database_dsn, settings.db_pool_max, settings.db_pool_timeout_s)
    store = build_store(settings)
    repository = DocumentRepository(pool, settings.db_lock_timeout_s)
    service = IngestService(repository, store, clock=lambda: datetime.now(UTC))

    # asynccontextmanager turns this generator into something async with can drive
    # so setup and teardown can await real io, here opening and closing the pool
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # the pool opens when the server starts and closes when it stops, so a shutdown is clean
        await pool.open(wait=True, timeout=10)
        try:
            # yield freezes this function right here while the server handles requests
            # on sigterm uvicorn stops taking new requests and lets running ones finish
            # then the asgi shutdown event resumes the function on the next line
            yield
        finally:
            await pool.close()
            store.close()

    # python evaluates arguments first, so lifespan is already a finished function object here
    # create_app hands it to FastAPI which stores it, nothing runs at this point
    # uvicorn is given the app that build returns, so it carries the lifespan
    # and runs it when the server starts and again when it stops
    # readiness asks the two things the api cannot work without, like the other processes do
    probes = {"postgres": postgres_probe(pool), "s3": s3_probe(store)}
    return create_app(
        service, settings, lifespan=lifespan, probes=probes, registry=CollectorRegistry()
    )


if __name__ == "__main__":
    configure_logging()
    if sys.platform == "win32":
        # psycopg async cannot use the default windows loop, linux needs nothing
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    config = IngestSettings()
    # loop none keeps the event loop policy set above, uvicorn would otherwise pick its own
    uvicorn.run(build(config), host=config.host, port=config.port, loop="none")
