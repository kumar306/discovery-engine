import asyncio
import signal
import sys
from contextlib import suppress

from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient

from libs.db.pool import create_pool
from libs.observability.logging import configure_logging
from services.embed.consumer import EventConsumer, kafka_consumer, kafka_dead_letter_sink
from services.embed.embedder import Embedder, build_onnx_embedder
from services.embed.pipeline import Pipeline
from services.embed.qdrant import ChunkIndex, new_client, verify_collection
from services.embed.repository import ChunkRepository
from services.embed.settings import EmbedSettings
from services.ingest.storage import DocumentStore, build_store


def build_pipeline(
    settings: EmbedSettings,
    pg_pool: AsyncConnectionPool,
    qdrant_client: AsyncQdrantClient,
    s3_store: DocumentStore,
    embedder: Embedder,
) -> Pipeline:
    """The four collaborators of the pipeline, each built from the settings that belong to it."""
    return Pipeline(
        ChunkRepository(pg_pool, settings.db_lock_timeout_s),
        ChunkIndex(qdrant_client, settings.qdrant_collection),
        s3_store,
        embedder,
        settings,
    )


async def serve(
    settings: EmbedSettings, stop: asyncio.Event, embedder: Embedder | None = None
) -> None:
    """Check everything fits, then handle messages until told to stop, and close what was opened."""
    pg_pool = create_pool(settings.database_dsn, settings.db_pool_max, settings.db_pool_timeout_s)
    await pg_pool.open(wait=True, timeout=10)
    qdrant_client = new_client(settings)
    s3_store = build_store(settings)
    model: Embedder | None = embedder
    try:
        # a worker for another model, or a collection that is not there, must not read a message
        # the check is made here, before the consumer exists, so nothing is ever taken off the topic
        await verify_collection(qdrant_client, settings)
        # the model loads here, and a model of the wrong size stops the process too
        model = embedder or build_onnx_embedder(settings)
        await model.verify()
        pipeline = build_pipeline(settings, pg_pool, qdrant_client, s3_store, model)
        async with kafka_consumer(settings) as consumer, kafka_dead_letter_sink(settings) as sink:
            await EventConsumer(consumer, sink, pipeline, settings).run(stop)
    finally:
        if model is not None:
            model.close()
        s3_store.close()
        await qdrant_client.close()
        await pg_pool.close()


async def main() -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for name in (signal.SIGTERM, signal.SIGINT):
        # windows cannot add signal handlers to the loop, there ctrl c raises instead
        with suppress(NotImplementedError):
            loop.add_signal_handler(name, stop.set)
    await serve(EmbedSettings(), stop)


if __name__ == "__main__":
    configure_logging()
    if sys.platform == "win32":
        # psycopg async cannot use the default windows loop, linux needs nothing
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
