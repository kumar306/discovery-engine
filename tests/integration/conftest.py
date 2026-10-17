"""Fixtures for tests that need the compose stack (run make up first).

Databases are created through the direct postgres port because create and drop database cannot run
inside a pooled transaction. Everything the application does goes through pgbouncer, as in prod.
"""

import asyncio
import json
import os
import random
import socket
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import httpx
import psycopg
import pytest
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from botocore.config import Config
from botocore.exceptions import ClientError
from psycopg import sql
from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from libs.db.migrate import migrate
from libs.db.pool import create_pool
from libs.resilience.bulkhead import Bulkhead
from libs.resilience.retry import RetryPolicy
from libs.schemas import DocEvent, UpsertRequest, doc_event_adapter
from services.embed.pipeline import Pipeline
from services.embed.qdrant import ChunkIndex, ensure_collection
from services.embed.repository import ChunkRepository
from services.embed.settings import EmbedSettings
from services.ingest.app import create_app
from services.ingest.repository import DocumentRepository
from services.ingest.service import IngestService
from services.ingest.settings import IngestSettings
from services.ingest.storage import DocumentStore
from tests.support.hash_embedder import HashEmbedder

DIRECT = os.environ.get("TEST_POSTGRES_DIRECT", "search:search_dev_only@127.0.0.1:5432")
POOLED = os.environ.get("TEST_POSTGRES_POOLED", "search:search_dev_only@127.0.0.1:6432")


@dataclass(frozen=True)
class Database:
    name: str
    direct_dsn: str
    pooled_dsn: str


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # everything in this folder needs the stack, so it is skipped by the default make test
    here = Path(__file__).parent
    for item in items:
        if here in Path(str(item.path)).parents:
            item.add_marker(pytest.mark.integration)


def create_database() -> Database:
    name = f"search_test_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(f"postgresql://{DIRECT}/postgres", autocommit=True) as admin:
        admin.execute(sql.SQL("create database {}").format(sql.Identifier(name)))
    return Database(name, f"postgresql://{DIRECT}/{name}", f"postgresql://{POOLED}/{name}")


def drop_database(name: str) -> None:
    # force is needed because pgbouncer keeps idle server connections open to the database
    with psycopg.connect(f"postgresql://{DIRECT}/postgres", autocommit=True) as admin:
        admin.execute(
            sql.SQL("drop database if exists {} with (force)").format(sql.Identifier(name))
        )


@pytest.fixture
def empty_database() -> Iterator[Database]:
    database = create_database()
    try:
        yield database
    finally:
        drop_database(database.name)


@pytest.fixture(scope="session")
def migrated_database() -> Iterator[Database]:
    database = create_database()
    try:
        migrate(database.pooled_dsn)
        yield database
    finally:
        # also runs when the migration itself fails, so a broken schema never leaks a database
        drop_database(database.name)


@pytest.fixture
def db(migrated_database: Database) -> Database:
    # the relay claims any unpublished row, so every test starts from empty tables
    with psycopg.connect(migrated_database.direct_dsn, autocommit=True) as conn:
        conn.execute("truncate documents, outbox, chunks, stale_chunks restart identity")
    return migrated_database


# the app pool is as big as the production default, tests never use more writers than this
APP_POOL_SIZE: int = IngestSettings.model_fields["db_pool_max"].default


@pytest.fixture
async def pool(db: Database) -> AsyncIterator[AsyncConnectionPool]:
    pool = create_pool(db.pooled_dsn, max_size=APP_POOL_SIZE, timeout=2.0)
    await pool.open(wait=True, timeout=10)
    try:
        yield pool
    finally:
        await pool.close()


# the s3 stand in and the readers below are shared by the storage and the api tests
ENDPOINT = "http://127.0.0.1:5000"
POLICY = RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.05)
S3_THREADS: int = IngestSettings.model_fields["s3_threads"].default
S3_BULKHEAD_LIMIT: int = IngestSettings.model_fields["s3_bulkhead_limit"].default


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def moto_client(endpoint: str = ENDPOINT, timeout: float = 3.0) -> Any:
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        # moto accepts any credentials, real aws would use the pod role
        aws_access_key_id="test",
        aws_secret_access_key="test",
        config=Config(
            connect_timeout=timeout,
            read_timeout=timeout,
            retries={"total_max_attempts": 1},
            max_pool_connections=S3_THREADS,
        ),
    )


def make_store(
    client: Any, bucket: str, limit: int = S3_BULKHEAD_LIMIT
) -> tuple[DocumentStore, Sleeps]:
    sleeps = Sleeps()
    store = DocumentStore(
        client,
        bucket,
        ThreadPoolExecutor(max_workers=S3_THREADS),
        Bulkhead(limit, wait_timeout=0.05),
        POLICY,
        sleeps,
        random.Random(1),
    )
    return store, sleeps


@pytest.fixture
def bucket() -> Iterator[str]:
    # every test gets its own bucket, bucket names allow only lowercase letters and digits here
    name = f"searchtest{uuid.uuid4().hex[:12]}"
    admin = moto_client()
    admin.create_bucket(Bucket=name)
    yield name
    for page in admin.get_paginator("list_objects_v2").paginate(Bucket=name):
        for item in page.get("Contents", []):
            admin.delete_object(Bucket=name, Key=item["Key"])
    admin.delete_bucket(Bucket=name)


async def wait_until_a_writer_is_blocked(dsn: str, give_up_after: float = 5.0) -> None:
    deadline = time.monotonic() + give_up_after
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        while time.monotonic() < deadline:
            row = await (
                await conn.execute(
                    "select count(*) from pg_stat_activity "
                    "where datname = current_database() and wait_event_type = 'Lock'"
                )
            ).fetchone()
            if row is not None and row[0] >= 1:
                return
            await asyncio.sleep(0.02)
    raise TimeoutError("no writer ever waited on the row lock")


def document(dsn: str, doc_id: str) -> dict[str, Any] | None:
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            "select latest_version, latest_content_sha, current_version, indexed_version, deleted, "
            "created_at, updated_at from documents where corpus = 'esci' and doc_id = %s",
            (doc_id,),
        ).fetchone()
    if row is None:
        return None
    names = ["latest", "sha", "current", "indexed", "deleted", "created_at", "updated_at"]
    return dict(zip(names, row, strict=True))


OUTBOX_COLUMNS = "select id, doc_id, request_id, event_schema, payload, published_at from outbox"


def outbox(dsn: str, doc_id: str | None = None) -> list[tuple[Any, ...]]:
    with psycopg.connect(dsn) as conn:
        if doc_id is None:
            return conn.execute(OUTBOX_COLUMNS + " order by id").fetchall()
        return conn.execute(OUTBOX_COLUMNS + " where doc_id = %s order by id", (doc_id,)).fetchall()


# the api itself, served in process through httpx with the real stores behind it
NOW = datetime(2026, 10, 12, 9, 30, tzinfo=UTC)


@dataclass
class Api:
    client: httpx.AsyncClient
    s3_store: DocumentStore
    bucket: str
    database: Database

    async def post(self, doc_id: str = "D1", version: int = 1, **fields: Any) -> httpx.Response:
        body = {
            "corpus": "esci",
            "doc_id": doc_id,
            "version": version,
            "title": "Mouse",
            "locale": "us",
        }
        body.update(fields)
        return await self.client.post("/v1/documents", content=json.dumps(body).encode())

    async def delete(self, doc_id: str = "D1", version: int = 1) -> httpx.Response:
        return await self.client.delete(f"/v1/documents/{doc_id}", params={"version": version})

    def object_bytes(self, key: str) -> bytes:
        return moto_client().get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def object_exists(self, key: str) -> bool:
        try:
            moto_client().head_object(Bucket=self.bucket, Key=key)
        except ClientError:
            return False
        return True

    def events(self, doc_id: str) -> list[Any]:
        return [
            doc_event_adapter.validate_json(json.dumps(row[4]))
            for row in outbox(self.database.direct_dsn, doc_id)
        ]


def request_for(doc_id: str = "D1", version: int = 1, **fields: Any) -> UpsertRequest:
    body = {
        "corpus": "esci",
        "doc_id": doc_id,
        "version": version,
        "title": "Mouse",
        "locale": "us",
    }
    body.update(fields)
    return UpsertRequest.model_validate(body)


@asynccontextmanager
async def serving(
    pool: AsyncConnectionPool,
    bucket: str,
    database: Database,
    *,
    s3_store: DocumentStore | None = None,
    pg_documents: Any = None,
    settings: IngestSettings | None = None,
) -> AsyncIterator[Api]:
    settings = settings or IngestSettings()
    s3_store = s3_store or make_store(moto_client(), bucket)[0]
    pg_documents = pg_documents or DocumentRepository(pool, settings.db_lock_timeout_s)
    service = IngestService(pg_documents, s3_store, clock=lambda: NOW)
    app = create_app(service, settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://api"
    ) as client:
        try:
            yield Api(client, s3_store, bucket, database)
        finally:
            s3_store.close()


@pytest.fixture
async def api(pool: AsyncConnectionPool, bucket: str, db: Database) -> AsyncIterator[Api]:
    async with serving(pool, bucket, db) as served:
        yield served


# kafka: every test gets its own topic, so messages from one test never reach another
KAFKA = os.environ.get(
    "TEST_KAFKA_BOOTSTRAP", IngestSettings.model_fields["kafka_bootstrap"].default
)


@asynccontextmanager
async def temporary_topic(prefix: str) -> AsyncIterator[str]:
    name = f"{prefix}.{uuid.uuid4().hex[:10]}"
    admin = AIOKafkaAdminClient(bootstrap_servers=KAFKA)
    await admin.start()
    try:
        await admin.create_topics([NewTopic(name=name, num_partitions=6, replication_factor=1)])
        yield name
    finally:
        with suppress(Exception):
            await admin.delete_topics([name])
        await admin.close()


@pytest.fixture
async def topic() -> AsyncIterator[str]:
    async with temporary_topic("doc.events.test") as name:
        yield name


@pytest.fixture
async def dlq_topic() -> AsyncIterator[str]:
    async with temporary_topic("doc.events.dlq.test") as name:
        yield name


def free_port() -> int:
    """A port nobody listens on right now, for a test that starts a server of its own."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
    return port


async def produce(
    topic_name: str, value: bytes | None, key: bytes | None = None, partition: int | None = None
) -> None:
    """Put one message on a topic, on a chosen partition when the test needs two apart."""
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA)
    await producer.start()
    try:
        await producer.send_and_wait(topic_name, value=value, key=key, partition=partition)
    finally:
        await producer.stop()


async def committed_offsets(group: str) -> dict[int, int]:
    """The offsets a group has committed, read through the admin api so the group is not joined."""
    admin = AIOKafkaAdminClient(bootstrap_servers=KAFKA)
    await admin.start()
    try:
        offsets = await admin.list_consumer_group_offsets(group)
    finally:
        await admin.close()
    return {tp.partition: meta.offset for tp, meta in offsets.items()}


async def eventually(check: Any, limit_s: float = 25.0, step: float = 0.1) -> None:
    """Ask again and again until the check is true, fail with a clear error when it never is."""
    deadline = time.monotonic() + limit_s
    while time.monotonic() < deadline:
        if await check():
            return
        await asyncio.sleep(step)
    raise AssertionError(f"still not true after {limit_s} s")


async def read_topic(
    name: str, expected: int, quiet_s: float = 0.7, limit_s: float = 20.0
) -> list[Any]:
    """Read until `expected` messages arrived, then keep listening for unexpected extras."""
    consumer = AIOKafkaConsumer(
        name,
        bootstrap_servers=KAFKA,
        group_id=None,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
    )
    await consumer.start()
    seen: list[Any] = []
    try:
        async with asyncio.timeout(limit_s):
            while len(seen) < expected:
                for batch in (await consumer.getmany(timeout_ms=500)).values():
                    seen.extend(batch)
        # nothing more should come: wait for a quiet moment before saying that is all there is
        quiet_until = time.monotonic() + quiet_s
        while time.monotonic() < quiet_until:
            for batch in (await consumer.getmany(timeout_ms=200)).values():
                seen.extend(batch)
                quiet_until = time.monotonic() + quiet_s
    finally:
        await consumer.stop()
    return seen


def add_events(
    dsn: str, count: int, doc_id: str = "D1", first_version: int = 1, tag: str = "x"
) -> list[int]:
    """Put `count` events in the outbox the way the api would, return their ids."""
    ids = []
    with psycopg.connect(dsn) as conn:
        for number in range(count):
            version = first_version + number
            row = conn.execute(
                "insert into outbox (doc_id, request_id, event_schema, payload) "
                "values (%s, %s, 'doc_event_v1', %s::jsonb) returning id",
                (
                    doc_id,
                    f"req-{tag}-{version}",
                    json.dumps({"doc_id": doc_id, "version": version, "tag": tag}),
                ),
            ).fetchone()
            assert row is not None
            ids.append(row[0])
    return ids


# qdrant: every test gets its own collection, with tiny dense vectors, and it is dropped afterwards
TEST_DIM = 4


@pytest.fixture
async def qdrant() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=EmbedSettings().qdrant_url, timeout=10)
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture(scope="session")
def shared_collection() -> Iterator[EmbedSettings]:
    # creating a collection and its indexes takes seconds, so one serves the whole session and
    # every test starts by wiping its points, the tests that need a differently shaped one build it
    settings = EmbedSettings(
        qdrant_collection=f"search_test_{uuid.uuid4().hex[:10]}", dense_dim=TEST_DIM
    )

    async def create() -> None:
        client = AsyncQdrantClient(url=settings.qdrant_url, timeout=10)
        try:
            await ensure_collection(client, settings)
        finally:
            await client.close()

    asyncio.run(create())
    try:
        yield settings
    finally:
        sync = QdrantClient(url=settings.qdrant_url, timeout=10)
        with suppress(Exception):
            sync.delete_collection(settings.qdrant_collection)
        sync.close()


@pytest.fixture
async def collection(qdrant: AsyncQdrantClient, shared_collection: EmbedSettings) -> EmbedSettings:
    # an empty filter matches every point
    await qdrant.delete(
        shared_collection.qdrant_collection,
        points_selector=models.FilterSelector(filter=models.Filter()),
        wait=True,
    )
    return shared_collection


async def read_points(
    client: AsyncQdrantClient, name: str, doc_id: str | None = None
) -> dict[uuid.UUID, dict[str, Any]]:
    """Every point of the collection (or of one product) with its payload, keyed by id."""
    where = None
    if doc_id is not None:
        where = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )
    found: dict[uuid.UUID, dict[str, Any]] = {}
    offset = None
    while True:
        points, offset = await client.scroll(
            name, scroll_filter=where, limit=100, offset=offset, with_payload=True
        )
        for point in points:
            found[uuid.UUID(str(point.id))] = dict(point.payload or {})
        if offset is None:
            return found


# the worker pipeline on the real stores, with the hash stub for the model


def words(count: int, tag: str) -> str:
    return " ".join(f"{tag}{n}" for n in range(count))


# with the stub's 64 token limit this is two chunks: the bullets, then the description
BASE: dict[str, Any] = {
    "title": "Acme mouse",
    "bullets": [words(20, "a"), words(20, "b")],
    "description": words(30, "d"),
}


@dataclass
class Rig:
    api: Api
    pipeline: Pipeline
    embedder: HashEmbedder
    pg_chunk_repo: ChunkRepository
    qdrant_index: ChunkIndex
    qdrant_client: AsyncQdrantClient
    embed_settings: EmbedSettings

    def pipeline_with(self, **parts: Any) -> Pipeline:
        return Pipeline(
            parts.get("pg_chunk_repo", self.pg_chunk_repo),
            parts.get("qdrant_index", self.qdrant_index),
            parts.get("s3_documents", self.api.s3_store),
            parts.get("embedder", self.embedder),
            self.embed_settings,
            parts.get("metrics"),
        )

    async def ingest(self, doc_id: str, version: int, **fields: Any) -> DocEvent:
        response = await self.api.post(doc_id, version, **(fields or BASE))
        assert response.status_code == 202
        return self.api.events(doc_id)[-1]  # type: ignore[no-any-return]

    async def remove(self, doc_id: str, version: int) -> DocEvent:
        assert (await self.api.delete(doc_id, version)).status_code == 202
        return self.api.events(doc_id)[-1]  # type: ignore[no-any-return]

    async def points(self, doc_id: str) -> dict[uuid.UUID, dict[str, Any]]:
        return await read_points(self.qdrant_client, self.embed_settings.qdrant_collection, doc_id)

    def row(self, doc_id: str) -> dict[str, Any]:
        found = document(self.api.database.direct_dsn, doc_id)
        assert found is not None
        return found

    async def untouched(self, doc_id: str) -> None:
        # nothing was committed and nothing is visible
        row = self.row(doc_id)
        assert (row["current"], row["indexed"]) == (None, None)
        snapshot = await self.pg_chunk_repo.snapshot("esci", doc_id)
        assert snapshot.current == {}
        assert snapshot.stale == frozenset()
        assert not any(p["state"] == "live" for p in (await self.points(doc_id)).values())


@pytest.fixture
async def rig(
    api: Api, pool: AsyncConnectionPool, qdrant: AsyncQdrantClient, collection: EmbedSettings
) -> Rig:
    pg_chunk_repo = ChunkRepository(pool)
    qdrant_index = ChunkIndex(qdrant, collection.qdrant_collection)
    embedder = HashEmbedder(dense_dim=TEST_DIM, max_tokens=64)
    pipeline = Pipeline(pg_chunk_repo, qdrant_index, api.s3_store, embedder, collection)
    return Rig(api, pipeline, embedder, pg_chunk_repo, qdrant_index, qdrant, collection)


def first_words(texts: list[str]) -> list[str]:
    # the second line of a chunk is where the body starts, the first word tells the chunks apart
    return [text.split("\n")[1].split()[0] for text in texts]
