"""Fixtures for tests that need the compose stack (run make up first).

Databases are created through the direct postgres port because create and drop database cannot run
inside a pooled transaction. Everything the application does goes through pgbouncer, as in prod.
"""

import os
import random
import uuid
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import psycopg
import pytest
from botocore.config import Config
from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from libs.db.migrate import migrate
from libs.db.pool import create_pool
from libs.resilience.bulkhead import Bulkhead
from libs.resilience.retry import RetryPolicy
from services.ingest.settings import IngestSettings
from services.ingest.storage import DocumentStore

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
        conn.execute("truncate documents, outbox restart identity")
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
