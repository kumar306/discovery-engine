"""Fixtures for tests that need the compose stack (run make up first).

Databases are created through the direct postgres port because create and drop database cannot run
inside a pooled transaction. Everything the application does goes through pgbouncer, as in prod.
"""

import os
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from libs.db.migrate import migrate
from libs.db.pool import create_pool
from services.ingest.settings import IngestSettings

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
