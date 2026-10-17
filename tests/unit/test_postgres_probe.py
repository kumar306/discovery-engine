"""The Postgres readiness probe, shared by every process that talks to the database."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from libs.observability.probes import postgres_probe


class FakeConnection:
    def __init__(self, queries: list[str]) -> None:
        self._queries = queries

    async def execute(self, sql: str) -> None:
        self._queries.append(sql)


class FakePool:
    def __init__(self, error: Exception | None = None) -> None:
        self.queries: list[str] = []
        self._error = error

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[FakeConnection]:
        if self._error:
            raise self._error
        yield FakeConnection(self.queries)


async def test_postgres_is_ready_when_a_connection_answers_a_query() -> None:
    pool = FakePool()
    await postgres_probe(pool)()  # type: ignore[arg-type]
    assert pool.queries == ["select 1"]


async def test_postgres_is_not_ready_when_no_connection_can_be_had() -> None:
    probe = postgres_probe(FakePool(ConnectionError("down")))  # type: ignore[arg-type]
    with pytest.raises(ConnectionError):
        await probe()
