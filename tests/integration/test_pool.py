"""The connection pool settings: no server side prepared statements, and autocommit semantics."""

import psycopg
import pytest

from libs.db.pool import create_pool

from .conftest import Database


async def prepared_count_after_repeats(conn: psycopg.AsyncConnection) -> int:
    # psycopg prepares a statement on the server once it has been run a few times
    for _ in range(20):
        await conn.execute("select 1")
    row = await (await conn.execute("select count(*) from pg_prepared_statements")).fetchone()
    assert row is not None
    return int(row[0])


async def test_the_pool_never_prepares_statements_on_the_server(db: Database) -> None:
    pool = create_pool(db.direct_dsn, max_size=1, timeout=2.0)
    await pool.open(wait=True, timeout=10)
    try:
        async with pool.connection() as conn:
            assert await prepared_count_after_repeats(conn) == 0
    finally:
        await pool.close()


async def test_a_default_connection_does_prepare_them_which_is_why_we_switch_it_off(
    db: Database,
) -> None:
    # the control for the test above: without the setting the same loop does leave statements behind
    async with await psycopg.AsyncConnection.connect(db.direct_dsn) as conn:
        assert await prepared_count_after_repeats(conn) > 0


async def test_a_plain_statement_commits_on_its_own(db: Database) -> None:
    pool = create_pool(db.pooled_dsn, max_size=2, timeout=2.0)
    await pool.open(wait=True, timeout=10)
    try:
        async with pool.connection() as conn:
            await conn.execute(
                "insert into documents (corpus, doc_id, latest_version) values ('esci', 'AUTO', 1)"
            )
        with psycopg.connect(db.direct_dsn) as other:
            row = other.execute("select doc_id from documents where doc_id = 'AUTO'").fetchone()
        assert row == ("AUTO",)
    finally:
        await pool.close()


async def insert_then_fail(conn: psycopg.AsyncConnection) -> None:
    async with conn.transaction():
        await conn.execute(
            "insert into documents (corpus, doc_id, latest_version) values ('esci', 'ROLLED', 1)"
        )
        raise RuntimeError("boom")


async def test_a_transaction_block_rolls_back_as_a_whole_when_its_body_fails(db: Database) -> None:
    pool = create_pool(db.pooled_dsn, max_size=2, timeout=2.0)
    await pool.open(wait=True, timeout=10)
    try:
        async with pool.connection() as conn:
            with pytest.raises(RuntimeError):
                await insert_then_fail(conn)
        with psycopg.connect(db.direct_dsn) as other:
            row = other.execute("select doc_id from documents where doc_id = 'ROLLED'").fetchone()
        assert row is None
    finally:
        await pool.close()


async def test_a_statement_outside_a_transaction_block_leaves_no_transaction_open(
    db: Database,
) -> None:
    # without autocommit psycopg would quietly open a transaction here and hold it until the end
    pool = create_pool(db.pooled_dsn, max_size=1, timeout=2.0)
    await pool.open(wait=True, timeout=10)
    try:
        async with pool.connection() as conn:
            await conn.execute("select 1")
            assert conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
    finally:
        await pool.close()
