from psycopg_pool import AsyncConnectionPool

from libs.observability.health import Probe


def postgres_probe(pg_pool: AsyncConnectionPool) -> Probe:
    async def probe() -> None:
        # a connection from the pool and a query that reaches the database through pgbouncer,
        # so a pool that is exhausted and a database that is down both fail here
        async with pg_pool.connection() as conn:
            await conn.execute("select 1")

    return probe
