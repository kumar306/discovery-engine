from psycopg_pool import AsyncConnectionPool


def create_pool(dsn: str, max_size: int, timeout: float) -> AsyncConnectionPool:
    return AsyncConnectionPool(
        dsn,
        min_size=1,
        max_size=max_size,
        # how long a caller waits for a free connection before PoolTimeout
        # the api turns that into a 503
        timeout=timeout,
        kwargs={
            # pgbouncer in transaction mode lends a different server connection to every transaction
            # so server side prepared statements are off, a statement is never kept on a connection
            "prepare_threshold": None,
            # autocommit is on so each conn.transaction() block is a real transaction
            # and a plain statement outside a block runs and commits on its own
            "autocommit": True,
        },
        open=False,
    )
