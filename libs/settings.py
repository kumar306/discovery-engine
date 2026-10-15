from typing import Self

from pydantic import PositiveFloat, PositiveInt, model_validator
from pydantic_settings import BaseSettings

# the settings of the three systems every service talks to, written once
# a service builds its own settings class from the ones it needs and gives it its own env prefix,
# so the ingest service reads INGEST_DATABASE_DSN and the worker reads EMBED_DATABASE_DSN, and
# nobody reads another service's variables

# the bulkhead may use at most this share of the s3 thread pool
# the rest stays free so the event loop never waits on a busy pool
BULKHEAD_SHARE = 0.8


class PostgresSettings(BaseSettings):
    # postgres, always reached through pgbouncer in transaction mode
    database_dsn: str = "postgresql://search:search_dev_only@127.0.0.1:6432/search"
    db_pool_max: PositiveInt = 8
    db_pool_timeout_s: PositiveFloat = 2.0
    # how long a write may wait for a row lock held by another transaction before it gives up
    db_lock_timeout_s: PositiveFloat = 2.0


class S3Settings(BaseSettings):
    # s3, the endpoint is only set for the local moto stand in
    s3_endpoint_url: str | None = "http://127.0.0.1:5000"
    s3_bucket: str = "search-documents"
    s3_region: str = "us-east-1"
    s3_timeout_s: PositiveFloat = 3.0
    s3_threads: PositiveInt = 10
    s3_bulkhead_limit: PositiveInt = 8
    s3_bulkhead_wait_s: PositiveFloat = 0.5

    @model_validator(mode="after")
    def bulkhead_fits_inside_the_thread_pool(self) -> Self:
        allowed = int(self.s3_threads * BULKHEAD_SHARE)
        if self.s3_bulkhead_limit > allowed:
            raise ValueError(
                f"s3_bulkhead_limit {self.s3_bulkhead_limit} is above {allowed}, "
                f"which is {BULKHEAD_SHARE:.0%} of s3_threads {self.s3_threads}"
            )
        return self


class KafkaSettings(BaseSettings):
    kafka_bootstrap: str = "127.0.0.1:19092"
    topic_events: str = "doc.events"
    topic_dlq: str = "doc.events.dlq"
