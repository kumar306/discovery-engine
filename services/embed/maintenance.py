import asyncio
import logging
import signal
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from typing import Protocol
from uuid import UUID

from prometheus_client import CollectorRegistry
from pydantic import PositiveFloat, PositiveInt, model_validator
from pydantic_settings import SettingsConfigDict

from libs.db.pool import create_pool
from libs.embedding.collection import qdrant_probe
from libs.observability.health import Heartbeat, create_health_app, run_health_server
from libs.observability.logging import configure_logging
from libs.observability.metrics import MaintenanceMetrics
from libs.observability.probes import postgres_probe
from libs.resilience.shutdown import run_until_stopped
from libs.settings import HealthSettings, PostgresSettings, QdrantSettings
from services.embed.qdrant import ChunkIndex, new_client
from services.embed.repository import ChunkRepository, ClaimedChunk

log = logging.getLogger("embed.maintenance")

# the compactor claims a stale queue row, deletes its point, then deletes the row, and holds no
# postgres lock while it talks to qdrant, the claim is the row's deletion_started_at
# a compactor that dies leaves the claim, and after the claim timeout the next pass takes it over
# that is only safe while a compactor that is merely slow cannot still be about to send its
# delete, so the claim timeout must be much longer than the qdrant timeout, the settings check it
MIN_TIMEOUT_RATIO = 10


class MaintenanceSettings(PostgresSettings, QdrantSettings, HealthSettings):
    model_config = SettingsConfigDict(env_prefix="MAINTENANCE_", env_file=None)

    # the api uses 8000, the worker 8081 and the relay 8083
    health_port: PositiveInt = 8082
    # a stale point is already invisible, so waiting costs only disk, and it covers a reader that
    # holds a chunk id from a search a moment ago
    stale_grace_s: PositiveFloat = 3600.0
    claim_timeout_s: PositiveFloat = 600.0
    compactor_batch_size: PositiveInt = 100
    compactor_interval_s: PositiveFloat = 30.0

    @model_validator(mode="after")
    def claim_timeout_is_much_longer_than_the_qdrant_timeout(self) -> "MaintenanceSettings":
        needed = self.qdrant_timeout_s * MIN_TIMEOUT_RATIO
        if self.claim_timeout_s < needed:
            raise ValueError(
                f"claim_timeout_s {self.claim_timeout_s:g} is below {needed:g}, which is "
                f"{MIN_TIMEOUT_RATIO} times qdrant_timeout_s {self.qdrant_timeout_s}"
            )
        return self


class StaleRepo(Protocol):
    async def claim_stale(
        self, *, grace_s: float, claim_timeout_s: float, limit: int
    ) -> list[ClaimedChunk]: ...
    async def forget_claimed(self, claimed: Sequence[ClaimedChunk]) -> None: ...
    async def oldest_stale_age_s(self) -> float: ...


class PointDeleter(Protocol):
    async def delete(self, chunk_ids: Sequence[UUID]) -> None: ...


class Compactor:
    def __init__(
        self,
        repo: StaleRepo,
        index: PointDeleter,
        *,
        grace_s: float,
        claim_timeout_s: float,
        batch_size: int,
        metrics: MaintenanceMetrics,
    ) -> None:
        self._repo = repo
        self._index = index
        self._grace_s = grace_s
        self._claim_timeout_s = claim_timeout_s
        self._batch_size = batch_size
        self._metrics = metrics

    async def compact_once(self) -> int:
        """One pass: claim a batch of due rows, delete their points, forget the rows."""
        claimed = await self._repo.claim_stale(
            grace_s=self._grace_s, claim_timeout_s=self._claim_timeout_s, limit=self._batch_size
        )
        if not claimed:
            return 0
        # no transaction is open here, a crash or an error leaves the claims in place and the
        # next pass takes them over after the claim timeout, deleting a point twice is harmless
        await self._index.delete([chunk.chunk_id for chunk in claimed])
        await self._repo.forget_claimed(claimed)
        self._metrics.stale_deleted.inc(len(claimed))
        return len(claimed)

    async def run(
        self,
        stop: asyncio.Event,
        heartbeat: Heartbeat,
        interval_s: float,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        wait = sleep or _stop_aware_sleep(stop)
        while not stop.is_set():
            heartbeat.beat()
            full_batch = False
            try:
                deleted = await self.compact_once()
                self._metrics.stale_oldest_age.set(await self._repo.oldest_stale_age_s())
            except Exception:
                # the claims stay, the next pass retries, a failing pass must not stop the process
                self._metrics.job_runs.labels(job="compactor", outcome="error").inc()
                log.exception("a compactor pass failed")
            else:
                self._metrics.job_runs.labels(job="compactor", outcome="ok").inc()
                self._metrics.job_last_success.labels(job="compactor").set(wall_clock())
                # a full batch means there is probably more waiting, so go on at once
                full_batch = deleted >= self._batch_size
            if not full_batch:
                await wait(interval_s)


def _stop_aware_sleep(stop: asyncio.Event) -> Callable[[float], Awaitable[None]]:
    async def pause(seconds: float) -> None:
        # a plain sleep that SIGTERM or SIGINT cuts short, so a stop does not wait out the interval
        with suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=seconds)

    return pause


async def serve(settings: MaintenanceSettings, stop: asyncio.Event) -> None:
    registry = CollectorRegistry()
    metrics = MaintenanceMetrics(registry)
    heartbeat = Heartbeat()
    pg_pool = create_pool(settings.database_dsn, settings.db_pool_max, settings.db_pool_timeout_s)
    await pg_pool.open(wait=True, timeout=10)
    qdrant_client = new_client(settings)
    try:
        health_app = create_health_app(
            probes={
                "postgres": postgres_probe(pg_pool),
                "qdrant": qdrant_probe(qdrant_client, settings.qdrant_collection),
            },
            heartbeat=heartbeat,
            registry=registry,
            liveness_max_age_s=settings.liveness_max_age_s,
            readiness_cache_s=settings.readiness_cache_s,
            probe_timeout_s=settings.probe_timeout_s,
        )
        async with run_health_server(health_app, settings.health_host, settings.health_port):
            compactor = Compactor(
                ChunkRepository(pg_pool, settings.db_lock_timeout_s),
                ChunkIndex(qdrant_client, settings.qdrant_collection),
                grace_s=settings.stale_grace_s,
                claim_timeout_s=settings.claim_timeout_s,
                batch_size=settings.compactor_batch_size,
                metrics=metrics,
            )
            await run_until_stopped(
                compactor.run(stop, heartbeat, settings.compactor_interval_s),
                stop,
                settings.shutdown_grace_s,
            )
    finally:
        await qdrant_client.close()
        await pg_pool.close()


async def main() -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for name in (signal.SIGTERM, signal.SIGINT):
        # windows cannot add signal handlers to the loop, there ctrl c raises instead
        with suppress(NotImplementedError):
            loop.add_signal_handler(name, stop.set)
    await serve(MaintenanceSettings(), stop)


if __name__ == "__main__":
    configure_logging()
    if sys.platform == "win32":
        # psycopg async cannot use the default windows loop, linux needs nothing
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
