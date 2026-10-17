"""The maintenance process on the real stack: it compacts, it is observable, it stops."""

import asyncio
import time

import httpx

from services.embed.maintenance import MaintenanceSettings, serve
from services.embed.settings import EmbedSettings

from .conftest import Database, Rig, eventually, free_port
from .test_compactor import FIVE, FOUR, run_sql, stale_rows


def settings_of(db: Database, collection: EmbedSettings, **changes: object) -> MaintenanceSettings:
    values: dict[str, object] = {
        "database_dsn": db.pooled_dsn,
        "qdrant_url": collection.qdrant_url,
        "qdrant_collection": collection.qdrant_collection,
        "stale_grace_s": 0.01,
        "compactor_interval_s": 0.2,
        "health_port": free_port(),
        **changes,
    }
    return MaintenanceSettings(**values)  # type: ignore[arg-type]


async def get(settings: MaintenanceSettings, path: str) -> httpx.Response:
    async with httpx.AsyncClient(timeout=5) as client:
        return await client.get(f"http://127.0.0.1:{settings.health_port}{path}")


async def says(settings: MaintenanceSettings, line: str) -> bool:
    try:
        return line in (await get(settings, "/metrics")).text
    except httpx.TransportError:  # the server is not up yet
        return False


async def test_the_process_deletes_stale_chunks_and_reports_what_it_does(
    rig: Rig, db: Database, collection: EmbedSettings
) -> None:
    await rig.pipeline.handle(await rig.ingest("P1", 1, **FIVE))
    await rig.pipeline.handle(await rig.ingest("P1", 2, **FOUR))
    assert len(stale_rows(rig, "P1")) == 1
    settings = settings_of(db, collection)
    stop = asyncio.Event()
    task = asyncio.create_task(serve(settings, stop))
    try:

        async def compacted() -> bool:
            return await says(settings, "embed_stale_chunks_deleted_total 1.0")

        await eventually(compacted)
        assert stale_rows(rig, "P1") == {}
        assert (await get(settings, "/readyz")).json() == {
            "ready": True,
            "checks": {"postgres": "ok", "qdrant": "ok"},
        }
        assert (await get(settings, "/healthz")).status_code == 200
        text = (await get(settings, "/metrics")).text
        assert 'maintenance_job_runs_total{job="compactor",outcome="ok"}' in text
        assert "embed_stale_chunks_oldest_age_seconds 0.0" in text
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=15)


async def test_an_idle_process_stops_at_once_and_not_after_its_interval(
    db: Database, collection: EmbedSettings
) -> None:
    settings = settings_of(db, collection, compactor_interval_s=60.0)
    stop = asyncio.Event()
    task = asyncio.create_task(serve(settings, stop))
    await eventually(lambda: says(settings, "maintenance_job_runs_total"))
    started = time.monotonic()
    stop.set()
    await asyncio.wait_for(task, timeout=15)
    assert time.monotonic() - started < 5  # not the sixty seconds of the interval


async def test_a_pass_that_cannot_reach_the_database_is_counted_and_the_process_goes_on(
    rig: Rig, db: Database, collection: EmbedSettings
) -> None:
    await rig.pipeline.handle(await rig.ingest("P2", 1, **FIVE))
    await rig.pipeline.handle(await rig.ingest("P2", 2, **FOUR))
    settings = settings_of(db, collection)
    stop = asyncio.Event()
    task = asyncio.create_task(serve(settings, stop))
    try:
        # the table is taken away for a moment: every pass fails, and then it comes back
        run_sql(rig, "alter table stale_chunks rename to stale_chunks_away")

        async def failing() -> bool:
            return await says(
                settings, 'maintenance_job_runs_total{job="compactor",outcome="error"}'
            )

        await eventually(failing)
        assert (await get(settings, "/healthz")).status_code == 200  # a failing job is not death
        run_sql(rig, "alter table stale_chunks_away rename to stale_chunks")

        async def compacted() -> bool:
            return await says(settings, "embed_stale_chunks_deleted_total 1.0")

        await eventually(compacted)
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=15)
