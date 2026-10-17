"""The health server on a real port, and its manners towards the process that runs it."""

import asyncio
import signal
import time

import httpx
import pytest
from fastapi import FastAPI
from prometheus_client import CollectorRegistry

from libs.observability import health
from libs.observability.health import Heartbeat, create_health_app, run_health_server


def an_app() -> FastAPI:
    return create_health_app(
        probes={},
        heartbeat=Heartbeat(),
        registry=CollectorRegistry(),
        liveness_max_age_s=30.0,
        readiness_cache_s=5.0,
        probe_timeout_s=1.0,
    )


async def test_the_endpoints_answer_on_a_real_port_and_the_port_is_freed_afterwards() -> None:
    async with run_health_server(an_app(), "127.0.0.1", 0) as port, httpx.AsyncClient() as client:
        for path in ("/healthz", "/readyz", "/metrics"):
            assert (await client.get(f"http://127.0.0.1:{port}{path}")).status_code == 200
    with pytest.raises(httpx.TransportError):
        async with httpx.AsyncClient() as client:
            await client.get(f"http://127.0.0.1:{port}/healthz")


async def test_a_port_that_is_taken_is_an_error_not_a_silent_failure() -> None:
    async with run_health_server(an_app(), "127.0.0.1", 0) as port:
        with pytest.raises(OSError, match="health server"):
            async with run_health_server(an_app(), "127.0.0.1", port):
                pass


async def test_the_server_leaves_the_signal_handlers_of_the_process_alone() -> None:
    # the entry point owns SIGINT and SIGTERM, uvicorn would take them over by default
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    async with run_health_server(an_app(), "127.0.0.1", 0):
        during = {sig: signal.getsignal(sig) for sig in before}
    after = {sig: signal.getsignal(sig) for sig in before}
    assert during == before
    assert after == before


# a scraper that is in the middle of a request when the process is told to stop


def an_app_with_a_request_that_never_ends() -> FastAPI:
    app = an_app()

    @app.get("/slow")
    async def slow() -> None:
        await asyncio.sleep(60)

    return app


async def seconds_to_stop_with_a_request_in_flight() -> float:
    app = an_app_with_a_request_that_never_ends()
    started = 0.0
    async with httpx.AsyncClient(timeout=30) as client:
        async with run_health_server(app, "127.0.0.1", 0) as port:
            pending = asyncio.create_task(client.get(f"http://127.0.0.1:{port}/slow"))
            await asyncio.sleep(0.3)  # the request has arrived and is being served
            started = time.monotonic()
        took = time.monotonic() - started
        pending.cancel()
    return took


async def test_a_request_in_flight_cannot_delay_the_stop_beyond_the_graceful_cap() -> None:
    # uvicorn waits for open requests, so without its cap this would wait for the full minute
    assert await seconds_to_stop_with_a_request_in_flight() < 4.0


async def test_a_server_that_does_not_stop_in_time_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(health, "STOP_TIMEOUT_S", 0.1)
    # the graceful cap is two seconds, the stop timeout is shorter, so the cancel is what ends it
    assert await seconds_to_stop_with_a_request_in_flight() < 1.0
