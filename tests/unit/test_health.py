"""The three endpoints of a process: alive, ready, and its metrics."""

import asyncio
import time

import httpx
import pytest
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry

from libs.observability.health import Heartbeat, Probe, create_health_app
from libs.observability.metrics import EmbedMetrics


class FakeClock:
    """Time that only moves when a test says so."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


async def fine() -> None:
    return None


def failing(error: Exception) -> Probe:
    async def probe() -> None:
        raise error

    return probe


def counting(calls: list[str], name: str) -> Probe:
    async def probe() -> None:
        calls.append(name)

    return probe


def build(
    clock: FakeClock,
    probes: dict[str, Probe] | None = None,
    registry: CollectorRegistry | None = None,
    probe_timeout_s: float = 1.0,
) -> tuple[FastAPI, Heartbeat]:
    heartbeat = Heartbeat(clock)
    app = create_health_app(
        probes={"postgres": fine} if probes is None else probes,
        heartbeat=heartbeat,
        registry=registry or CollectorRegistry(),
        liveness_max_age_s=30.0,
        readiness_cache_s=5.0,
        probe_timeout_s=probe_timeout_s,
        clock=clock,
    )
    return app, heartbeat


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


async def get(app: FastAPI, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://health") as client:
        return await client.get(path)


# liveness: is the loop still turning


async def test_alive_while_the_loop_keeps_beating(clock: FakeClock) -> None:
    app, heartbeat = build(clock)
    assert (await get(app, "/healthz")).status_code == 200  # before the first beat counts as alive
    clock.now += 29
    assert (await get(app, "/healthz")).status_code == 200
    heartbeat.beat()
    clock.now += 29
    assert (await get(app, "/healthz")).status_code == 200


async def test_the_limit_itself_is_still_alive(clock: FakeClock) -> None:
    app, _ = build(clock)
    clock.now += 30  # exactly the limit
    assert (await get(app, "/healthz")).status_code == 200
    clock.now += 0.5
    assert (await get(app, "/healthz")).status_code == 503


async def test_not_alive_when_the_loop_stopped_beating(clock: FakeClock) -> None:
    app, heartbeat = build(clock)
    clock.now += 31
    response = await get(app, "/healthz")
    assert response.status_code == 503
    assert response.json() == {"alive": False}
    heartbeat.beat()  # the loop came back
    assert (await get(app, "/healthz")).status_code == 200


# readiness: can the process reach what it needs


async def test_ready_when_every_probe_answers(clock: FakeClock) -> None:
    app, _ = build(clock, {"postgres": fine, "qdrant": fine, "kafka": fine})
    response = await get(app, "/readyz")
    assert response.status_code == 200
    assert response.json() == {
        "ready": True,
        "checks": {"postgres": "ok", "qdrant": "ok", "kafka": "ok"},
    }


async def test_not_ready_when_one_probe_fails_and_the_message_is_not_leaked(
    clock: FakeClock,
) -> None:
    secret = RuntimeError("password=hunter2 at postgres://user@host")
    app, _ = build(clock, {"postgres": failing(secret), "qdrant": fine})
    response = await get(app, "/readyz")
    assert response.status_code == 503
    assert response.json() == {
        "ready": False,
        "checks": {"postgres": "failed: RuntimeError", "qdrant": "ok"},
    }
    assert "hunter2" not in response.text  # the type of the error is shown, never its text


async def test_a_probe_that_hangs_is_cut_by_its_timeout(clock: FakeClock) -> None:
    async def hangs() -> None:
        await asyncio.sleep(30)

    app, _ = build(clock, {"qdrant": hangs}, probe_timeout_s=0.05)
    started = time.monotonic()
    response = await get(app, "/readyz")
    assert time.monotonic() - started < 2
    assert response.status_code == 503
    assert response.json()["checks"] == {"qdrant": "timeout"}


async def test_the_probes_run_at_the_same_time_not_one_after_another(clock: FakeClock) -> None:
    a_started, b_started = asyncio.Event(), asyncio.Event()

    def waits_for(mine: asyncio.Event, other: asyncio.Event) -> Probe:
        async def probe() -> None:
            mine.set()
            # this can only finish if the other probe has started too
            await asyncio.wait_for(other.wait(), timeout=2)

        return probe

    app, _ = build(
        clock, {"a": waits_for(a_started, b_started), "b": waits_for(b_started, a_started)}
    )
    assert (await get(app, "/readyz")).status_code == 200


async def test_the_answer_is_cached_for_a_few_seconds(clock: FakeClock) -> None:
    calls: list[str] = []
    app, _ = build(clock, {"postgres": counting(calls, "postgres")})
    for _ in range(5):
        await get(app, "/readyz")
    assert calls == ["postgres"]  # five requests, one real check
    clock.now += 6
    await get(app, "/readyz")
    assert calls == ["postgres", "postgres"]


async def test_a_failure_is_cached_too_so_a_down_dependency_is_not_hammered(
    clock: FakeClock,
) -> None:
    attempts = 0

    async def down() -> None:
        nonlocal attempts
        attempts += 1
        raise ConnectionError

    app, _ = build(clock, {"qdrant": down})
    for _ in range(4):
        assert (await get(app, "/readyz")).status_code == 503
    assert attempts == 1


async def test_ready_again_after_the_dependency_comes_back(clock: FakeClock) -> None:
    healthy = False

    async def flaky() -> None:
        if not healthy:
            raise ConnectionError

    app, _ = build(clock, {"qdrant": flaky})
    assert (await get(app, "/readyz")).status_code == 503
    healthy = True
    assert (await get(app, "/readyz")).status_code == 503  # still the cached answer
    clock.now += 6
    assert (await get(app, "/readyz")).status_code == 200


async def test_requests_that_arrive_together_share_one_check(clock: FakeClock) -> None:
    calls: list[str] = []
    release = asyncio.Event()

    async def slow() -> None:
        calls.append("slow")
        await release.wait()

    app, _ = build(clock, {"postgres": slow})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://health") as client:
        pending = [asyncio.create_task(client.get("/readyz")) for _ in range(8)]
        await asyncio.sleep(0.05)
        release.set()
        responses = await asyncio.gather(*pending)
    assert all(r.status_code == 200 for r in responses)
    assert calls == ["slow"]


async def test_no_probes_means_ready(clock: FakeClock) -> None:
    app, _ = build(clock, {})
    assert (await get(app, "/readyz")).json() == {"ready": True, "checks": {}}


# metrics


async def test_the_metrics_endpoint_serves_the_registry_in_the_prometheus_format(
    clock: FakeClock,
) -> None:
    registry = CollectorRegistry()
    EmbedMetrics(registry).events.labels(result="applied").inc()
    app, _ = build(clock, registry=registry)
    response = await get(app, "/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == CONTENT_TYPE_LATEST
    assert 'embed_events_total{result="applied"} 1.0' in response.text


async def test_there_are_no_docs_pages_on_the_health_port(clock: FakeClock) -> None:
    app, _ = build(clock)
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert (await get(app, path)).status_code == 404
