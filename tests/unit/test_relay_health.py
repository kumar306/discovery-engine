"""What the relay gives the platform: a heartbeat, a producer ready signal, a backlog gauge."""

import asyncio
from typing import Any

import pytest
from prometheus_client import CollectorRegistry

from libs.observability.health import Heartbeat
from libs.observability.metrics import RelayMetrics
from services.ingest.probes import PublisherReadyProbe
from services.ingest.relay import report_backlog
from tests.unit.test_relay_loop import ScriptedRelay

# the producer is ready once it exists


async def test_not_ready_before_a_producer_has_started() -> None:
    with pytest.raises(RuntimeError, match="not connected"):
        await PublisherReadyProbe()()


async def test_ready_once_the_loop_has_marked_the_producer_as_connected() -> None:
    probe = PublisherReadyProbe()
    probe.mark_connected()
    assert await probe() is None


# the heartbeat of the loop


async def test_the_relay_loop_beats_on_every_turn() -> None:
    clock = [100.0]
    heartbeat = Heartbeat(lambda: clock[0])
    relay = ScriptedRelay([3, 0, 5, 0], heartbeat=heartbeat)
    ages: list[float] = []
    original = relay.publish_batch

    async def look_then_let_time_pass() -> int:
        ages.append(heartbeat.age())  # the moment just after the loop beat
        clock[0] += 10
        return await original()

    relay.publish_batch = look_then_let_time_pass  # type: ignore[method-assign]
    clock[0] += 500  # a long time since the heartbeat was created
    await relay.run(relay.stop)
    assert ages == [0, 0, 0, 0]


# the backlog gauge


class Reads:
    """A stand in for the database count: each call takes the next answer or raises it."""

    def __init__(self, *answers: int | Exception) -> None:
        self.answers = list(answers)
        self.calls = 0

    async def __call__(self) -> int:
        self.calls += 1
        answer = self.answers.pop(0) if self.answers else 0
        if isinstance(answer, Exception):
            raise answer
        return answer


def gauge(registry: CollectorRegistry) -> float | None:
    return registry.get_sample_value("outbox_backlog")


async def run_reporter(
    reads: Reads, turns: int, registry: CollectorRegistry
) -> tuple[list[float], list[float | None]]:
    stop = asyncio.Event()
    waits: list[float] = []
    seen: list[float | None] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        seen.append(gauge(registry))  # what the gauge says when the reporter goes to sleep
        if len(waits) >= turns:
            stop.set()

    await report_backlog(reads, RelayMetrics(registry).outbox_backlog, 5.0, stop, sleep)
    return waits, seen


async def test_the_gauge_follows_the_count_once_per_interval() -> None:
    waits, seen = await run_reporter(Reads(40, 25, 0), turns=3, registry=CollectorRegistry())
    assert seen == [40, 25, 0]
    assert waits == [5.0, 5.0, 5.0]


async def test_a_count_that_fails_leaves_the_last_value_and_the_reporter_goes_on() -> None:
    errors: Any = ConnectionError("the database did not answer")
    registry = CollectorRegistry()
    waits, seen = await run_reporter(Reads(40, errors, 7), turns=3, registry=registry)
    assert seen == [40, 40, 7]  # the failure did not reset the gauge and did not stop the loop
    assert len(waits) == 3


async def test_the_reporter_stops_when_told_to() -> None:
    stop = asyncio.Event()
    stop.set()
    reads = Reads(1)
    await report_backlog(reads, RelayMetrics(CollectorRegistry()).outbox_backlog, 5.0, stop)
    assert reads.calls == 0
