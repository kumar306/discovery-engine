"""The relay's loop: when it sleeps, when it backs off, when it stops. No database, no kafka.

publish_batch is replaced by a script so each test can say exactly what the batches did.
"""

import asyncio
import random
from typing import Any

import pytest

from libs.resilience.retry import RetryPolicy
from services.ingest.relay import Relay


class MaxRandom(random.Random):
    """Always draws the top of the range, so the ceiling of each wait can be asserted."""

    def uniform(self, a: float, b: float) -> float:
        return b


class ScriptedRelay(Relay):
    """Each call takes the next outcome: a number is a published batch, an exception is raised."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.stop = asyncio.Event()
        self.delays: list[float] = []
        super().__init__(
            pool=None,  # type: ignore[arg-type]
            publisher=None,  # type: ignore[arg-type]
            batch_size=100,
            poll_interval_s=0.5,
            sleep=self._record,
            rng=MaxRandom(),
            policy=RetryPolicy(base_delay=0.1, max_delay=5.0),
        )

    async def _record(self, seconds: float) -> None:
        self.delays.append(seconds)

    async def publish_batch(self) -> int:
        outcome = self.script.pop(0)
        if not self.script:
            self.stop.set()  # the script is used up, the loop ends after this batch
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, int)
        return outcome

    async def go(self) -> list[float]:
        await self.run(self.stop)
        return self.delays


BOOM = RuntimeError("kafka said no")


async def test_an_empty_outbox_makes_the_relay_sleep_instead_of_asking_again_at_once() -> None:
    assert await ScriptedRelay([0, 0, 0]).go() == [0.5, 0.5, 0.5]


async def test_a_full_batch_is_followed_by_the_next_one_without_any_sleep() -> None:
    assert await ScriptedRelay([100, 100, 0]).go() == [0.5]


async def test_failures_back_off_with_growing_waits_and_the_loop_survives_them() -> None:
    # three failures, then a batch, then an empty look: waits of 0.1, 0.2, 0.4 and the idle sleep
    assert await ScriptedRelay([BOOM, BOOM, BOOM, 5, 0]).go() == [0.1, 0.2, 0.4, 0.5]


async def test_one_good_batch_resets_the_backoff() -> None:
    assert await ScriptedRelay([BOOM, 5, BOOM, 0]).go() == [0.1, 0.1, 0.5]


async def test_the_wait_grows_up_to_the_cap_and_stays_there() -> None:
    delays = await ScriptedRelay([BOOM] * 10).go()
    assert delays == [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 5.0, 5.0, 5.0, 5.0]


async def test_a_relay_that_is_already_stopped_never_touches_the_outbox() -> None:
    relay = ScriptedRelay([])  # an empty script would raise if a batch were ever attempted
    relay.stop.set()
    assert await relay.go() == []


async def test_cancellation_is_not_swallowed_by_the_failure_handling() -> None:
    relay = ScriptedRelay([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await relay.go()
