"""The loop around the compactor: when it runs again, what it counts, and what it survives."""

import asyncio
from typing import Any
from uuid import uuid4

from prometheus_client import CollectorRegistry

from libs.observability.health import Heartbeat
from libs.observability.metrics import MaintenanceMetrics
from services.embed.maintenance import Compactor
from services.embed.repository import ClaimedChunk


def chunks(count: int) -> list[ClaimedChunk]:
    return [ClaimedChunk("esci", "D1", uuid4()) for _ in range(count)]


class FakeRepo:
    """claim_stale takes the next scripted answer: a list of claimed chunks, or an error."""

    def __init__(self, *claims: list[ClaimedChunk] | Exception, age: float = 0.0) -> None:
        self.claims = list(claims)
        self.forgotten: list[ClaimedChunk] = []
        self.age = age

    async def claim_stale(self, *, grace_s: float, claim_timeout_s: float, limit: int) -> Any:
        outcome = self.claims.pop(0) if self.claims else []
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def forget_claimed(self, claimed: list[ClaimedChunk]) -> None:
        self.forgotten.extend(claimed)

    async def oldest_stale_age_s(self) -> float:
        return self.age


class FakeIndex:
    def __init__(self) -> None:
        self.deleted: list[Any] = []

    async def delete(self, chunk_ids: Any) -> None:
        self.deleted.extend(chunk_ids)


class Loop:
    def __init__(self, repo: FakeRepo, passes: int, batch: int = 100) -> None:
        self.registry = CollectorRegistry()
        self.clock = [1000.0]
        self.heartbeat = Heartbeat(lambda: self.clock[0])
        self.stop = asyncio.Event()
        self.waits: list[float] = []
        self.ages_at_each_pass: list[float] = []
        self._passes_left = passes
        self.compactor = Compactor(
            repo,  # type: ignore[arg-type]
            FakeIndex(),  # type: ignore[arg-type]
            grace_s=3600,
            claim_timeout_s=600,
            batch_size=batch,
            metrics=MaintenanceMetrics(self.registry),
        )

    async def go(self) -> None:
        original = self.compactor.compact_once

        async def counted() -> int:
            self.ages_at_each_pass.append(self.heartbeat.age())
            self.clock[0] += 10
            self._passes_left -= 1
            if self._passes_left <= 0:
                self.stop.set()
            return await original()

        self.compactor.compact_once = counted  # type: ignore[method-assign]

        async def sleep(seconds: float) -> None:
            self.waits.append(seconds)

        await self.compactor.run(
            self.stop, self.heartbeat, interval_s=30.0, sleep=sleep, wall_clock=lambda: 5555.0
        )

    def value(self, name: str, **labels: str) -> float | None:
        return self.registry.get_sample_value(name, labels)


async def test_an_empty_queue_is_looked_at_once_per_interval() -> None:
    loop = Loop(FakeRepo([], [], []), passes=3)
    await loop.go()
    assert loop.waits == [30.0, 30.0, 30.0]
    assert loop.value("maintenance_job_runs_total", job="compactor", outcome="ok") == 3


async def test_a_full_batch_is_followed_by_the_next_pass_without_waiting() -> None:
    loop = Loop(FakeRepo(chunks(5), chunks(5), chunks(2)), passes=3, batch=5)
    await loop.go()
    assert loop.waits == [30.0]  # two full batches in a row, then a short one, then the wait
    assert loop.value("embed_stale_chunks_deleted_total") == 12


async def test_a_failing_pass_is_counted_logged_and_the_loop_goes_on() -> None:
    loop = Loop(FakeRepo(chunks(2), ConnectionError("qdrant down"), chunks(1)), passes=3)
    await loop.go()
    assert loop.value("maintenance_job_runs_total", job="compactor", outcome="ok") == 2
    assert loop.value("maintenance_job_runs_total", job="compactor", outcome="error") == 1
    assert loop.value("embed_stale_chunks_deleted_total") == 3
    assert len(loop.waits) == 3  # it waited after the failure as well


async def test_the_time_of_the_last_good_pass_does_not_move_on_a_failure() -> None:
    loop = Loop(FakeRepo(ConnectionError("down")), passes=1)
    await loop.go()
    assert loop.value("maintenance_job_last_success_seconds", job="compactor") is None
    good = Loop(FakeRepo([]), passes=1)
    await good.go()
    assert good.value("maintenance_job_last_success_seconds", job="compactor") == 5555.0


async def test_the_oldest_age_gauge_follows_the_queue() -> None:
    loop = Loop(FakeRepo([], age=7200.0), passes=1)
    await loop.go()
    assert loop.value("embed_stale_chunks_oldest_age_seconds") == 7200.0


async def test_the_loop_beats_on_every_pass() -> None:
    loop = Loop(FakeRepo([], [], []), passes=3)
    loop.clock[0] += 500  # a long time since the heartbeat was created
    await loop.go()
    assert loop.ages_at_each_pass == [0, 0, 0]


async def test_a_loop_told_to_stop_does_not_start_another_pass() -> None:
    loop = Loop(FakeRepo([]), passes=1)
    loop.stop.set()
    await loop.go()
    assert loop.ages_at_each_pass == []
