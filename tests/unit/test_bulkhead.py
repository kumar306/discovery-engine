"""Bulkhead: at most N calls in flight, the rest wait a little and then get shed."""

import asyncio

import pytest

from libs.resilience.bulkhead import Bulkhead, Overloaded


async def settle() -> None:
    # let every ready task run until it parks on a gate or a permit
    for _ in range(10):
        await asyncio.sleep(0)


async def test_at_most_the_limit_run_at_once() -> None:
    bulkhead = Bulkhead(limit=2, wait_timeout=5.0)
    gate = asyncio.Event()
    running = 0
    peak = 0

    async def work() -> None:
        nonlocal running, peak
        async with bulkhead:
            running += 1
            peak = max(peak, running)
            await gate.wait()
            running -= 1

    tasks = [asyncio.create_task(work()) for _ in range(5)]
    await settle()
    assert peak == 2
    gate.set()
    await asyncio.gather(*tasks)
    assert (peak, running, bulkhead.rejected) == (2, 0, 0)


async def test_a_caller_that_waits_too_long_is_shed_and_counted() -> None:
    bulkhead = Bulkhead(limit=1, wait_timeout=0.01)
    holder_gate = asyncio.Event()

    async def holder() -> None:
        async with bulkhead:
            await holder_gate.wait()

    task = asyncio.create_task(holder())
    await settle()
    with pytest.raises(Overloaded):
        async with bulkhead:
            pytest.fail("the body must not run without a permit")
    assert bulkhead.rejected == 1
    holder_gate.set()
    await task


async def test_a_waiter_gets_in_when_a_permit_frees_before_its_timeout() -> None:
    bulkhead = Bulkhead(limit=1, wait_timeout=5.0)
    holder_gate = asyncio.Event()
    entered = asyncio.Event()

    async def holder() -> None:
        async with bulkhead:
            await holder_gate.wait()

    async def waiter() -> None:
        async with bulkhead:
            entered.set()

    holding = asyncio.create_task(holder())
    await settle()
    waiting = asyncio.create_task(waiter())
    await settle()
    assert not entered.is_set()
    holder_gate.set()
    await asyncio.gather(holding, waiting)
    assert entered.is_set()
    assert bulkhead.rejected == 0


async def test_the_permit_comes_back_after_an_exception() -> None:
    bulkhead = Bulkhead(limit=1, wait_timeout=0.01)
    with pytest.raises(ValueError, match="boom"):
        async with bulkhead:
            raise ValueError("boom")
    async with bulkhead:  # would raise Overloaded if the permit had leaked
        pass


async def test_the_permit_comes_back_after_cancellation() -> None:
    bulkhead = Bulkhead(limit=1, wait_timeout=0.01)

    async def holder() -> None:
        async with bulkhead:
            await asyncio.Event().wait()

    task = asyncio.create_task(holder())
    await settle()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with bulkhead:
        pass


async def test_a_shed_caller_does_not_leak_or_steal_a_permit() -> None:
    bulkhead = Bulkhead(limit=1, wait_timeout=0.01)
    holder_gate = asyncio.Event()

    async def holder() -> None:
        async with bulkhead:
            await holder_gate.wait()

    task = asyncio.create_task(holder())
    await settle()
    for _ in range(3):
        with pytest.raises(Overloaded):
            async with bulkhead:
                pass
    holder_gate.set()
    await task
    async with bulkhead:  # exactly one permit exists again
        with pytest.raises(Overloaded):
            async with bulkhead:
                pass
