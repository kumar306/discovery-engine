"""Stopping a loop: let it finish the message in hand, but never wait for it forever."""

import asyncio
import time

import pytest

from libs.resilience.shutdown import run_until_stopped


async def test_a_loop_that_finishes_its_message_after_stop_is_not_cancelled() -> None:
    stop = asyncio.Event()
    finished: list[str] = []

    async def loop() -> None:
        await stop.wait()
        await asyncio.sleep(0.05)  # the message in hand
        finished.append("message done")

    task = asyncio.create_task(run_until_stopped(loop(), stop, grace_s=5))
    await asyncio.sleep(0.01)
    stop.set()
    assert await asyncio.wait_for(task, 5) is True  # it ended by itself
    assert finished == ["message done"]


async def test_a_loop_that_ignores_stop_is_cancelled_when_the_grace_period_is_over() -> None:
    stop = asyncio.Event()
    cleaned_up: list[str] = []

    async def stuck() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cleaned_up.append("cancelled")  # a finally block in the real loop would run here
            raise

    task = asyncio.create_task(run_until_stopped(stuck(), stop, grace_s=0.1))
    await asyncio.sleep(0.01)
    started = time.monotonic()
    stop.set()
    assert await asyncio.wait_for(task, 5) is False  # it had to be cancelled
    assert time.monotonic() - started < 2
    assert cleaned_up == ["cancelled"]


async def test_the_grace_period_is_a_wait_not_a_delay() -> None:
    stop = asyncio.Event()

    async def quick() -> None:
        await stop.wait()

    task = asyncio.create_task(run_until_stopped(quick(), stop, grace_s=30))
    await asyncio.sleep(0.01)
    started = time.monotonic()
    stop.set()
    await asyncio.wait_for(task, 5)
    assert time.monotonic() - started < 1  # a loop that ends at once does not wait out the 30 s


async def test_an_error_in_the_loop_is_raised_to_the_caller() -> None:
    async def broken() -> None:
        raise RuntimeError("the consumer died")

    with pytest.raises(RuntimeError, match="the consumer died"):
        await asyncio.wait_for(run_until_stopped(broken(), asyncio.Event(), grace_s=5), 5)


async def test_a_loop_that_ends_without_a_stop_is_fine_and_leaves_no_task_behind() -> None:
    async def short() -> None:
        await asyncio.sleep(0.01)

    before = len(asyncio.all_tasks())
    assert await asyncio.wait_for(run_until_stopped(short(), asyncio.Event(), grace_s=5), 5)
    assert len(asyncio.all_tasks()) == before  # the task that watched for stop is gone too


async def test_when_the_caller_is_cancelled_the_loop_is_cancelled_with_it() -> None:
    stop = asyncio.Event()
    cleaned_up: list[str] = []

    async def loop() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cleaned_up.append("cancelled")
            raise

    task = asyncio.create_task(run_until_stopped(loop(), stop, grace_s=5))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    await asyncio.sleep(0.01)
    assert cleaned_up == ["cancelled"]
