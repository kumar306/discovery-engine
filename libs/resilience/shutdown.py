import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

log = logging.getLogger("resilience.shutdown")


async def run_until_stopped(
    run: Coroutine[Any, Any, None], stop: asyncio.Event, grace_s: float
) -> bool:
    """Run a loop until it ends. After stop is set, give it grace_s to finish the message in hand.

    Returns True when the loop ended by itself and False when it had to be cancelled.
    An error raised by the loop is raised to the caller.
    """
    loop_task = asyncio.create_task(run)
    stop_task = asyncio.create_task(stop.wait())
    try:
        await asyncio.wait({loop_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        if not loop_task.done():
            # stop was set: the loop checks it at the top of each turn, so it ends by itself as
            # soon as the message in hand is finished, and the wait ends the moment it does
            await asyncio.wait({loop_task}, timeout=grace_s)
    finally:
        # the watcher of stop is always cleaned up, and a loop still running now has used its
        # grace period, or the caller itself was cancelled, either way it must not be left behind
        stop_task.cancel()
        loop_task.cancel()  # does nothing to a task that is already done
    # both are awaited so no task is left half cancelled behind us
    outcome, _ = await asyncio.gather(loop_task, stop_task, return_exceptions=True)
    if isinstance(outcome, asyncio.CancelledError):
        log.warning(
            "the loop did not stop in time and was cancelled",
            extra={"fields": {"grace_s": grace_s}},
        )
        return False
    if isinstance(outcome, BaseException):
        raise outcome
    return True
