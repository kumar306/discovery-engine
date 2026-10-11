import asyncio
from types import TracebackType


class Overloaded(Exception):
    """No permit became free in time: shed the call (503 with Retry-After)."""


class Bulkhead:
    def __init__(self, limit: int, wait_timeout: float) -> None:
        self._permits = asyncio.Semaphore(limit)
        self._wait_timeout = wait_timeout
        self.rejected = 0

    async def __aenter__(self) -> None:
        try:
            async with asyncio.timeout(self._wait_timeout):
                await self._permits.acquire()
        except TimeoutError:
            self.rejected += 1
            raise Overloaded("too many calls in flight") from None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # runs on success, on an exception and on cancellation, so a permit is never lost
        self._permits.release()
