import asyncio
from types import TracebackType


class Overloaded(Exception):
    """No permit became free in time: shed the call (503 with Retry-After)."""


# a bulkhead counts how many calls may be inside one guarded section at once, it is a semaphore
# with a wait timeout, a permit is one count of it, and a permit owns no thread, no connection
# and almost no memory, it is only the right to be inside
# what a permit stands for is whatever the section uses, for the s3 store a thread of its pool
# and a socket, one permit covers the whole call including retries and waits between them, so
# permits in use can be higher than threads busy, that is why the limit stays under the pool
# every Bulkhead object has its own count, two bulkheads are two separate limits that add up
# and never share, and every process has its own, so the ceiling of a service is the limit
# times the number of processes
# layers do not multiply, a request holds one permit of each layer it is in right now, so with
# 32 admitted requests, 8 s3 permits and 8 database connections, at most 8 of the 32 are in
# s3 at one moment and at most 8 in the database, not 32 times 8
class Bulkhead:
    def __init__(self, limit: int, wait_timeout: float) -> None:
        self._permits = asyncio.Semaphore(limit)
        self._wait_timeout = wait_timeout
        self.rejected = 0
        # read by the metrics: calls inside the guarded section right now, and calls waiting for a
        # permit right now (one event loop thread changes them, so plain integers are enough)
        self.in_flight = 0
        self.waiting = 0

    async def __aenter__(self) -> None:
        self.waiting += 1
        try:
            async with asyncio.timeout(self._wait_timeout):
                await self._permits.acquire()
        except TimeoutError:
            self.rejected += 1
            raise Overloaded("too many calls in flight") from None
        finally:
            # a caller leaves the waiting line whether it got a permit, timed out or was cancelled
            self.waiting -= 1
        self.in_flight += 1

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # runs on success, on an exception and on cancellation, so a permit is never lost
        self.in_flight -= 1
        self._permits.release()
