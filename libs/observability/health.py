import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest

Clock = Callable[[], float]
# a probe returns when the dependency answers and raises when it does not
Probe = Callable[[], Awaitable[None]]

# how long the health port may take to open before the process gives up on it
START_TIMEOUT_S = 5.0
# how long it may take to stop, a scraper that keeps a connection open must not delay a shutdown
STOP_TIMEOUT_S = 5.0


class Heartbeat:
    """The loop of a process calls beat() every turn, so its age says whether the loop turns."""

    def __init__(self, clock: Clock = time.monotonic) -> None:
        self._clock = clock
        # a process that has not turned yet is alive, it only just started
        self._last = clock()

    def beat(self) -> None:
        self._last = self._clock()

    def age(self) -> float:
        return self._clock() - self._last


@dataclass(frozen=True)
class Readiness:
    ready: bool
    # probe name -> "ok", "timeout" or "failed: <error type>", never the text of an error
    checks: dict[str, str]


class ReadinessChecker:
    def __init__(
        self, probes: Mapping[str, Probe], timeout_s: float, cache_s: float, clock: Clock
    ) -> None:
        self._probes = dict(probes)
        self._timeout_s = timeout_s
        self._cache_s = cache_s
        self._clock = clock
        self._cached: Readiness | None = None
        self._cached_at = 0.0
        # one check at a time: requests that arrive during a check wait for it and then
        # read its answer from the cache, so a burst of requests never becomes a burst of probes
        self._lock = asyncio.Lock()

    async def check(self) -> Readiness:
        async with self._lock:
            fresh = self._cached is not None and self._clock() - self._cached_at < self._cache_s
            if fresh and self._cached is not None:
                return self._cached
            names = list(self._probes)
            # all probes at once, so the slowest one sets the time and not the sum
            outcomes = await asyncio.gather(*(self._run(self._probes[name]) for name in names))
            report = Readiness(
                all(o == "ok" for o in outcomes), dict(zip(names, outcomes, strict=True))
            )
            # a failure is cached as well, a dependency that is down is not asked every second
            self._cached, self._cached_at = report, self._clock()
            return report

    async def _run(self, probe: Probe) -> str:
        try:
            async with asyncio.timeout(self._timeout_s):
                await probe()
        except TimeoutError:
            return "timeout"
        except Exception as error:
            return f"failed: {type(error).__name__}"
        return "ok"


def add_health_routes(
    app: FastAPI,
    *,
    probes: Mapping[str, Probe],
    registry: CollectorRegistry,
    alive: Callable[[], bool],
    readiness_cache_s: float,
    probe_timeout_s: float,
    clock: Clock = time.monotonic,
) -> None:
    """Add /healthz, /readyz and /metrics to an app, so a process that serves traffic on its
    own port and a process that only needs a health port define them once.

    alive says whether the process is alive: a loop process answers with the age of its
    heartbeat, an http service answers True, because if its event loop were stuck this route
    would not answer at all and that is what makes the platform restart it.
    """
    checker = ReadinessChecker(probes, probe_timeout_s, readiness_cache_s, clock)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> JSONResponse:
        # shallow on purpose: only "is the process turning", a dependency being down must
        # not fail this, restarting the process would not fix it
        is_alive = alive()
        return JSONResponse({"alive": is_alive}, status_code=200 if is_alive else 503)

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> JSONResponse:
        report = await checker.check()
        body = {"ready": report.ready, "checks": report.checks}
        return JSONResponse(body, status_code=200 if report.ready else 503)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


def create_health_app(
    *,
    probes: Mapping[str, Probe],
    heartbeat: Heartbeat,
    registry: CollectorRegistry,
    liveness_max_age_s: float,
    readiness_cache_s: float,
    probe_timeout_s: float,
    clock: Clock = time.monotonic,
) -> FastAPI:
    # no docs pages: this port is for the platform and prometheus, not for people
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    add_health_routes(
        app,
        probes=probes,
        registry=registry,
        alive=lambda: heartbeat.age() <= liveness_max_age_s,
        readiness_cache_s=readiness_cache_s,
        probe_timeout_s=probe_timeout_s,
        clock=clock,
    )
    return app


class _HealthServer(uvicorn.Server):
    # uvicorn would take over SIGINT and SIGTERM while it serves and raise them again when it
    # stops, but those signals belong to the entry point of the process, which stops the
    # consumer in an orderly way, so here the server leaves them alone
    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


@asynccontextmanager
async def run_health_server(app: FastAPI, host: str, port: int) -> AsyncIterator[int]:
    """Serve the app on its own port for the length of the block, and yield the port it got."""
    config = uvicorn.Config(app, host=host, port=port, log_config=None, timeout_graceful_shutdown=2)
    server = _HealthServer(config)

    async def serve() -> None:
        try:
            await server.serve()
        except SystemExit as exit_:
            # uvicorn exits the process when it cannot bind, here that is an ordinary error
            raise OSError(f"the health server could not start on {host}:{port}") from exit_

    task = asyncio.create_task(serve())
    try:
        async with asyncio.timeout(START_TIMEOUT_S):
            while not server.started:
                if task.done():
                    task.result()  # raises the reason it could not start
                    raise OSError("the health server stopped while starting")
                await asyncio.sleep(0.01)
        bound: int = server.servers[0].sockets[0].getsockname()[1]
        yield bound
    finally:
        server.should_exit = True
        # a server that never started has already failed, its error was raised above
        # one that does not stop in time is cancelled, so the shutdown of the process never hangs
        done, _ = await asyncio.wait({task}, timeout=STOP_TIMEOUT_S)
        if not done:
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
