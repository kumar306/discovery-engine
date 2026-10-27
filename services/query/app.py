import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, nullcontext
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.routing import Match

from libs.observability.health import Probe, add_health_routes
from libs.observability.logging import request_id_var
from libs.observability.metrics import QueryMetrics
from libs.resilience.bulkhead import Bulkhead, Overloaded
from services.query.flow_chunks import ChunkFlow, register_chunk_route
from services.query.flow_search import SearchFlow, register_search_route
from services.query.primitives import Unavailable
from services.query.settings import QuerySettings

log = logging.getLogger("query.request")

# only plain safe characters are accepted from the client
# a header with newlines or control characters could forge extra log lines, that is log injection
_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")

type Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]

# these answer on the traffic port, take no admission slot and are not counted as traffic
_PLATFORM_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})


def _route_label(request: Request) -> str:
    # the route template, not the path, and one label for everything that matched no route
    route = request.scope.get("route")
    if route is not None:
        return str(route.path)
    # a request shed by the admission limit never reached the router, so it is matched here
    for candidate in request.app.routes:
        if candidate.matches(request.scope)[0] is Match.FULL:
            return str(candidate.path)
    return "other"


def create_app(
    search_flow: SearchFlow,
    chunk_flow: ChunkFlow,
    settings: QuerySettings,
    metrics: QueryMetrics,
    registry: CollectorRegistry,
    lifespan: Lifespan | None = None,
    admission: Bulkhead | None = None,
    probes: Mapping[str, Probe] | None = None,
) -> FastAPI:
    app = FastAPI(title="query service", lifespan=lifespan)
    # the health routes are on the traffic port on purpose: a probe on a separate port could stay
    # green while the port that serves the clients is stuck
    # being full is not unhealthy, so readiness looks at the dependencies only
    add_health_routes(
        app,
        probes=probes or {},
        registry=registry,
        alive=lambda: True,
        readiness_cache_s=settings.readiness_cache_s,
        probe_timeout_s=settings.probe_timeout_s,
    )
    register_search_route(app, search_flow)
    register_chunk_route(app, chunk_flow)
    # the outermost limit, requests being served at once
    admission = admission or Bulkhead(settings.max_inflight_requests, settings.admission_wait_s)
    # read at the moment of a scrape, nothing is counted on the request path
    metrics.admission_in_flight.set_function(lambda: admission.in_flight)
    metrics.admission_waiting.set_function(lambda: admission.waiting)

    def error_response(
        request: Request,
        status: int,
        error: str,
        message: str,
        headers: dict[str, str] | None = None,
        **extra: Any,
    ) -> JSONResponse:
        request.state.outcome = error
        body = {
            "error": error,
            "message": message,
            "request_id": request.state.request_id,
            **extra,
        }
        return JSONResponse(status_code=status, content=body, headers=headers)

    def retry_headers() -> dict[str, str]:
        # a 503 always says when to come back, so clients back off instead of hammering
        return {"Retry-After": str(settings.retry_after_s)}

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        candidate = request.headers.get("x-request-id", "")
        request_id = candidate if _REQUEST_ID.fullmatch(candidate) else uuid.uuid4().hex
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            declared = request.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > settings.max_body_bytes:
                # said before the body is read, so one client cannot make us buffer unbounded memory
                response: Response = error_response(
                    request,
                    413,
                    "payload_too_large",
                    f"the body may not be bigger than {settings.max_body_bytes} bytes",
                )
            else:
                try:
                    # only the api paths count, probes and docs must still answer when it is full
                    slot = admission if request.url.path.startswith("/v1/") else nullcontext()
                    async with slot:
                        response = await call_next(request)
                except Overloaded:
                    metrics.admission_shed.inc()
                    response = error_response(
                        request,
                        503,
                        "overloaded",
                        "too many requests, try again shortly",
                        retry_headers(),
                    )
                except Exception:
                    # anything unplanned: log it with the request id, tell the client nothing
                    log.exception("unhandled error")
                    response = error_response(
                        request, 500, "internal_error", "something went wrong on our side"
                    )
            response.headers["X-Request-ID"] = request_id
            elapsed = time.perf_counter() - started
            if request.url.path not in _PLATFORM_PATHS:
                route = _route_label(request)
                metrics.requests.labels(
                    route=route, status_class=f"{response.status_code // 100}xx"
                ).inc()
                metrics.request_seconds.labels(route=route).observe(elapsed)
            log.info(
                "request",
                extra={
                    "fields": {
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "outcome": getattr(request.state, "outcome", str(response.status_code)),
                        "duration_ms": round(elapsed * 1000, 2),
                    }
                },
            )
            return response
        finally:
            request_id_var.reset(token)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        # 422: the request was understood but its content breaks our rules, sending it again
        # unchanged can never work, the input values are not echoed back
        fields = [
            {
                "field": "body"
                if item["type"] == "json_invalid"
                else ".".join(str(part) for part in item["loc"][1:]) or "body",
                "message": item["msg"],
                "type": item["type"],
            }
            for item in error.errors()
        ]
        code = (
            "invalid_json"
            if any(f["type"] == "json_invalid" for f in fields)
            else "validation_failed"
        )
        return error_response(request, 422, code, "some fields break the rules", errors=fields)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException) -> Response:
        # fastapi answers 400 when the body cannot even be decoded, for example bytes that are
        # not text, that is the same mistake as invalid json and gets the same answer in our format
        if error.status_code == 400:
            return error_response(
                request,
                422,
                "invalid_json",
                "the body is not valid json",
                errors=[
                    {
                        "field": "body",
                        "message": "the body cannot be read",
                        "type": "body_unreadable",
                    }
                ],
            )
        # a missing route or a wrong method keeps fastapi's own answer
        return await http_exception_handler(request, error)

    @app.exception_handler(Unavailable)
    async def unavailable(request: Request, error: Unavailable) -> JSONResponse:
        return error_response(
            request, 503, "unavailable", "the search backend is not answering", retry_headers()
        )

    return app
