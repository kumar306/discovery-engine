import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, nullcontext
from typing import Any, Protocol

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry
from psycopg import OperationalError
from psycopg.errors import LockNotAvailable
from psycopg_pool import PoolTimeout
from pydantic import TypeAdapter, ValidationError
from starlette.routing import Match

from libs.observability.health import Probe, add_health_routes
from libs.observability.logging import request_id_var
from libs.observability.metrics import IngestMetrics
from libs.resilience.bulkhead import Bulkhead, Overloaded
from libs.schemas import UpsertRequest
from libs.schemas.documents import MAX_VERSION, DocId
from services.ingest.api_models import AcceptedBody, ErrorBody, ValidationFailedBody
from services.ingest.service import WriteAccepted, WriteRefused
from services.ingest.settings import IngestSettings
from services.ingest.storage import StorageUnavailable

log = logging.getLogger("ingest.request")

# only plain safe characters are accepted from the client
# a header with newlines or control characters could forge extra log lines, that is log injection
_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")

_DOC_ID = TypeAdapter(DocId)
# whole numbers only: no sign, no spaces, no decimals, no digits from other scripts
_VERSION = re.compile(r"[0-9]{1,19}")

type Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]

# these answer on the traffic port, take no admission slot and are not counted as traffic
_PLATFORM_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})

_RETRY_AFTER = {
    "Retry-After": {
        "description": "Seconds to wait before trying again.",
        "schema": {"type": "integer"},
    }
}

# the upsert route reads the raw body itself, to cap its size before parsing, so fastapi sees no
# body model, this adds the request body to the openapi document, by reference to the model
_UPSERT_BODY = {
    "requestBody": {
        "required": True,
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/UpsertRequest"}}},
    }
}

_UPSERT_RESPONSES: dict[int | str, dict[str, Any]] = {
    202: {"model": AcceptedBody, "description": "Accepted: durable, not yet searchable."},
    409: {"model": ErrorBody, "description": "A newer version is held, or this one differs."},
    413: {"model": ErrorBody, "description": "The body is bigger than the limit."},
    422: {"model": ValidationFailedBody, "description": "A field breaks the rules."},
    503: {
        "model": ErrorBody,
        "description": "Overloaded or a dependency is down.",
        "headers": _RETRY_AFTER,
    },
}

# the delete route reads its two inputs itself, like the upsert route reads its body, so that a
# missing or wrong one is answered in our own error format and not in fastapi's, which means
# fastapi cannot describe them, they are described here, and the contract test holds the
# description to the code
_DELETE_PARAMETERS = {
    "parameters": [
        {
            "name": "doc_id",
            "in": "path",
            "required": True,
            "schema": _DOC_ID.json_schema(),
        },
        {
            "name": "version",
            "in": "query",
            "required": True,
            "schema": {"type": "integer", "minimum": 0, "maximum": MAX_VERSION},
        },
    ]
}

_DELETE_RESPONSES: dict[int | str, dict[str, Any]] = {
    202: {"model": AcceptedBody, "description": "Accepted: the tombstone is durable."},
    409: {"model": ErrorBody, "description": "A newer version is held."},
    422: {"model": ValidationFailedBody, "description": "The id or the version is wrong."},
    503: {
        "model": ErrorBody,
        "description": "Overloaded or a dependency is down.",
        "headers": _RETRY_AFTER,
    },
}


def _route_label(request: Request) -> str:
    # the route template, not the path, a path with a document id in it would make a new series
    # for every document, and a path that matched no route is one label for all of them
    route = request.scope.get("route")
    if route is not None:
        return str(route.path)
    # a request that was shed by the admission limit never reached the router, so the scope has
    # no route yet, it is matched here, otherwise the shed requests of every route would be
    # counted as other and a dashboard could not show which route is being shed
    for candidate in request.app.routes:
        if candidate.matches(request.scope)[0] is Match.FULL:
            return str(candidate.path)
    return "other"


def _document_the_request_body(app: FastAPI) -> None:
    """Put the UpsertRequest model into the openapi document that the upsert route points at."""
    original = app.openapi

    def openapi() -> dict[str, Any]:
        schema = original()
        schemas = schema.setdefault("components", {}).setdefault("schemas", {})
        if "UpsertRequest" not in schemas:
            model = UpsertRequest.model_json_schema(ref_template="#/components/schemas/{model}")
            schemas.update(model.pop("$defs", {}))
            schemas["UpsertRequest"] = model
        return schema

    app.openapi = openapi  # type: ignore[method-assign]


class IngestPort(Protocol):
    async def upsert(
        self, request: UpsertRequest, request_id: str
    ) -> WriteAccepted | WriteRefused: ...

    async def delete(
        self, doc_id: str, version: int, request_id: str
    ) -> WriteAccepted | WriteRefused: ...


class ApiError(Exception):
    def __init__(self, status: int, error: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status = status
        self.error = error
        self.message = message
        self.extra = extra


def _too_large(limit: int) -> ApiError:
    return ApiError(413, "payload_too_large", f"the body may not be bigger than {limit} bytes")


async def _read_body(request: Request, limit: int) -> bytes:
    # 413 means the body is bigger than we allow, and we say so before reading all of it
    # otherwise one client could make us buffer unbounded memory, the limit protects the event loop
    # content length is checked first, then the stream is counted too because the header can lie
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise _too_large(limit)
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise _too_large(limit)
        chunks.append(chunk)
    return b"".join(chunks)


def _field_errors(error: ValidationError, prefix: str = "") -> list[dict[str, str]]:
    details = error.errors(include_url=False, include_context=False, include_input=False)
    return [
        {
            "field": ".".join([prefix, *(str(part) for part in item["loc"])]).strip("."),
            "message": item["msg"],
            "type": item["type"],
        }
        for item in details
    ]


def _validation_failed(errors: list[dict[str, str]]) -> ApiError:
    # 422 means the request was understood but its content breaks our field rules
    # the client has to fix the fields, sending the same body again can never work
    # every field error comes back in one answer and the input values are not echoed
    return ApiError(422, "validation_failed", "some fields break the rules", errors=errors)


def _parse_upsert(body: bytes) -> UpsertRequest:
    try:
        return UpsertRequest.model_validate_json(body)
    except ValidationError as error:
        errors = _field_errors(error)
        if any(item["type"] == "json_invalid" for item in errors):
            raise ApiError(
                422, "invalid_json", "the body is not valid json", errors=errors
            ) from error
        raise _validation_failed(errors) from error


def _parse_delete(doc_id: str, version: str | None) -> tuple[str, int]:
    errors: list[dict[str, str]] = []
    try:
        _DOC_ID.validate_python(doc_id)
    except ValidationError as error:
        errors.extend(_field_errors(error, "doc_id"))
    number = -1
    if version is None:
        errors.append({"field": "version", "message": "the version is required", "type": "missing"})
    elif _VERSION.fullmatch(version) and int(version) <= MAX_VERSION:
        number = int(version)
    else:
        message = f"the version must be a whole number from 0 to {MAX_VERSION}"
        errors.append({"field": "version", "message": message, "type": "invalid_version"})
    if errors:
        raise _validation_failed(errors)
    return doc_id, number


def create_app(
    service: IngestPort,
    settings: IngestSettings,
    lifespan: Lifespan | None = None,
    admission: Bulkhead | None = None,
    probes: Mapping[str, Probe] | None = None,
    registry: CollectorRegistry | None = None,
) -> FastAPI:
    app = FastAPI(title="ingest api", lifespan=lifespan)
    registry = registry or CollectorRegistry()
    metrics = IngestMetrics(registry)
    # the health routes are on the traffic port on purpose: a probe on a separate port could stay
    # green while the port that serves the clients is stuck
    # being full is not unhealthy, shedding load is the api doing its job, so readiness looks at
    # the dependencies only, and taking a full pod out of rotation would push its load onto the rest
    add_health_routes(
        app,
        probes=probes or {},
        registry=registry,
        alive=lambda: True,
        readiness_cache_s=settings.readiness_cache_s,
        probe_timeout_s=settings.probe_timeout_s,
    )
    _document_the_request_body(app)
    # the outermost limit, requests being served at once, separate from the s3 and database limits
    admission = admission or Bulkhead(settings.max_inflight_requests, settings.admission_wait_s)

    def unavailable(request: Request, outcome: str, error: str, message: str) -> JSONResponse:
        request.state.outcome = outcome
        body = {"error": error, "message": message, "request_id": request.state.request_id}
        # a 503 always says when to come back, so clients back off instead of hammering
        return JSONResponse(
            status_code=503, content=body, headers={"Retry-After": str(settings.retry_after_s)}
        )

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
            try:
                # only the api paths count, probes and docs must still answer when the api is full
                # nullcontext does nothing but it also works with async with
                # so both branches are used the same way below and other paths take no slot
                slot = admission if request.url.path.startswith("/v1/") else nullcontext()
                async with slot:
                    response = await call_next(request)
            except Overloaded:
                response = unavailable(
                    request, "admission_full", "overloaded", "too many requests, try again shortly"
                )
            except Exception:
                # anything we did not plan for: log it with the request id, tell the client nothing
                request.state.outcome = "internal_error"
                log.exception("unhandled error")
                response = JSONResponse(
                    status_code=500,
                    content={
                        "error": "internal_error",
                        "message": "something went wrong on our side",
                        "request_id": request_id,
                    },
                )
            response.headers["X-Request-ID"] = request_id
            elapsed = time.perf_counter() - started
            if request.url.path not in _PLATFORM_PATHS:
                route = _route_label(request)
                status_class = f"{response.status_code // 100}xx"
                metrics.requests.labels(route=route, status_class=status_class).inc()
                metrics.request_seconds.labels(route=route).observe(elapsed)
            state = request.state
            log.info(
                "request",
                extra={
                    "fields": {
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "outcome": getattr(state, "outcome", str(response.status_code)),
                        "duration_ms": round(elapsed * 1000, 2),
                        "doc_id": getattr(state, "doc_id", None),
                        "version": getattr(state, "version", None),
                    }
                },
            )
            return response
        finally:
            request_id_var.reset(token)

    @app.exception_handler(ApiError)
    async def api_error(request: Request, error: ApiError) -> JSONResponse:
        request.state.outcome = error.error
        body = {
            "error": error.error,
            "message": error.message,
            "request_id": request.state.request_id,
        }
        body.update(error.extra)
        return JSONResponse(status_code=error.status, content=body)

    @app.exception_handler(StorageUnavailable)
    async def storage_unavailable(request: Request, error: StorageUnavailable) -> JSONResponse:
        return unavailable(
            request, "storage_unavailable", "storage_unavailable", "storage is not answering"
        )

    @app.exception_handler(Overloaded)
    async def storage_overloaded(request: Request, error: Overloaded) -> JSONResponse:
        return unavailable(
            request, "storage_overloaded", "overloaded", "storage is busy, try again shortly"
        )

    @app.exception_handler(OperationalError)
    async def database_busy(request: Request, error: OperationalError) -> JSONResponse:
        # pool timeouts, lock timeouts, a restarting database and refused connections all end here
        outcome = (
            "pool_exhausted"
            if isinstance(error, PoolTimeout)
            else "lock_timeout"
            if isinstance(error, LockNotAvailable)
            else "database_unavailable"
        )
        return unavailable(
            request, outcome, "database_busy", "the database is busy, try again shortly"
        )

    def answer(request: Request, result: WriteAccepted | WriteRefused) -> JSONResponse:
        if isinstance(result, WriteRefused):
            raise ApiError(
                409,
                result.error,
                "a newer version is already held"
                if result.error == "stale_version"
                else "this version already exists with different content",
                current_version=result.current_version,
            )
        request.state.outcome = "accepted"
        return JSONResponse(
            status_code=202,
            content={
                "request_id": request.state.request_id,
                "doc_id": result.doc_id,
                "version": result.version,
                "accepted_at": result.accepted_at.isoformat(),
            },
        )

    @app.post(
        "/v1/documents",
        status_code=202,
        responses=_UPSERT_RESPONSES,
        openapi_extra=_UPSERT_BODY,
    )
    async def upsert(request: Request) -> JSONResponse:
        document = _parse_upsert(await _read_body(request, settings.max_body_bytes))
        request.state.doc_id, request.state.version = document.doc_id, document.version
        return answer(request, await service.upsert(document, request.state.request_id))

    @app.delete(
        "/v1/documents/{doc_id}",
        status_code=202,
        responses=_DELETE_RESPONSES,
        openapi_extra=_DELETE_PARAMETERS,
    )
    async def delete(request: Request) -> JSONResponse:
        doc_id, number = _parse_delete(
            request.path_params["doc_id"], request.query_params.get("version")
        )
        request.state.doc_id, request.state.version = doc_id, number
        return answer(request, await service.delete(doc_id, number, request.state.request_id))

    return app
