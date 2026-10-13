import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from typing import Any, Protocol

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from psycopg import OperationalError
from psycopg.errors import LockNotAvailable
from psycopg_pool import PoolTimeout
from pydantic import TypeAdapter, ValidationError

from libs.observability.logging import request_id_var
from libs.resilience.bulkhead import Bulkhead, Overloaded
from libs.schemas import UpsertRequest
from libs.schemas.documents import MAX_VERSION, DocId
from services.ingest.service import Accepted, Refused
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


class IngestPort(Protocol):
    async def upsert(self, request: UpsertRequest, request_id: str) -> Accepted | Refused: ...

    async def delete(self, doc_id: str, version: int, request_id: str) -> Accepted | Refused: ...


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
) -> FastAPI:
    app = FastAPI(title="ingest api", lifespan=lifespan)
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
            state = request.state
            log.info(
                "request",
                extra={
                    "fields": {
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "outcome": getattr(state, "outcome", str(response.status_code)),
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
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

    def answer(request: Request, result: Accepted | Refused) -> JSONResponse:
        if isinstance(result, Refused):
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

    @app.post("/v1/documents", status_code=202)
    async def upsert(request: Request) -> JSONResponse:
        document = _parse_upsert(await _read_body(request, settings.max_body_bytes))
        request.state.doc_id, request.state.version = document.doc_id, document.version
        return answer(request, await service.upsert(document, request.state.request_id))

    @app.delete("/v1/documents/{doc_id}", status_code=202)
    async def delete(request: Request, doc_id: str, version: str | None = None) -> JSONResponse:
        doc_id, number = _parse_delete(doc_id, version)
        request.state.doc_id, request.state.version = doc_id, number
        return answer(request, await service.delete(doc_id, number, request.state.request_id))

    return app
