import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from libs.observability.logging import request_id_var
from libs.schemas import UpsertRequest
from services.ingest.service import Accepted, Refused
from services.ingest.settings import IngestSettings

log = logging.getLogger("ingest.request")

# only plain safe characters are accepted from the client
# a header with newlines or control characters could forge extra log lines, that is log injection
_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


class IngestPort(Protocol):
    async def upsert(self, request: UpsertRequest, request_id: str) -> Accepted | Refused: ...


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


def _parse_upsert(body: bytes) -> UpsertRequest:
    try:
        return UpsertRequest.model_validate_json(body)
    except ValidationError as error:
        # 422 means the request was understood but its content breaks our field rules
        # the client has to fix the fields, sending the same body again can never work
        # every field error comes back in one answer and the input values are not echoed
        details = error.errors(include_url=False, include_context=False, include_input=False)
        fields = [
            {
                "field": ".".join(str(part) for part in item["loc"]),
                "message": item["msg"],
                "type": item["type"],
            }
            for item in details
        ]
        malformed = any(item["type"] == "json_invalid" for item in details)
        raise ApiError(
            422,
            "invalid_json" if malformed else "validation_failed",
            "the body is not valid json" if malformed else "some fields break the rules",
            errors=fields,
        ) from error


def create_app(service: IngestPort, settings: IngestSettings) -> FastAPI:
    app = FastAPI(title="ingest api")

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
            response = await call_next(request)
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

    @app.post("/v1/documents", status_code=202)
    async def upsert(request: Request) -> JSONResponse:
        document = _parse_upsert(await _read_body(request, settings.max_body_bytes))
        request.state.doc_id, request.state.version = document.doc_id, document.version
        result = await service.upsert(document, request.state.request_id)
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

    return app
