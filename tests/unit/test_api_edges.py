"""The HTTP edges of the ingest api that need no database and no S3: ids, size cap, field rules.

The service behind the app is a recorder, so every test can also say whether it was reached.
"""

import io
import json
import logging
import re
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from libs.observability.logging import JsonFormatter
from libs.schemas import UpsertRequest
from services.ingest.app import create_app
from services.ingest.service import Accepted, Refused
from services.ingest.settings import IngestSettings

NOW = datetime(2026, 10, 12, 9, 30, tzinfo=UTC)
LIMIT = 2000


class RecordingIngest:
    def __init__(self, result: Accepted | Refused | None = None) -> None:
        self.result = result or Accepted("D1", 1, NOW)
        self.calls: list[tuple[UpsertRequest, str]] = []

    async def upsert(self, request: UpsertRequest, request_id: str) -> Accepted | Refused:
        self.calls.append((request, request_id))
        return self.result


def product(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "corpus": "esci",
        "doc_id": "D1",
        "version": 1,
        "title": "Wireless Mouse",
        "locale": "us",
    }
    return {**body, **overrides}


class LogCapture:
    def __init__(self) -> None:
        self.stream = io.StringIO()

    def lines(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.stream.getvalue().splitlines()]


@pytest.fixture
def logs() -> Iterator[LogCapture]:
    capture = LogCapture()
    handler = logging.StreamHandler(capture.stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("ingest.request")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield capture
    logger.removeHandler(handler)


async def make_client(service: RecordingIngest) -> httpx.AsyncClient:
    app = create_app(service, IngestSettings(max_body_bytes=LIMIT))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api")


@pytest.fixture
async def service() -> RecordingIngest:
    return RecordingIngest()


@pytest.fixture
async def client(service: RecordingIngest) -> AsyncIterator[httpx.AsyncClient]:
    async with await make_client(service) as client:
        yield client


async def post(client: httpx.AsyncClient, body: Any, **headers: str) -> httpx.Response:
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return await client.post("/v1/documents", content=content, headers=headers)


# the request id


async def test_a_request_id_is_generated_when_the_client_sends_none(
    client: httpx.AsyncClient,
) -> None:
    response = await post(client, product())
    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-request-id"])


async def test_a_valid_request_id_from_the_client_is_kept_and_passed_on(
    client: httpx.AsyncClient, service: RecordingIngest
) -> None:
    response = await post(client, product(), **{"X-Request-ID": "client-abc.123_X"})
    assert response.headers["x-request-id"] == "client-abc.123_X"
    assert response.json()["request_id"] == "client-abc.123_X"
    assert service.calls[0][1] == "client-abc.123_X"


@pytest.mark.parametrize(
    "bad",
    ["has space", "a" * 65, "x;y", "<script>", "id\nforged log line", "id\r\nSet-Cookie: x=1"],
)
async def test_a_request_id_with_unsafe_characters_is_replaced(
    client: httpx.AsyncClient, bad: str
) -> None:
    response = await post(client, product(), **{"X-Request-ID": bad})
    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-request-id"])


async def test_every_kind_of_response_carries_the_request_id(client: httpx.AsyncClient) -> None:
    responses = [
        await post(client, product()),  # 202
        await post(client, b"{"),  # 422
        await post(client, b" " * (LIMIT + 1)),  # 413
        await client.get("/nothing-here"),  # 404
        await client.get("/v1/documents"),  # 405
    ]
    assert [r.status_code for r in responses] == [202, 422, 413, 404, 405]
    assert all(re.fullmatch(r"[0-9a-f]{32}", r.headers["x-request-id"]) for r in responses)


async def test_each_request_writes_one_log_line_with_its_request_id(
    client: httpx.AsyncClient, logs: LogCapture
) -> None:
    response = await post(client, product(doc_id="D7", version=4), **{"X-Request-ID": "trace-1"})
    (line,) = logs.lines()
    assert line["request_id"] == "trace-1" == response.headers["x-request-id"]
    assert (line["method"], line["path"], line["status"]) == ("POST", "/v1/documents", 202)
    assert (line["outcome"], line["doc_id"], line["version"]) == ("accepted", "D7", 4)
    assert line["duration_ms"] >= 0


async def test_a_rejected_request_is_logged_with_its_reason(
    client: httpx.AsyncClient, logs: LogCapture
) -> None:
    await post(client, b" " * (LIMIT + 1))
    (line,) = logs.lines()
    assert (line["status"], line["outcome"]) == (413, "payload_too_large")


# the size cap


async def test_a_declared_size_over_the_limit_is_refused_before_the_service_is_reached(
    client: httpx.AsyncClient, service: RecordingIngest
) -> None:
    response = await post(client, b" " * (LIMIT + 1))
    assert response.status_code == 413
    assert response.json()["error"] == "payload_too_large"
    assert service.calls == []


async def test_a_chunked_body_over_the_limit_is_refused_too(
    client: httpx.AsyncClient, service: RecordingIngest
) -> None:
    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(5):  # no content length is sent, only the stream shows the real size
            yield b" " * (LIMIT // 2)

    response = await client.post("/v1/documents", content=chunks())
    assert response.status_code == 413
    assert service.calls == []


async def test_a_body_of_exactly_the_limit_is_accepted_and_one_more_byte_is_not(
    client: httpx.AsyncClient, service: RecordingIngest
) -> None:
    document = json.dumps(product()).encode()
    exact = document + b" " * (LIMIT - len(document))  # trailing spaces are valid json
    assert len(exact) == LIMIT
    assert (await post(client, exact)).status_code == 202
    assert (await post(client, exact + b" ")).status_code == 413
    assert len(service.calls) == 1


# the field rules


@pytest.mark.parametrize(
    ("body", "code", "fields"),
    [
        pytest.param(b"", "invalid_json", None, id="empty body"),
        pytest.param(b"{", "invalid_json", None, id="broken json"),
        pytest.param(b"[]", "validation_failed", None, id="not an object"),
        pytest.param(
            product(esci_label="E"), "validation_failed", {"esci_label"}, id="unknown field"
        ),
        pytest.param(product(doc_id="has/slash"), "validation_failed", {"doc_id"}, id="bad doc id"),
        pytest.param(product(version="3"), "validation_failed", {"version"}, id="version as text"),
        pytest.param(product(version=True), "validation_failed", {"version"}, id="version as bool"),
        pytest.param(product(version=-1), "validation_failed", {"version"}, id="negative version"),
        pytest.param(product(locale="xx"), "validation_failed", {"locale"}, id="unknown locale"),
        pytest.param(
            product(doc_id="", version=-1, locale="xx"),
            "validation_failed",
            {"doc_id", "version", "locale"},
            id="three problems at once",
        ),
    ],
)
async def test_bad_bodies_get_a_422_naming_the_fields_and_never_reach_the_service(
    client: httpx.AsyncClient,
    service: RecordingIngest,
    body: Any,
    code: str,
    fields: set[str] | None,
) -> None:
    response = await post(client, body)
    assert response.status_code == 422
    assert response.json()["error"] == code
    if fields is not None:
        assert {item["field"] for item in response.json()["errors"]} == fields
    assert service.calls == []


async def test_a_missing_title_is_named(client: httpx.AsyncClient) -> None:
    body = product()
    del body["title"]
    response = await post(client, body)
    assert [item["field"] for item in response.json()["errors"]] == ["title"]


async def test_the_422_never_echoes_what_the_client_sent(client: httpx.AsyncClient) -> None:
    response = await post(client, product(doc_id="TOP-SECRET/../value", title="x" * 500))
    assert response.status_code == 422
    assert "TOP-SECRET" not in response.text
    assert all(set(item) == {"field", "message", "type"} for item in response.json()["errors"])


# what the service answers


async def test_a_valid_body_reaches_the_service_once_and_gets_the_202_body(
    client: httpx.AsyncClient, service: RecordingIngest
) -> None:
    response = await post(client, product(doc_id="D1", version=1))
    assert response.status_code == 202
    body = response.json()
    assert body == {
        "request_id": response.headers["x-request-id"],
        "doc_id": "D1",
        "version": 1,
        "accepted_at": NOW.isoformat(),
    }
    ((received, request_id),) = service.calls
    assert (received.doc_id, received.version, request_id) == ("D1", 1, body["request_id"])


@pytest.mark.parametrize(
    ("refusal", "error", "current"),
    [
        pytest.param(Refused("stale_version", 7), "stale_version", 7, id="stale"),
        pytest.param(Refused("content_conflict", 5), "content_conflict", 5, id="conflict"),
        pytest.param(
            Refused("content_conflict", None), "content_conflict", None, id="leftover object"
        ),
    ],
)
async def test_a_refusal_becomes_a_409_with_the_current_version(
    refusal: Refused, error: str, current: int | None
) -> None:
    async with await make_client(RecordingIngest(refusal)) as client:
        response = await post(client, product())
    assert response.status_code == 409
    body = response.json()
    assert (body["error"], body["current_version"]) == (error, current)
    assert body["request_id"] == response.headers["x-request-id"]
    # the human text has to match the code, a stale write is not a content conflict
    assert ("newer version" in body["message"]) == (error == "stale_version")
    assert ("different content" in body["message"]) == (error == "content_conflict")
