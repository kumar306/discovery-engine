"""The contract the api publishes: /openapi.json must say what the api really takes and returns."""

from typing import Any

import httpx
import pytest

from services.ingest.app import create_app
from services.ingest.settings import IngestSettings
from tests.unit.test_api_edges import RecordingIngest


@pytest.fixture
def document() -> dict[str, Any]:
    app = create_app(RecordingIngest(), IngestSettings())
    schema: dict[str, Any] = app.openapi()
    return schema


def refs_in(node: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref":
                found.add(value)
            else:
                found |= refs_in(value)
    elif isinstance(node, list):
        for item in node:
            found |= refs_in(item)
    return found


def test_the_upsert_route_declares_a_json_request_body(document: dict[str, Any]) -> None:
    body = document["paths"]["/v1/documents"]["post"]["requestBody"]
    assert body["required"] is True
    assert body["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/UpsertRequest"
    }


def test_the_request_schema_is_the_real_model_with_its_rules(document: dict[str, Any]) -> None:
    schema = document["components"]["schemas"]["UpsertRequest"]
    assert {"corpus", "doc_id", "version", "title", "locale"} <= set(schema["required"])
    assert set(schema["properties"]) >= {"brand", "color", "bullets", "description"}
    assert schema["additionalProperties"] is False  # an unknown field is refused, as documented
    assert schema["properties"]["bullets"]["maxItems"] == 20


def test_every_reference_in_the_document_points_at_something_that_exists(
    document: dict[str, Any],
) -> None:
    schemas = document["components"]["schemas"]
    for ref in refs_in(document):
        assert ref.startswith("#/components/schemas/"), ref
        assert ref.removeprefix("#/components/schemas/") in schemas, ref


def test_the_upsert_route_documents_every_answer_it_can_give(document: dict[str, Any]) -> None:
    responses = document["paths"]["/v1/documents"]["post"]["responses"]
    assert {"202", "409", "413", "422", "503"} <= set(responses)


def test_the_delete_route_documents_every_answer_it_can_give(document: dict[str, Any]) -> None:
    responses = document["paths"]["/v1/documents/{doc_id}"]["delete"]["responses"]
    assert {"202", "409", "422", "503"} <= set(responses)


def test_a_503_tells_clients_when_to_come_back(document: dict[str, Any]) -> None:
    response = document["paths"]["/v1/documents"]["post"]["responses"]["503"]
    assert "Retry-After" in response["headers"]


def test_the_422_answer_lists_the_field_errors(document: dict[str, Any]) -> None:
    schemas = document["components"]["schemas"]
    failed = schemas["ValidationFailedBody"]
    assert {"error", "message", "request_id", "errors"} <= set(failed["properties"])
    item = schemas["FieldError"]
    assert set(item["required"]) == {"field", "message", "type"}


def test_a_409_carries_the_current_version(document: dict[str, Any]) -> None:
    assert "current_version" in document["components"]["schemas"]["ErrorBody"]["properties"]


async def test_the_document_is_served_over_http() -> None:
    app = create_app(RecordingIngest(), IngestSettings())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://a") as c:
        response = await c.get("/openapi.json")
    assert response.status_code == 200
    assert "/v1/documents" in response.json()["paths"]
