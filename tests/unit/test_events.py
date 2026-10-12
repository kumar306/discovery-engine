"""The S3 key and the two things every accepted write produces: the bytes for S3 and the event."""

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from libs.schemas import DeleteEvent, UpsertEvent, UpsertRequest, doc_event_adapter
from libs.schemas.hashing import canonical_bytes, content_sha
from services.ingest.events import EVENT_SCHEMA, prepare_delete, prepare_upsert, s3_key

NOW = datetime(2026, 10, 12, 9, 30, tzinfo=UTC)
LATER = datetime(2026, 10, 12, 9, 31, tzinfo=UTC)

# the canonical form of make_request() is pinned in test_hashing, this is the same document
PINNED_SHA = "ffaf9c81703a8e1eae3c0656e10c3d2d504ef6cdd58563936b9b1d8985d6ce26"


def make_request(**overrides: Any) -> UpsertRequest:
    body: dict[str, Any] = {
        "corpus": "esci",
        "doc_id": "B07ABC1234",
        "version": 3,
        "title": "Wireless Mouse",
        "brand": "Logitech",
        "color": "Black",
        "bullets": ["Quiet clicks", "18 month battery"],
        "description": "A compact wireless mouse.",
        "locale": "us",
    }
    return UpsertRequest.model_validate({**body, **overrides})


# the key


@pytest.mark.parametrize(
    ("corpus", "doc_id", "version", "expected"),
    [
        pytest.param("esci", "B07ABC1234", 3, "esci/B07ABC1234/v3.json", id="a normal document"),
        pytest.param("esci", "D1", 0, "esci/D1/v0.json", id="version zero"),
        pytest.param("esci", "a_b-c", 12, "esci/a_b-c/v12.json", id="underscore and dash"),
    ],
)
def test_the_key_is_corpus_then_document_then_version(
    corpus: str, doc_id: str, version: int, expected: str
) -> None:
    assert s3_key(corpus, doc_id, version) == expected


@given(
    st.from_regex(r"[A-Za-z0-9_-]{1,64}", fullmatch=True),
    st.integers(min_value=0, max_value=2**63 - 1),
)
def test_a_valid_document_id_can_never_climb_out_of_its_folder(doc_id: str, version: int) -> None:
    parts = s3_key("esci", doc_id, version).split("/")
    assert parts == ["esci", doc_id, f"v{version}.json"]


def test_every_version_of_a_document_gets_its_own_key() -> None:
    keys = {s3_key("esci", "D1", version) for version in range(50)}
    assert len(keys) == 50


# the upsert


def test_the_bytes_are_the_canonical_document_and_the_sha_is_theirs() -> None:
    prepared = prepare_upsert(make_request(), NOW)
    assert prepared.data == canonical_bytes(make_request())
    assert prepared.content_sha == content_sha(prepared.data)
    assert prepared.content_sha == PINNED_SHA


def test_the_event_describes_exactly_what_goes_to_s3() -> None:
    prepared = prepare_upsert(make_request(), NOW)
    event = doc_event_adapter.validate_json(prepared.event_json)
    assert isinstance(event, UpsertEvent)
    assert event.s3_key == prepared.key == "esci/B07ABC1234/v3.json"
    assert event.content_sha == prepared.content_sha
    assert event.size_bytes == len(prepared.data)
    assert (event.corpus, event.doc_id, event.version) == ("esci", "B07ABC1234", 3)
    assert (event.op, event.schema_version, event.produced_at) == ("upsert", 1, NOW)


def test_the_event_json_has_exactly_the_contract_keys() -> None:
    document = json.loads(prepare_upsert(make_request(), NOW).event_json)
    assert set(document) == {
        "schema_version",
        "corpus",
        "doc_id",
        "version",
        "produced_at",
        "op",
        "s3_key",
        "content_sha",
        "size_bytes",
    }


def test_the_same_request_and_time_always_give_the_same_event() -> None:
    assert prepare_upsert(make_request(), NOW) == prepare_upsert(make_request(), NOW)


def test_only_the_timestamp_changes_when_the_clock_moves() -> None:
    first = json.loads(prepare_upsert(make_request(), NOW).event_json)
    second = json.loads(prepare_upsert(make_request(), LATER).event_json)
    assert first.pop("produced_at") != second.pop("produced_at")
    assert first == second


def test_a_new_version_changes_the_key_the_sha_and_the_event() -> None:
    old = prepare_upsert(make_request(version=3), NOW)
    new = prepare_upsert(make_request(version=4), NOW)
    assert old.key != new.key
    assert old.content_sha != new.content_sha  # the version is part of the stored document
    assert old.event_json != new.event_json


def test_an_event_without_a_timezone_cannot_be_built() -> None:
    with pytest.raises(ValidationError):
        prepare_upsert(make_request(), datetime(2026, 10, 12, 9, 30))


# the delete


def test_a_delete_event_has_no_object_fields() -> None:
    prepared = prepare_delete("esci", "B07ABC1234", 4, NOW)
    event = doc_event_adapter.validate_json(prepared.event_json)
    assert isinstance(event, DeleteEvent)
    assert (event.corpus, event.doc_id, event.version, event.op) == (
        "esci",
        "B07ABC1234",
        4,
        "delete",
    )
    assert set(json.loads(prepared.event_json)) == {
        "schema_version",
        "corpus",
        "doc_id",
        "version",
        "produced_at",
        "op",
    }


def test_a_delete_of_an_invalid_document_id_cannot_be_built() -> None:
    with pytest.raises(ValidationError):
        prepare_delete("esci", "has/slash", 4, NOW)


def test_the_schema_name_is_the_one_the_relay_will_send_as_a_header() -> None:
    assert EVENT_SCHEMA == "doc_event_v1"
