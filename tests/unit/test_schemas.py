"""Contract tests for the ingest schemas: chain A1 (the HTTP body) and the C1.2 DocEvent."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from libs.schemas import (
    DeleteEvent,
    UpsertEvent,
    UpsertRequest,
    doc_event_adapter,
)

MAX_VERSION = 2**63 - 1  # largest value a Postgres bigint can hold


def make_upsert(**overrides: Any) -> dict[str, Any]:
    """A valid request body as a client would send it; tests override one field at a time."""
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
    return {**body, **overrides}


def make_event(**overrides: Any) -> dict[str, Any]:
    """A valid upsert DocEvent as it travels through Kafka."""
    body: dict[str, Any] = {
        "schema_version": 1,
        "corpus": "esci",
        "doc_id": "B07ABC1234",
        "version": 3,
        "op": "upsert",
        "s3_key": "esci/B07ABC1234/v3.json",
        "content_sha": "a" * 64,
        "size_bytes": 512,
        "produced_at": datetime(2026, 10, 10, 12, 0, tzinfo=UTC),
    }
    return {**body, **overrides}


def make_delete_event(**overrides: Any) -> dict[str, Any]:
    """A valid delete DocEvent: no S3 object, because a delete has no new content to read."""
    body: dict[str, Any] = {
        "schema_version": 1,
        "corpus": "esci",
        "doc_id": "B07ABC1234",
        "version": 4,
        "op": "delete",
        "produced_at": datetime(2026, 10, 10, 12, 0, tzinfo=UTC),
    }
    return {**body, **overrides}


def bad_fields(exc_info: pytest.ExceptionInfo[ValidationError]) -> set[str]:
    """The names of the fields that failed, e.g. {'doc_id', 'version'}."""
    return {".".join(str(part) for part in err["loc"]) for err in exc_info.value.errors()}


def bad_event_fields(exc_info: pytest.ExceptionInfo[ValidationError]) -> set[str]:
    """Like bad_fields, minus the leading 'upsert' or 'delete' tag a union puts on every path."""
    return {".".join(str(part) for part in err["loc"][1:]) for err in exc_info.value.errors()}


# 1. shape: strict model, unknown fields rejected


def test_valid_upsert_is_accepted() -> None:
    req = UpsertRequest.model_validate(make_upsert())
    assert req.doc_id == "B07ABC1234"
    assert req.version == 3
    assert req.bullets == ["Quiet clicks", "18 month battery"]


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        UpsertRequest.model_validate(make_upsert(esci_label="E"))
    assert bad_fields(exc_info) == {"esci_label"}


# 2. doc_id: it becomes part of an S3 key and a point id, so the alphabet is tight


@pytest.mark.parametrize("doc_id", ["A", "B07ABC1234", "a" * 64, "with_under-score_9"])
def test_doc_id_accepts_valid_values(doc_id: str) -> None:
    assert UpsertRequest.model_validate(make_upsert(doc_id=doc_id)).doc_id == doc_id


@pytest.mark.parametrize(
    "doc_id",
    ["", "a" * 65, "has/slash", "../up", "with space", "tab\t", "nul\x00", "ünï", "trailing\n"],
)
def test_doc_id_rejects_invalid_values(doc_id: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        UpsertRequest.model_validate(make_upsert(doc_id=doc_id))
    assert bad_fields(exc_info) == {"doc_id"}


# 3. version: a strict integer in the bigint range


@pytest.mark.parametrize("version", [0, 1, MAX_VERSION])
def test_version_accepts_valid_values(version: int) -> None:
    assert UpsertRequest.model_validate(make_upsert(version=version)).version == version


@pytest.mark.parametrize("version", [-1, MAX_VERSION + 1, 1.5, 1.0, "3", True, None])
def test_version_rejects_invalid_values(version: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        UpsertRequest.model_validate(make_upsert(version=version))
    assert bad_fields(exc_info) == {"version"}


# 4. product fields: required, nullable, lengths, locale


@pytest.mark.parametrize(
    ("field", "limit"),
    [("title", 400), ("brand", 100), ("color", 100), ("description", 5000)],
)
def test_text_field_length_boundary(field: str, limit: int) -> None:
    assert UpsertRequest.model_validate(make_upsert(**{field: "x" * limit}))
    with pytest.raises(ValidationError) as exc_info:
        UpsertRequest.model_validate(make_upsert(**{field: "x" * (limit + 1)}))
    assert bad_fields(exc_info) == {field}


def test_title_is_required() -> None:
    body = make_upsert()
    del body["title"]
    with pytest.raises(ValidationError) as exc_info:
        UpsertRequest.model_validate(body)
    assert bad_fields(exc_info) == {"title"}


@pytest.mark.parametrize("title", [None, "", "   ", "\t\n"])
def test_title_must_have_visible_text(title: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        UpsertRequest.model_validate(make_upsert(title=title))
    assert bad_fields(exc_info) == {"title"}


@pytest.mark.parametrize("field", ["brand", "color", "description"])
def test_optional_fields_may_be_null_or_missing_but_not_empty(field: str) -> None:
    assert getattr(UpsertRequest.model_validate(make_upsert(**{field: None})), field) is None
    body = make_upsert()
    del body[field]
    assert getattr(UpsertRequest.model_validate(body), field) is None
    with pytest.raises(ValidationError) as exc_info:
        UpsertRequest.model_validate(make_upsert(**{field: ""}))
    assert bad_fields(exc_info) == {field}


def test_bullets_default_to_empty_and_are_bounded() -> None:
    body = make_upsert()
    del body["bullets"]
    assert UpsertRequest.model_validate(body).bullets == []
    assert UpsertRequest.model_validate(make_upsert(bullets=["b"] * 20))
    with pytest.raises(ValidationError) as too_many:
        UpsertRequest.model_validate(make_upsert(bullets=["b"] * 21))
    assert bad_fields(too_many) == {"bullets"}
    with pytest.raises(ValidationError) as too_long:
        UpsertRequest.model_validate(make_upsert(bullets=["ok", "x" * 1001]))
    assert bad_fields(too_long) == {"bullets.1"}


@pytest.mark.parametrize("locale", ["us", "es", "jp"])
def test_locale_accepts_the_esci_locales(locale: str) -> None:
    assert UpsertRequest.model_validate(make_upsert(locale=locale)).locale == locale


@pytest.mark.parametrize("locale", ["US", "en_US", "", "xx", None])
def test_locale_rejects_everything_else(locale: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        UpsertRequest.model_validate(make_upsert(locale=locale))
    assert bad_fields(exc_info) == {"locale"}


@pytest.mark.parametrize("title", ["a\ud800b", "\udfff"])
def test_lone_surrogates_are_rejected_so_text_always_encodes_to_utf8(title: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        UpsertRequest.model_validate(make_upsert(title=title))
    assert bad_fields(exc_info) == {"title"}


# 5. errors are field level and complete, so the 422 body can name every problem at once


def test_every_bad_field_is_reported_together() -> None:
    with pytest.raises(ValidationError) as exc_info:
        UpsertRequest.model_validate(
            make_upsert(doc_id="", version=-1, title="", locale="xx", esci_label="E")
        )
    assert bad_fields(exc_info) == {"doc_id", "version", "title", "locale", "esci_label"}


# 6. DocEvent: what the outbox writes and the worker reads. Two classes, told apart by `op`.

EVENT_FACTORIES = [make_event, make_delete_event]


def test_upsert_event_round_trips_through_json() -> None:
    event = doc_event_adapter.validate_python(make_event())
    assert isinstance(event, UpsertEvent)
    assert doc_event_adapter.validate_json(doc_event_adapter.dump_json(event)) == event


def test_delete_event_round_trips_through_json() -> None:
    event = doc_event_adapter.validate_python(make_delete_event())
    assert isinstance(event, DeleteEvent)
    assert doc_event_adapter.validate_json(doc_event_adapter.dump_json(event)) == event


@pytest.mark.parametrize("make", EVENT_FACTORIES)
@pytest.mark.parametrize("schema_version", [0, 2, "1", None])
def test_unknown_schema_version_is_rejected(
    make: Callable[..., dict[str, Any]], schema_version: object
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        doc_event_adapter.validate_python(make(schema_version=schema_version))
    assert bad_event_fields(exc_info) == {"schema_version"}


@pytest.mark.parametrize("make", EVENT_FACTORIES)
@pytest.mark.parametrize(
    ("field", "value"),
    [("doc_id", "has/slash"), ("version", -1), ("corpus", "other")],
)
def test_both_events_share_the_same_field_rules(
    make: Callable[..., dict[str, Any]], field: str, value: object
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        doc_event_adapter.validate_python(make(**{field: value}))
    assert bad_event_fields(exc_info) == {field}


@pytest.mark.parametrize("make", EVENT_FACTORIES)
def test_timestamp_must_carry_a_timezone(make: Callable[..., dict[str, Any]]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        doc_event_adapter.validate_python(make(produced_at=datetime(2026, 10, 10, 12, 0)))
    assert bad_event_fields(exc_info) == {"produced_at"}


@pytest.mark.parametrize("op", ["merge", "UPSERT", None])
def test_unknown_op_is_rejected(op: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        doc_event_adapter.validate_python(make_event(op=op))
    assert [err["type"] for err in exc_info.value.errors()] == ["union_tag_invalid"]


def test_missing_op_is_rejected() -> None:
    raw = make_event()
    del raw["op"]
    with pytest.raises(ValidationError) as exc_info:
        doc_event_adapter.validate_python(raw)
    assert [err["type"] for err in exc_info.value.errors()] == ["union_tag_not_found"]


@pytest.mark.parametrize("field", ["s3_key", "content_sha", "size_bytes"])
def test_upsert_event_needs_its_object_fields(field: str) -> None:
    missing = make_event()
    del missing[field]
    null = make_event(**{field: None})
    for raw in (missing, null):
        with pytest.raises(ValidationError) as exc_info:
            doc_event_adapter.validate_python(raw)
        assert bad_event_fields(exc_info) == {field}


@pytest.mark.parametrize(
    ("field", "value"),
    [("s3_key", "esci/B07ABC1234/v4.json"), ("content_sha", "a" * 64), ("size_bytes", 512)],
)
def test_delete_event_must_not_carry_object_fields(field: str, value: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        doc_event_adapter.validate_python(make_delete_event(**{field: value}))
    assert bad_event_fields(exc_info) == {field}
