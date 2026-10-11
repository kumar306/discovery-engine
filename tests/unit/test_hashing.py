"""A3.1 canonical bytes and content sha, C4.8 chunk ids: identities that must never drift."""

import json
import re
import uuid
from typing import Any

import pytest

from libs.schemas import UpsertRequest
from libs.schemas.hashing import canonical_bytes, chunk_id, content_sha

MODEL = "bge-small-en-v1.5"
HASH_A = "a" * 64
HASH_B = "b" * 64


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


# 10. canonical serialisation: the same document is always the same bytes


def test_key_order_in_the_input_does_not_change_the_bytes() -> None:
    forward = UpsertRequest.model_validate_json(
        '{"corpus":"esci","doc_id":"D1","version":1,"title":"t","locale":"us"}'
    )
    backward = UpsertRequest.model_validate_json(
        '{"locale":"us","title":"t","version":1,"doc_id":"D1","corpus":"esci"}'
    )
    assert canonical_bytes(forward) == canonical_bytes(backward)


def test_output_is_compact_json_with_sorted_keys() -> None:
    data = canonical_bytes(make_request())
    assert b": " not in data
    assert b", " not in data
    assert list(json.loads(data)) == sorted(json.loads(data))


def test_list_order_is_kept() -> None:
    first = canonical_bytes(make_request(bullets=["one", "two"]))
    swapped = canonical_bytes(make_request(bullets=["two", "one"]))
    assert first != swapped
    assert json.loads(first)["bullets"] == ["one", "two"]


def test_omitted_and_null_optional_fields_give_the_same_bytes() -> None:
    omitted = UpsertRequest.model_validate(
        {"corpus": "esci", "doc_id": "D1", "version": 1, "title": "t", "locale": "us"}
    )
    explicit = make_request(
        doc_id="D1", version=1, title="t", brand=None, color=None, bullets=[], description=None
    )
    assert canonical_bytes(omitted) == canonical_bytes(explicit)


def test_non_ascii_text_is_written_as_utf8_not_escaped() -> None:
    data = canonical_bytes(make_request(title="Café 鼠标"))
    assert "Café 鼠标".encode() in data
    assert b"\\u" not in data


def test_the_serialiser_does_not_normalise_unicode() -> None:
    composed = canonical_bytes(make_request(title="café"))
    decomposed = canonical_bytes(make_request(title="café"))
    assert composed != decomposed  # NFC is the composer's job (C4.2), not the stored document's


# 11. pinned vectors: if one of these fails, every stored hash and point id has just changed

PINNED_BYTES = (
    b'{"brand":"Logitech","bullets":["Quiet clicks","18 month battery"],"color":"Black",'
    b'"corpus":"esci","description":"A compact wireless mouse.","doc_id":"B07ABC1234",'
    b'"locale":"us","title":"Wireless Mouse","version":3}'
)
PINNED_SHA = "ffaf9c81703a8e1eae3c0656e10c3d2d504ef6cdd58563936b9b1d8985d6ce26"


def test_canonical_form_is_pinned() -> None:
    assert canonical_bytes(make_request()) == PINNED_BYTES


def test_content_sha_of_the_pinned_document() -> None:
    assert content_sha(PINNED_BYTES) == PINNED_SHA


def test_content_sha_matches_the_published_sha256_test_vector() -> None:
    assert content_sha(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_content_sha_is_lowercase_hex_of_the_length_the_schema_expects() -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", content_sha(b""))


# 12. chunk ids: content addressed, per document and per embedding model


def test_chunk_id_is_pinned() -> None:
    expected = uuid.UUID("504b51fe-291e-5f5d-955a-2bda5b2443a2")
    assert chunk_id("B07ABC1234", MODEL, HASH_A) == expected


def test_chunk_id_is_stable_and_a_version_5_uuid() -> None:
    first = chunk_id("D1", MODEL, HASH_A)
    assert chunk_id("D1", MODEL, HASH_A) == first
    assert first.version == 5


@pytest.mark.parametrize(
    ("other_doc", "other_model", "other_hash"),
    [
        pytest.param("D2", MODEL, HASH_A, id="another document"),
        pytest.param("D1", "bge-base-en-v1.5", HASH_A, id="another embedding model"),
        pytest.param("D1", MODEL, HASH_B, id="other chunk text"),
    ],
)
def test_chunk_id_changes_when_any_input_changes(
    other_doc: str, other_model: str, other_hash: str
) -> None:
    assert chunk_id(other_doc, other_model, other_hash) != chunk_id("D1", MODEL, HASH_A)


def test_chunk_id_inputs_cannot_shift_between_fields() -> None:
    assert chunk_id("D1", "m:x", HASH_A) != chunk_id("D1:m", "x", HASH_A)
    assert chunk_id("ab", "c", HASH_A) != chunk_id("a", "bc", HASH_A)
