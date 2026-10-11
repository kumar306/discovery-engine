"""A2 version pre-check: a cheap early rejection before any write (A4.2 is the real guard)."""

import pytest

from services.ingest.rules import KnownDocument, PrecheckOutcome, precheck

SHA_A = "a" * 64
SHA_B = "b" * 64

PROCEED = PrecheckOutcome.PROCEED
DUPLICATE = PrecheckOutcome.ACCEPT_DUPLICATE
STALE = PrecheckOutcome.REJECT_STALE
CONFLICT = PrecheckOutcome.REJECT_CONFLICT


@pytest.mark.parametrize(
    ("version", "content_sha", "known", "expected"),
    [
        pytest.param(0, SHA_A, None, PROCEED, id="unknown document, version zero"),
        pytest.param(7, SHA_A, None, PROCEED, id="unknown document, any version"),
        pytest.param(6, SHA_B, KnownDocument(5, SHA_A), PROCEED, id="higher version"),
        pytest.param(4, SHA_B, KnownDocument(5, SHA_A), STALE, id="lower version"),
        pytest.param(0, SHA_A, KnownDocument(5, SHA_A), STALE, id="lower, even with same content"),
        pytest.param(
            5, SHA_A, KnownDocument(5, SHA_A), DUPLICATE, id="equal version, same content"
        ),
        pytest.param(
            5, SHA_B, KnownDocument(5, SHA_A), CONFLICT, id="equal version, other content"
        ),
        pytest.param(6, None, KnownDocument(5, SHA_A), PROCEED, id="delete over an upsert"),
        pytest.param(5, None, KnownDocument(5, None), DUPLICATE, id="delete retried"),
        pytest.param(
            5, None, KnownDocument(5, SHA_A), CONFLICT, id="delete at an upsert's version"
        ),
        pytest.param(5, SHA_A, KnownDocument(5, None), CONFLICT, id="upsert at a delete's version"),
        pytest.param(6, SHA_A, KnownDocument(5, None), PROCEED, id="upsert after a delete"),
        pytest.param(4, SHA_A, KnownDocument(5, None), STALE, id="older upsert after a delete"),
    ],
)
def test_precheck_outcome(
    version: int, content_sha: str | None, known: KnownDocument | None, expected: PrecheckOutcome
) -> None:
    assert precheck(version, content_sha, known) is expected
