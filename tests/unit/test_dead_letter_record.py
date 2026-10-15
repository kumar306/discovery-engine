"""What goes into a dead letter record: the original message untouched, and what went wrong."""

from types import SimpleNamespace
from typing import Any

from libs.resilience.retry import PermanentError
from services.embed.consumer import dead_letter_for


def message(**fields: Any) -> SimpleNamespace:
    base: dict[str, Any] = {
        "topic": "doc.events",
        "partition": 3,
        "offset": 41,
        "key": b"D1",
        "value": b'{"op": "upsert"}',
        "headers": [("request_id", b"req-1")],
    }
    base.update(fields)
    return SimpleNamespace(**base)


def test_the_key_and_the_body_go_through_untouched_so_a_replay_sends_what_came_in() -> None:
    record = dead_letter_for(message(), PermanentError("bad"), attempts=1)
    assert (record.key, record.value) == (b"D1", b'{"op": "upsert"}')


def test_the_original_headers_stay_first_and_four_headers_say_what_went_wrong() -> None:
    record = dead_letter_for(message(), PermanentError("sha mismatch"), attempts=2)
    assert record.headers[0] == ("request_id", b"req-1")
    assert dict(record.headers) == {
        "request_id": b"req-1",
        "dlq_error_type": b"PermanentError",
        "dlq_error": b"sha mismatch",
        "dlq_attempts": b"2",
        "dlq_source": b"doc.events:3:41",
    }


def test_a_long_error_is_cut_so_the_record_stays_small() -> None:
    record = dead_letter_for(message(), PermanentError("x" * 5000), attempts=1)
    assert len(dict(record.headers)["dlq_error"]) == 500


def test_a_message_without_key_or_headers_is_still_a_valid_record() -> None:
    record = dead_letter_for(message(key=None, headers=None), PermanentError("bad"), attempts=0)
    assert record.key is None
    assert [name for name, _ in record.headers] == [
        "dlq_error_type",
        "dlq_error",
        "dlq_attempts",
        "dlq_source",
    ]


def test_a_message_without_a_body_keeps_it_empty() -> None:
    record = dead_letter_for(message(value=None), PermanentError("no body"), attempts=0)
    assert record.value is None
