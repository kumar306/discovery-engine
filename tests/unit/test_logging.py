"""JSON log lines: one line per record, carrying the request id of the request being served."""

import io
import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest

from libs.observability.logging import JsonFormatter, configure_logging, request_id_var


def emit(message: str = "hello", *args: Any, **kwargs: Any) -> str:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger(f"test.json.{id(stream)}")
    logger.handlers[:] = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.info(message, *args, **kwargs)
    return stream.getvalue()


@pytest.fixture
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def test_one_record_is_exactly_one_line_of_json() -> None:
    output = emit("hello")
    assert output.count("\n") == 1
    assert json.loads(output)["message"] == "hello"


def test_the_standard_keys_are_there() -> None:
    entry = json.loads(emit("hello"))
    assert entry["level"] == "INFO"
    assert entry["logger"].startswith("test.json.")
    assert datetime.fromisoformat(entry["time"]).tzinfo is not None
    assert set(entry) >= {"time", "level", "logger", "message", "request_id"}


def test_the_request_id_is_null_outside_a_request_and_set_inside_one() -> None:
    assert json.loads(emit())["request_id"] is None
    token = request_id_var.set("req-123")
    try:
        assert json.loads(emit())["request_id"] == "req-123"
    finally:
        request_id_var.reset(token)
    assert json.loads(emit())["request_id"] is None


def test_extra_fields_become_top_level_keys() -> None:
    entry = json.loads(emit("request", extra={"fields": {"status": 202, "doc_id": "D1"}}))
    assert (entry["status"], entry["doc_id"]) == (202, "D1")


def test_a_message_with_newlines_and_quotes_stays_on_one_line_and_round_trips() -> None:
    nasty = 'first line\nsecond "quoted" line\r\nthird'
    output = emit("%s", nasty)
    assert output.count("\n") == 1
    assert json.loads(output)["message"] == nasty


def test_an_exception_is_included_as_text_on_the_same_line() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        output = emit("failed", exc_info=True)
    assert output.count("\n") == 1
    assert "ValueError: boom" in json.loads(output)["exception"]


def test_values_json_cannot_write_are_turned_into_text() -> None:
    when = datetime(2026, 10, 12, 9, 30, tzinfo=UTC)
    entry = json.loads(emit("x", extra={"fields": {"when": when, "tags": {"a"}}}))
    assert entry["when"] == str(when)
    assert entry["tags"] == str({"a"})


def test_non_ascii_text_is_kept_readable() -> None:
    output = emit("café 鼠标")
    assert "\\u" not in output
    assert json.loads(output)["message"] == "café 鼠标"


def test_configure_logging_installs_one_json_handler_even_when_called_twice(
    capsys: pytest.CaptureFixture[str], restore_root_logger: None
) -> None:
    configure_logging("INFO")
    configure_logging("INFO")
    assert len(logging.getLogger().handlers) == 1
    logging.getLogger("somewhere").info("hi")
    lines = capsys.readouterr().out.strip().splitlines()
    assert [json.loads(line)["message"] for line in lines] == ["hi"]
