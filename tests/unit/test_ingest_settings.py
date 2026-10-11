"""Ingest settings: sane defaults, environment overrides, and one safety rule between two pools."""

import os

import pytest
from pydantic import ValidationError

from services.ingest.settings import IngestSettings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # a variable set in the developer's shell must not change what these tests see
    for name in [name for name in os.environ if name.startswith("INGEST_")]:
        monkeypatch.delenv(name)


def test_the_defaults_are_valid_and_point_at_pgbouncer() -> None:
    settings = IngestSettings()
    assert settings.database_dsn.endswith(":6432/search")
    assert settings.topic_events == "doc.events"
    assert settings.s3_bulkhead_limit <= int(settings.s3_threads * 0.8)


def test_environment_variables_override_the_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INGEST_S3_THREADS", "20")
    monkeypatch.setenv("INGEST_S3_BULKHEAD_LIMIT", "16")
    monkeypatch.setenv("INGEST_TOPIC_EVENTS", "doc.events.test.abc")
    settings = IngestSettings()
    assert (settings.s3_threads, settings.s3_bulkhead_limit) == (20, 16)
    assert settings.topic_events == "doc.events.test.abc"


@pytest.mark.parametrize(
    ("threads", "limit", "valid"),
    [
        pytest.param(10, 8, True, id="exactly eighty percent"),
        pytest.param(10, 9, False, id="one above eighty percent"),
        pytest.param(5, 4, True, id="small pool, four of five"),
        pytest.param(5, 5, False, id="the whole small pool"),
        pytest.param(2, 1, True, id="two threads, one permit"),
        pytest.param(1, 1, False, id="one thread leaves no headroom"),
    ],
)
def test_the_bulkhead_must_leave_headroom_in_the_thread_pool(
    monkeypatch: pytest.MonkeyPatch, threads: int, limit: int, valid: bool
) -> None:
    monkeypatch.setenv("INGEST_S3_THREADS", str(threads))
    monkeypatch.setenv("INGEST_S3_BULKHEAD_LIMIT", str(limit))
    if valid:
        assert IngestSettings().s3_bulkhead_limit == limit
        return
    with pytest.raises(ValidationError, match="s3_bulkhead_limit"):
        IngestSettings()


@pytest.mark.parametrize(
    "name",
    [
        "INGEST_DB_POOL_MAX",
        "INGEST_DB_POOL_TIMEOUT_S",
        "INGEST_S3_TIMEOUT_S",
        "INGEST_S3_BULKHEAD_WAIT_S",
        "INGEST_MAX_BODY_BYTES",
        "INGEST_RELAY_BATCH_SIZE",
        "INGEST_RELAY_POLL_INTERVAL_S",
    ],
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_sizes_and_timeouts_must_be_positive(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        IngestSettings()
