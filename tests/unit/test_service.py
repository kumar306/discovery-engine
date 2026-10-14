"""The ingest flow without any io: precheck, then s3, then the compare and set, in that order."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from libs.schemas import UpsertRequest
from services.ingest.events import EVENT_SCHEMA, prepare_delete, prepare_upsert
from services.ingest.repository import AcceptResult, PgWriteResult
from services.ingest.rules import KnownDocument
from services.ingest.service import IngestService, WriteAccepted, WriteRefused
from services.ingest.storage import S3PutResult

T0 = datetime(2026, 10, 12, 9, 30, tzinfo=UTC)


def request(version: int = 5, title: str = "Wireless Mouse") -> UpsertRequest:
    return UpsertRequest.model_validate(
        {"corpus": "esci", "doc_id": "D1", "version": version, "title": title, "locale": "us"}
    )


class StubRepository:
    def __init__(self, known: KnownDocument | None, accept: AcceptResult | None = None) -> None:
        self._known, self._accept = known, accept
        self.accepted: list[tuple[Any, ...]] = []

    async def known(self, corpus: str, doc_id: str) -> KnownDocument | None:
        return self._known

    async def accept(self, *args: Any) -> AcceptResult:
        self.accepted.append(args)
        assert self._accept is not None, "accept must not be reached in this test"
        return self._accept


class StubStore:
    def __init__(self, result: S3PutResult = S3PutResult.S3_PUT_CREATED) -> None:
        self._result = result
        self.puts: list[tuple[str, bytes, str]] = []

    async def put_if_absent(self, key: str, data: bytes, sha: str) -> S3PutResult:
        self.puts.append((key, data, sha))
        return self._result


def service(repo: StubRepository, store: StubStore) -> IngestService:
    ticks = iter(T0 + timedelta(seconds=n) for n in range(10))
    return IngestService(repo, store, clock=lambda: next(ticks))  # type: ignore[arg-type]


SHA = prepare_upsert(request(), T0).content_sha


# the precheck answers before anything is written


@pytest.mark.parametrize(
    ("known", "expected"),
    [
        pytest.param(
            KnownDocument(7, SHA), WriteRefused("stale_version", 7), id="a newer version is held"
        ),
        pytest.param(
            KnownDocument(5, "f" * 64),
            WriteRefused("content_conflict", 5),
            id="same version, other content",
        ),
        pytest.param(
            KnownDocument(5, None),
            WriteRefused("content_conflict", 5),
            id="same version, was a delete",
        ),
    ],
)
async def test_the_precheck_refuses_without_touching_s3_or_postgres(
    known: KnownDocument, expected: WriteRefused
) -> None:
    repo, store = StubRepository(known), StubStore()
    assert await service(repo, store).upsert(request(5), "r") == expected
    assert store.puts == []
    assert repo.accepted == []


async def test_a_completed_request_is_accepted_again_without_any_write() -> None:
    repo, store = StubRepository(KnownDocument(5, SHA)), StubStore()
    result = await service(repo, store).upsert(request(5), "r")
    assert isinstance(result, WriteAccepted)
    assert (store.puts, repo.accepted) == ([], [])


# the order of the writes


async def test_s3_is_written_first_then_postgres_with_the_matching_event() -> None:
    repo = StubRepository(None, AcceptResult(PgWriteResult.PG_WRITE_APPLIED, 5))
    store = StubStore(S3PutResult.S3_PUT_CREATED)
    result = await service(repo, store).upsert(request(5), "req-9")
    prepared = prepare_upsert(request(5), T0)
    assert store.puts == [(prepared.key, prepared.data, prepared.content_sha)]
    ((corpus, doc_id, version, sha, outgoing),) = repo.accepted
    assert (corpus, doc_id, version, sha) == ("esci", "D1", 5, prepared.content_sha)
    assert (outgoing.request_id, outgoing.event_schema) == ("req-9", EVENT_SCHEMA)
    assert outgoing.payload == prepared.event_json
    assert isinstance(result, WriteAccepted)


async def test_an_s3_conflict_stops_before_postgres() -> None:
    repo, store = StubRepository(None), StubStore(S3PutResult.S3_PUT_CONFLICT)
    assert await service(repo, store).upsert(request(5), "r") == WriteRefused(
        "content_conflict", None
    )
    assert repo.accepted == []


async def test_an_s3_conflict_reports_the_version_we_hold() -> None:
    repo, store = StubRepository(KnownDocument(3, SHA)), StubStore(S3PutResult.S3_PUT_CONFLICT)
    assert await service(repo, store).upsert(request(5), "r") == WriteRefused("content_conflict", 3)


async def test_an_object_already_there_from_a_crashed_attempt_still_gets_its_row() -> None:
    repo = StubRepository(None, AcceptResult(PgWriteResult.PG_WRITE_APPLIED, 5))
    store = StubStore(S3PutResult.S3_PUT_ALREADY_THERE)
    assert isinstance(await service(repo, store).upsert(request(5), "r"), WriteAccepted)
    assert len(repo.accepted) == 1


# what the compare and set can still say after the precheck passed


@pytest.mark.parametrize(
    ("accept", "expected"),
    [
        pytest.param(
            AcceptResult(PgWriteResult.PG_WRITE_APPLIED, 5),
            WriteAccepted("D1", 5, T0 + timedelta(seconds=1)),
            id="won",
        ),
        pytest.param(
            AcceptResult(PgWriteResult.PG_WRITE_DUPLICATE, 5),
            WriteAccepted("D1", 5, T0 + timedelta(seconds=1)),
            id="an identical write won first",
        ),
        pytest.param(
            AcceptResult(PgWriteResult.PG_WRITE_REJECTED, 8),
            WriteRefused("stale_version", 8),
            id="a newer version won first",
        ),
        pytest.param(
            AcceptResult(PgWriteResult.PG_WRITE_REJECTED, 5),
            WriteRefused("content_conflict", 5),
            id="same version, other content won",
        ),
    ],
)
async def test_the_compare_and_set_has_the_last_word(
    accept: AcceptResult, expected: WriteAccepted | WriteRefused
) -> None:
    repo, store = StubRepository(None, accept), StubStore()
    assert await service(repo, store).upsert(request(5), "r") == expected


async def test_the_event_time_is_read_before_the_work_and_accepted_at_after_it() -> None:
    repo = StubRepository(None, AcceptResult(PgWriteResult.PG_WRITE_APPLIED, 5))
    result = await service(repo, StubStore()).upsert(request(5), "r")
    assert isinstance(result, WriteAccepted)
    assert result.accepted_at == T0 + timedelta(seconds=1)
    assert prepare_upsert(request(5), T0).event_json == repo.accepted[0][4].payload


# delete: the same flow without the s3 step


@pytest.mark.parametrize(
    ("known", "expected"),
    [
        pytest.param(
            KnownDocument(7, None), WriteRefused("stale_version", 7), id="a newer version is held"
        ),
        pytest.param(
            KnownDocument(5, SHA),
            WriteRefused("content_conflict", 5),
            id="an upsert holds this version",
        ),
    ],
)
async def test_a_delete_the_precheck_refuses_writes_nothing(
    known: KnownDocument, expected: WriteRefused
) -> None:
    repo, store = StubRepository(known), StubStore()
    assert await service(repo, store).delete("D1", 5, "r") == expected
    assert (repo.accepted, store.puts) == ([], [])


async def test_a_delete_that_was_already_accepted_is_accepted_again_without_a_write() -> None:
    repo, store = StubRepository(KnownDocument(5, None)), StubStore()
    assert isinstance(await service(repo, store).delete("D1", 5, "r"), WriteAccepted)
    assert (repo.accepted, store.puts) == ([], [])


async def test_a_delete_goes_to_postgres_with_no_sha_and_never_touches_s3() -> None:
    repo = StubRepository(KnownDocument(3, SHA), AcceptResult(PgWriteResult.PG_WRITE_APPLIED, 5))
    store = StubStore()
    result = await service(repo, store).delete("D1", 5, "req-4")
    ((corpus, doc_id, version, sha, outgoing),) = repo.accepted
    assert (corpus, doc_id, version, sha) == ("esci", "D1", 5, None)
    assert (outgoing.request_id, outgoing.event_schema) == ("req-4", EVENT_SCHEMA)
    assert outgoing.payload == prepare_delete("esci", "D1", 5, T0).event_json
    assert store.puts == []
    assert isinstance(result, WriteAccepted)


@pytest.mark.parametrize(
    ("accept", "expected"),
    [
        pytest.param(
            AcceptResult(PgWriteResult.PG_WRITE_APPLIED, 5),
            WriteAccepted("D1", 5, T0 + timedelta(seconds=1)),
            id="won",
        ),
        pytest.param(
            AcceptResult(PgWriteResult.PG_WRITE_DUPLICATE, 5),
            WriteAccepted("D1", 5, T0 + timedelta(seconds=1)),
            id="an identical delete won first",
        ),
        pytest.param(
            AcceptResult(PgWriteResult.PG_WRITE_REJECTED, 8),
            WriteRefused("stale_version", 8),
            id="a newer version won first",
        ),
        pytest.param(
            AcceptResult(PgWriteResult.PG_WRITE_REJECTED, 5),
            WriteRefused("content_conflict", 5),
            id="an upsert won this version",
        ),
    ],
)
async def test_the_compare_and_set_has_the_last_word_for_a_delete_too(
    accept: AcceptResult, expected: WriteAccepted | WriteRefused
) -> None:
    repo, store = StubRepository(None, accept), StubStore()
    assert await service(repo, store).delete("D1", 5, "r") == expected
