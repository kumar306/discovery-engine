"""The S3 adapter against Moto, the local S3 stand in: real conditional writes, real failures."""

import asyncio
import threading
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest

from libs.resilience.bulkhead import Overloaded
from libs.schemas import UpsertRequest
from libs.schemas.hashing import content_sha
from services.ingest.events import prepare_upsert
from services.ingest.storage import DocumentStore, MissingObject, PutResult, StorageUnavailable

from .conftest import POLICY, S3_BULKHEAD_LIMIT, Sleeps, make_store, moto_client

NOW = datetime(2026, 10, 12, 9, 30, tzinfo=UTC)


@pytest.fixture
def store(bucket: str) -> Iterator[tuple[DocumentStore, Sleeps]]:
    created = make_store(moto_client(), bucket)
    yield created
    created[0].close()


def request(doc_id: str = "D1", version: int = 1, title: str = "Wireless Mouse") -> UpsertRequest:
    return UpsertRequest.model_validate(
        {"corpus": "esci", "doc_id": doc_id, "version": version, "title": title, "locale": "us"}
    )


def stored(bucket: str, key: str) -> bytes:
    return moto_client().get_object(Bucket=bucket, Key=key)["Body"].read()


async def test_the_canonical_bytes_land_at_the_derived_key(
    store: tuple[DocumentStore, Sleeps], bucket: str
) -> None:
    prepared = prepare_upsert(request(), NOW)
    result = await store[0].put_if_absent(prepared.key, prepared.data, prepared.content_sha)
    assert result is PutResult.CREATED
    assert prepared.key == "esci/D1/v1.json"
    assert stored(bucket, prepared.key) == prepared.data
    assert content_sha(stored(bucket, prepared.key)) == prepared.content_sha
    head = moto_client().head_object(Bucket=bucket, Key=prepared.key)
    assert head["ContentType"] == "application/json"
    assert store[1].delays == []


async def test_writing_the_same_document_again_is_already_there_and_changes_nothing(
    store: tuple[DocumentStore, Sleeps], bucket: str
) -> None:
    prepared = prepare_upsert(request(), NOW)
    await store[0].put_if_absent(prepared.key, prepared.data, prepared.content_sha)
    again = await store[0].put_if_absent(prepared.key, prepared.data, prepared.content_sha)
    assert again is PutResult.ALREADY_THERE
    assert stored(bucket, prepared.key) == prepared.data


async def test_different_content_at_the_same_key_is_a_conflict_and_the_first_one_stays(
    store: tuple[DocumentStore, Sleeps], bucket: str
) -> None:
    first = prepare_upsert(request(title="Wireless Mouse"), NOW)
    second = prepare_upsert(request(title="Other Title"), NOW)
    assert first.key == second.key
    await store[0].put_if_absent(first.key, first.data, first.content_sha)
    result = await store[0].put_if_absent(second.key, second.data, second.content_sha)
    assert result is PutResult.CONFLICT
    assert stored(bucket, first.key) == first.data


async def test_two_writers_of_different_content_get_one_winner_and_one_conflict(
    store: tuple[DocumentStore, Sleeps], bucket: str
) -> None:
    for round_number in range(5):
        a = prepare_upsert(request(f"R{round_number}", title="Title A"), NOW)
        b = prepare_upsert(request(f"R{round_number}", title="Title B"), NOW)
        results = await asyncio.gather(
            store[0].put_if_absent(a.key, a.data, a.content_sha),
            store[0].put_if_absent(b.key, b.data, b.content_sha),
        )
        assert sorted(results) == sorted([PutResult.CREATED, PutResult.CONFLICT])
        winner = a if results[0] is PutResult.CREATED else b
        assert stored(bucket, a.key) == winner.data


async def test_two_writers_of_the_same_content_get_created_and_already_there(
    store: tuple[DocumentStore, Sleeps], bucket: str
) -> None:
    for round_number in range(5):
        prepared = prepare_upsert(request(f"T{round_number}"), NOW)
        results = await asyncio.gather(
            store[0].put_if_absent(prepared.key, prepared.data, prepared.content_sha),
            store[0].put_if_absent(prepared.key, prepared.data, prepared.content_sha),
        )
        assert sorted(results) == sorted([PutResult.CREATED, PutResult.ALREADY_THERE])


async def test_as_many_parallel_writes_as_the_bulkhead_allows_all_succeed(
    store: tuple[DocumentStore, Sleeps], bucket: str
) -> None:
    prepared = [prepare_upsert(request(f"P{n}"), NOW) for n in range(S3_BULKHEAD_LIMIT)]
    results = await asyncio.gather(
        *(store[0].put_if_absent(p.key, p.data, p.content_sha) for p in prepared)
    )
    assert results == [PutResult.CREATED] * len(prepared)
    assert all(stored(bucket, p.key) == p.data for p in prepared)


# failures


async def test_an_unreachable_endpoint_is_retried_then_storage_unavailable(bucket: str) -> None:
    # nothing listens on port 9, so every attempt is refused straight away
    store, sleeps = make_store(moto_client("http://127.0.0.1:9", timeout=0.3), bucket)
    try:
        prepared = prepare_upsert(request(), NOW)
        with pytest.raises(StorageUnavailable):
            await store.put_if_absent(prepared.key, prepared.data, prepared.content_sha)
        assert len(sleeps.delays) == POLICY.max_attempts - 1
    finally:
        store.close()


async def test_a_missing_bucket_is_not_retried_and_the_store_recovers_once_it_exists() -> None:
    name = f"searchtest{uuid.uuid4().hex[:12]}"
    client = moto_client()
    store, sleeps = make_store(client, name, limit=1)
    prepared = prepare_upsert(request(), NOW)
    try:
        with pytest.raises(StorageUnavailable):
            await store.put_if_absent(prepared.key, prepared.data, prepared.content_sha)
        assert sleeps.delays == []
        client.create_bucket(Bucket=name)
        # the single permit was given back, so this call is not shed
        result = await store.put_if_absent(prepared.key, prepared.data, prepared.content_sha)
        assert result is PutResult.CREATED
    finally:
        store.close()
        client.delete_object(Bucket=name, Key=prepared.key)
        client.delete_bucket(Bucket=name)


class BlockingClient:
    """The real client, except that put_object waits for the test to let it go."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self.entered = threading.Event()
        self.release = threading.Event()

    def put_object(self, **kwargs: Any) -> Any:
        self.entered.set()
        self.release.wait(timeout=10)
        return self._real.put_object(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


async def test_a_slow_write_fills_the_bulkhead_and_the_next_caller_is_shed(bucket: str) -> None:
    blocking = BlockingClient(moto_client())
    store, _ = make_store(blocking, bucket, limit=1)
    slow = prepare_upsert(request("SLOW"), NOW)
    other = prepare_upsert(request("OTHER"), NOW)
    try:
        first = asyncio.create_task(store.put_if_absent(slow.key, slow.data, slow.content_sha))
        await asyncio.to_thread(blocking.entered.wait, 5)
        with pytest.raises(Overloaded):
            await store.put_if_absent(other.key, other.data, other.content_sha)
        blocking.release.set()
        assert await first is PutResult.CREATED
        assert (
            await store.put_if_absent(other.key, other.data, other.content_sha) is PutResult.CREATED
        )
    finally:
        blocking.release.set()
        store.close()


async def test_a_stored_document_is_read_back_byte_for_byte(
    store: tuple[DocumentStore, Sleeps],
) -> None:
    prepared = prepare_upsert(request(title="Ergonomic Mouse"), NOW)
    await store[0].put_if_absent(prepared.key, prepared.data, prepared.content_sha)
    data = await store[0].get(prepared.key)
    assert data == prepared.data
    assert content_sha(data) == prepared.content_sha


async def test_a_key_that_was_never_written_is_a_missing_object(
    store: tuple[DocumentStore, Sleeps],
) -> None:
    with pytest.raises(MissingObject):
        await store[0].get("esci/NOPE/v1.json")
    assert store[1].delays == []  # not retried


async def test_reading_from_a_bucket_that_does_not_exist_is_not_a_missing_document() -> None:
    created = make_store(moto_client(), f"no-such-bucket-{uuid.uuid4().hex[:8]}")
    try:
        with pytest.raises(StorageUnavailable):
            await created[0].get("esci/D1/v1.json")
    finally:
        created[0].close()
