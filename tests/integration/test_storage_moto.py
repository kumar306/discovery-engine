"""The S3 adapter against Moto, the local S3 stand in: real conditional writes, real failures."""

import asyncio
import random
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import boto3
import pytest
from botocore.config import Config

from libs.resilience.bulkhead import Bulkhead, Overloaded
from libs.resilience.retry import RetryPolicy
from libs.schemas import UpsertRequest
from libs.schemas.hashing import content_sha
from services.ingest.events import prepare_upsert
from services.ingest.settings import IngestSettings
from services.ingest.storage import DocumentStore, PutResult, StorageUnavailable

ENDPOINT = "http://127.0.0.1:5000"
POLICY = RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.05)
NOW = datetime(2026, 10, 12, 9, 30, tzinfo=UTC)
DEFAULTS = IngestSettings()


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def moto_client(endpoint: str = ENDPOINT, timeout: float = 3.0) -> Any:
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        # moto accepts any credentials, real aws would use the pod role
        aws_access_key_id="test",
        aws_secret_access_key="test",
        config=Config(
            connect_timeout=timeout,
            read_timeout=timeout,
            retries={"total_max_attempts": 1},
            max_pool_connections=DEFAULTS.s3_threads,
        ),
    )


def make_store(
    client: Any, bucket: str, limit: int = DEFAULTS.s3_bulkhead_limit
) -> tuple[DocumentStore, Sleeps]:
    sleeps = Sleeps()
    store = DocumentStore(
        client,
        bucket,
        ThreadPoolExecutor(max_workers=DEFAULTS.s3_threads),
        Bulkhead(limit, wait_timeout=0.05),
        POLICY,
        sleeps,
        random.Random(1),
    )
    return store, sleeps


@pytest.fixture
def bucket() -> Iterator[str]:
    # every test gets its own bucket, bucket names allow only lowercase letters and digits here
    name = f"searchtest{uuid.uuid4().hex[:12]}"
    admin = moto_client()
    admin.create_bucket(Bucket=name)
    yield name
    for page in admin.get_paginator("list_objects_v2").paginate(Bucket=name):
        for item in page.get("Contents", []):
            admin.delete_object(Bucket=name, Key=item["Key"])
    admin.delete_bucket(Bucket=name)


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
    prepared = [prepare_upsert(request(f"P{n}"), NOW) for n in range(DEFAULTS.s3_bulkhead_limit)]
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
