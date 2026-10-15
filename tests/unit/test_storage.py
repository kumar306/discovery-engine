"""The S3 adapter with a scripted stand in for boto3: error handling, retries, shedding, wiring."""

import asyncio
import io
import random
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    NoCredentialsError,
    ParamValidationError,
    ReadTimeoutError,
)

from libs.resilience.bulkhead import Bulkhead, Overloaded
from libs.resilience.retry import PermanentError, RetryPolicy, TransientError
from libs.schemas.hashing import content_sha
from services.ingest.settings import IngestSettings
from services.ingest.storage import (
    DocumentStore,
    MissingObject,
    S3PutResult,
    StorageUnavailable,
    build_store,
    classify,
)

POLICY = RetryPolicy(max_attempts=3, base_delay=0.1, max_delay=5.0)
DATA = b'{"a": 1}'
OTHER = b'{"a": 2}'


def client_error(status: int, code: str) -> ClientError:
    response = {
        "Error": {"Code": code, "Message": "m"},
        "ResponseMetadata": {"HTTPStatusCode": status},
    }
    return ClientError(response, "PutObject")


class StubS3:
    """Replays a script: each put or get call takes the next outcome and the last one repeats."""

    def __init__(self, puts: list[object], gets: list[object] | None = None) -> None:
        self.puts, self.gets = puts, gets or []
        self.put_calls: list[dict[str, Any]] = []
        self.get_calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.put_calls.append(kwargs)
        self.entered.set()
        self.release.wait(timeout=5)
        outcome = self.puts[min(len(self.put_calls) - 1, len(self.puts) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return {}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        self.get_calls += 1
        self.entered.set()
        self.release.wait(timeout=5)
        outcome = self.gets[min(self.get_calls - 1, len(self.gets) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, bytes)
        return {"Body": io.BytesIO(outcome)}


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


@pytest.fixture
def make_store() -> Iterator[Any]:
    created: list[DocumentStore] = []

    def build(client: StubS3, limit: int = 4, wait: float = 0.05) -> tuple[DocumentStore, Sleeps]:
        sleeps = Sleeps()
        store = DocumentStore(
            client,
            "bucket",
            ThreadPoolExecutor(max_workers=8),
            Bulkhead(limit, wait),
            POLICY,
            sleeps,
            random.Random(1),
        )
        created.append(store)
        return store, sleeps

    yield build
    for store in created:
        store.close()


# what counts as worth retrying


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(client_error(500, "InternalError"), TransientError, id="s3 500"),
        pytest.param(client_error(503, "SlowDown"), TransientError, id="slow down"),
        pytest.param(
            client_error(409, "ConditionalRequestConflict"), TransientError, id="racing write"
        ),
        pytest.param(client_error(400, "RequestTimeout"), TransientError, id="request timeout"),
        pytest.param(client_error(403, "AccessDenied"), PermanentError, id="access denied"),
        pytest.param(client_error(404, "NoSuchBucket"), PermanentError, id="no such bucket"),
        pytest.param(client_error(403, "InvalidAccessKeyId"), PermanentError, id="bad key"),
        pytest.param(client_error(400, "InvalidRequest"), PermanentError, id="bad request"),
        pytest.param(
            EndpointConnectionError(endpoint_url="x"), TransientError, id="cannot connect"
        ),
        pytest.param(ConnectTimeoutError(endpoint_url="x"), TransientError, id="connect timeout"),
        pytest.param(ReadTimeoutError(endpoint_url="x"), TransientError, id="read timeout"),
        pytest.param(
            ConnectionClosedError(endpoint_url="x"), TransientError, id="connection closed"
        ),
        pytest.param(NoCredentialsError(), PermanentError, id="no credentials"),
        pytest.param(ParamValidationError(report="bad"), PermanentError, id="bad parameters"),
    ],
)
def test_classification(error: Exception, expected: type[Exception]) -> None:
    assert type(classify(error)) is expected


# the put


async def test_a_new_object_is_written_only_if_absent(make_store: Any) -> None:
    client = StubS3([None])
    store, sleeps = make_store(client)
    assert await store.put_if_absent("esci/D1/v1.json", DATA, "sha") is S3PutResult.S3_PUT_CREATED
    assert client.put_calls == [
        {
            "Bucket": "bucket",
            "Key": "esci/D1/v1.json",
            "Body": DATA,
            "ContentType": "application/json",
            "IfNoneMatch": "*",
        }
    ]
    assert sleeps.delays == []


async def test_transient_errors_are_retried_then_it_works(make_store: Any) -> None:
    client = StubS3(
        [client_error(503, "SlowDown"), EndpointConnectionError(endpoint_url="x"), None]
    )
    store, sleeps = make_store(client)
    assert await store.put_if_absent("k", DATA, "sha") is S3PutResult.S3_PUT_CREATED
    assert len(client.put_calls) == 3
    assert len(sleeps.delays) == 2


async def test_a_transient_error_that_never_clears_becomes_storage_unavailable(
    make_store: Any,
) -> None:
    client = StubS3([client_error(503, "SlowDown")])
    store, sleeps = make_store(client)
    with pytest.raises(StorageUnavailable):
        await store.put_if_absent("k", DATA, "sha")
    assert len(client.put_calls) == POLICY.max_attempts
    assert len(sleeps.delays) == POLICY.max_attempts - 1  # no wait after the last attempt


async def test_a_permanent_error_is_not_retried(make_store: Any) -> None:
    client = StubS3([client_error(403, "AccessDenied")])
    store, sleeps = make_store(client)
    with pytest.raises(StorageUnavailable):
        await store.put_if_absent("k", DATA, "sha")
    assert len(client.put_calls) == 1
    assert sleeps.delays == []


# an object that is already there


async def test_an_existing_object_with_the_same_content_is_already_there(make_store: Any) -> None:
    client = StubS3([client_error(412, "PreconditionFailed")], gets=[DATA])
    store, _ = make_store(client)
    assert (
        await store.put_if_absent("k", DATA, content_sha(DATA)) is S3PutResult.S3_PUT_ALREADY_THERE
    )
    assert (len(client.put_calls), client.get_calls) == (1, 1)


async def test_an_existing_object_with_other_content_is_a_conflict(make_store: Any) -> None:
    client = StubS3([client_error(412, "PreconditionFailed")], gets=[OTHER])
    store, _ = make_store(client)
    assert await store.put_if_absent("k", DATA, content_sha(DATA)) is S3PutResult.S3_PUT_CONFLICT


async def test_a_failing_read_of_the_existing_object_retries_the_whole_call(
    make_store: Any,
) -> None:
    client = StubS3(
        [client_error(412, "PreconditionFailed")],
        gets=[client_error(500, "InternalError"), DATA],
    )
    store, sleeps = make_store(client)
    assert (
        await store.put_if_absent("k", DATA, content_sha(DATA)) is S3PutResult.S3_PUT_ALREADY_THERE
    )
    assert (len(client.put_calls), client.get_calls, len(sleeps.delays)) == (2, 2, 1)


async def test_every_write_asks_for_if_none_match_so_nothing_is_overwritten(
    make_store: Any,
) -> None:
    client = StubS3([client_error(503, "SlowDown"), None])
    store, _ = make_store(client)
    await store.put_if_absent("k", DATA, "sha")
    assert all(call["IfNoneMatch"] == "*" for call in client.put_calls)


# reading: the worker fetches the document an event points at


async def test_a_document_is_read_back_whole(make_store: Any) -> None:
    client = StubS3([None], gets=[DATA])
    store, sleeps = make_store(client)
    assert await store.get("esci/D1/v1.json") == DATA
    assert (client.get_calls, sleeps.delays) == (1, [])


async def test_a_read_that_fails_at_first_is_retried_then_works(make_store: Any) -> None:
    gets = [client_error(500, "InternalError"), client_error(503, "SlowDown"), DATA]
    client = StubS3([None], gets=gets)
    store, sleeps = make_store(client)
    assert await store.get("k") == DATA
    assert (client.get_calls, len(sleeps.delays)) == (3, 2)


async def test_a_read_that_never_clears_is_storage_unavailable_not_a_missing_document(
    make_store: Any,
) -> None:
    client = StubS3([None], gets=[client_error(500, "InternalError")])
    store, _ = make_store(client)
    with pytest.raises(StorageUnavailable):
        await store.get("k")
    assert client.get_calls == 3


async def test_a_key_that_does_not_exist_is_a_permanent_error_and_is_not_retried(
    make_store: Any,
) -> None:
    client = StubS3([None], gets=[client_error(404, "NoSuchKey")])
    store, sleeps = make_store(client)
    with pytest.raises(MissingObject) as caught:
        await store.get("esci/D1/v1.json")
    assert isinstance(caught.value, PermanentError)
    assert (client.get_calls, sleeps.delays) == (1, [])


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(client_error(404, "NoSuchBucket"), id="the bucket is missing"),
        pytest.param(client_error(403, "AccessDenied"), id="access is refused"),
        pytest.param(client_error(403, "InvalidAccessKeyId"), id="the credentials are wrong"),
    ],
)
async def test_a_setup_problem_is_never_mistaken_for_a_missing_document(
    make_store: Any, error: ClientError
) -> None:
    # one wrong setting would otherwise dead letter every single message
    client = StubS3([None], gets=[error])
    store, _ = make_store(client)
    with pytest.raises(StorageUnavailable):
        await store.get("k")
    assert client.get_calls == 1  # a permanent boto error is not retried either


async def test_a_read_goes_through_the_bulkhead_too(make_store: Any) -> None:
    client = StubS3([None], gets=[DATA])
    client.release.clear()  # the first read will sit inside its worker thread
    store, _ = make_store(client, limit=1, wait=0.05)
    first = asyncio.create_task(store.get("a"))
    await asyncio.to_thread(client.entered.wait, 5)
    with pytest.raises(Overloaded):
        await store.get("b")
    client.release.set()
    assert await first == DATA


# shedding


async def test_a_full_bulkhead_sheds_the_next_caller_and_frees_up_afterwards(
    make_store: Any,
) -> None:
    client = StubS3([None])
    client.release.clear()  # the first put will sit inside its worker thread
    store, _ = make_store(client, limit=1, wait=0.05)
    first = asyncio.create_task(store.put_if_absent("a", DATA, "sha"))
    await asyncio.to_thread(client.entered.wait, 5)
    with pytest.raises(Overloaded):
        await store.put_if_absent("b", DATA, "sha")
    client.release.set()
    assert await first is S3PutResult.S3_PUT_CREATED
    assert await store.put_if_absent("c", DATA, "sha") is S3PutResult.S3_PUT_CREATED


async def test_the_permit_comes_back_after_a_failure(make_store: Any) -> None:
    client = StubS3([client_error(403, "AccessDenied"), None])
    store, _ = make_store(client, limit=1, wait=0.05)
    with pytest.raises(StorageUnavailable):
        await store.put_if_absent("a", DATA, "sha")
    assert await store.put_if_absent("b", DATA, "sha") is S3PutResult.S3_PUT_CREATED


# wiring


def test_the_store_is_built_from_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INGEST_S3_TIMEOUT_S", "1.5")
    monkeypatch.setenv("INGEST_S3_THREADS", "20")
    monkeypatch.setenv("INGEST_S3_BULKHEAD_LIMIT", "16")
    monkeypatch.setenv("INGEST_S3_ENDPOINT_URL", "http://127.0.0.1:5000")
    store = build_store(IngestSettings())
    try:
        config = store._s3_client.meta.config
        assert (config.connect_timeout, config.read_timeout) == (1.5, 1.5)
        # our own retry policy is the only one, botocore must not retry underneath it
        assert config.retries["total_max_attempts"] == 1
        assert config.max_pool_connections == 20
        assert store._s3_client.meta.endpoint_url == "http://127.0.0.1:5000"
    finally:
        store.close()
