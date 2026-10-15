import asyncio
import random
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, HTTPClientError
from botocore.exceptions import ConnectionError as BotoConnectionError

from libs.resilience.bulkhead import Bulkhead
from libs.resilience.retry import (
    PermanentError,
    Rejected,
    RetryPolicy,
    TransientError,
    call_with_retries,
)
from libs.schemas.hashing import content_sha
from services.ingest.settings import IngestSettings

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

# s3 answers these when it is busy or in the middle of another conditional write, a retry can work
_TRANSIENT_CODES = frozenset(
    {
        "SlowDown",
        "RequestTimeout",
        "ServiceUnavailable",
        "InternalError",
        "ConditionalRequestConflict",
        "Throttling",
        "ThrottlingException",
        "RequestLimitExceeded",
    }
)


class S3PutResult(StrEnum):
    S3_PUT_CREATED = "created"
    S3_PUT_ALREADY_THERE = "already_there"
    S3_PUT_CONFLICT = "conflict"


class StorageUnavailable(Exception):
    """S3 failed after the retries, or failed in a way a retry cannot fix: the api answers 503."""


class MissingObject(PermanentError):
    """The event points at an object that is not in the bucket, no retry makes it appear."""


def classify(error: ClientError | BotoCoreError) -> TransientError | PermanentError:
    if isinstance(error, ClientError):
        code = error.response.get("Error", {}).get("Code", "")
        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        if status >= 500 or code in _TRANSIENT_CODES:
            return TransientError(f"s3 said {status} {code}")
        return PermanentError(f"s3 said {status} {code}")
    # connection and timeout errors can clear up, missing credentials or bad parameters cannot
    if isinstance(error, BotoConnectionError | HTTPClientError):
        return TransientError(f"s3 connection problem: {error}")
    return PermanentError(f"s3 client problem: {error}")


def _precondition_failed(error: ClientError) -> bool:
    return error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 412


def _is_missing_object(error: Exception) -> bool:
    # classify wrapped the boto error and the original is the cause, only no such key means
    # the document is missing, a missing bucket or a refused access is a setup problem and
    # not a fault of the message
    cause = error.__cause__
    code = cause.response.get("Error", {}).get("Code") if isinstance(cause, ClientError) else None
    return code == "NoSuchKey"


class DocumentStore:
    def __init__(
        self,
        s3_client: "S3Client",
        s3_bucket: str,
        s3_executor: ThreadPoolExecutor,
        s3_bulkhead: Bulkhead,
        policy: RetryPolicy,
        sleep: Callable[[float], Awaitable[None]],
        rng: random.Random,
    ) -> None:
        self._s3_client = s3_client
        self._s3_bucket = s3_bucket
        self._s3_executor = s3_executor
        self._s3_bulkhead = s3_bulkhead
        self._policy = policy
        self._sleep = sleep
        self._rng = rng

    async def put_if_absent(self, key: str, data: bytes, sha: str) -> S3PutResult:
        loop = asyncio.get_running_loop()
        # one permit covers the call and its retries, so at most the limit of calls are in flight
        # a full bulkhead raises Overloaded here, the api turns that into a 503 as well
        # how two requests run this at the same time, request 1 and request 2
        # request 1 takes a bulkhead permit, then call_with_retries receives the lambda below
        # the lambda is only a recipe, nothing has run yet and no thread is used
        # inside call_with_retries the first operation() call runs the recipe
        # run_in_executor offloads the blocking boto3 call to the thread pool executor right away
        # and returns a future, a placeholder for the answer the thread will produce
        # request 1 awaits that future and yields to the event loop while its thread talks to s3
        # request 2 now runs, takes its own permit and does the same with another thread
        # when the thread of request 1 returns S3_PUT_CREATED the loop completes the future
        # and request 1 wakes up exactly where it awaited
        async with self._s3_bulkhead:
            try:
                # the lambda delays the call, arguments are evaluated before the function runs
                # without it run_in_executor would start the s3 call before call_with_retries begins
                # the retry loop would then receive a future and not something it can call again
                # a future cannot be rerun, so there would be no retry and no per attempt handling
                return await call_with_retries(
                    lambda: loop.run_in_executor(
                        self._s3_executor, self._put_blocking, key, data, sha
                    ),
                    self._policy,
                    self._sleep,
                    self._rng,
                )
            except Rejected as rejected:
                raise StorageUnavailable(str(rejected)) from rejected.error

    async def get(self, key: str) -> bytes:
        loop = asyncio.get_running_loop()
        # the same bulkhead and retries as a write, one permit covers the call and its retries
        async with self._s3_bulkhead:
            try:
                return await call_with_retries(
                    lambda: loop.run_in_executor(self._s3_executor, self._read_blocking, key),
                    self._policy,
                    self._sleep,
                    self._rng,
                )
            except Rejected as rejected:
                # a document that is not there is the fault of the message and is dead lettered
                # anything else, an outage, a refused access, a missing bucket, is not
                if _is_missing_object(rejected.error):
                    raise MissingObject(key) from rejected.error
                raise StorageUnavailable(str(rejected)) from rejected.error

    def close(self) -> None:
        self._s3_executor.shutdown(wait=False, cancel_futures=True)

    # everything below runs on a worker thread, boto3 calls block
    def _put_blocking(self, key: str, data: bytes, sha: str) -> S3PutResult:
        try:
            self._s3_client.put_object(
                Bucket=self._s3_bucket,
                Key=key,
                Body=data,
                ContentType="application/json",
                IfNoneMatch="*",
            )
            return S3PutResult.S3_PUT_CREATED
        except ClientError as error:
            # 412 is not a failure, it means the object is already there and we compare it below
            if not _precondition_failed(error):
                raise classify(error) from error
        except BotoCoreError as error:
            raise classify(error) from error
        existing = self._read_blocking(key)
        return (
            S3PutResult.S3_PUT_ALREADY_THERE
            if content_sha(existing) == sha
            else S3PutResult.S3_PUT_CONFLICT
        )

    def _read_blocking(self, key: str) -> bytes:
        try:
            return self._s3_client.get_object(Bucket=self._s3_bucket, Key=key)["Body"].read()
        except (ClientError, BotoCoreError) as error:
            raise classify(error) from error


def build_store(
    settings: IngestSettings,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: random.Random | None = None,
) -> DocumentStore:
    client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        region_name=settings.s3_region,
        config=Config(
            connect_timeout=settings.s3_timeout_s,
            read_timeout=settings.s3_timeout_s,
            # our own retry policy is the only one, botocore must not retry on top of it
            # total attempts of one means no retries, max attempts of one would still allow a retry
            retries={"total_max_attempts": 1},
            # one http connection per worker thread, the default of ten would not fit a bigger pool
            max_pool_connections=settings.s3_threads,
        ),
    )
    return DocumentStore(
        s3_client=client,
        s3_bucket=settings.s3_bucket,
        s3_executor=ThreadPoolExecutor(max_workers=settings.s3_threads, thread_name_prefix="s3"),
        s3_bulkhead=Bulkhead(settings.s3_bulkhead_limit, settings.s3_bulkhead_wait_s),
        policy=RetryPolicy(),
        sleep=sleep,
        # the rng only spreads retry waits, it protects nothing, so a plain generator is fine
        rng=rng or random.Random(),  # noqa: S311
    )
