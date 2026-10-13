"""The outbox relay against real Postgres (through PgBouncer) and real Kafka (Redpanda)."""

import asyncio
import json
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from typing import Any

import psycopg
import pytest
from aiokafka.errors import KafkaTimeoutError
from psycopg_pool import AsyncConnectionPool

from libs.schemas import UpsertEvent, doc_event_adapter
from libs.schemas.hashing import content_sha
from services.ingest.relay import KafkaPublisher, OutboxMessage, Relay, kafka_publisher, serve
from services.ingest.settings import IngestSettings

from .conftest import Api, Database, add_events, read_topic

BATCH = 100


class Quiet:
    """A sleep that records the wait and gives the other tasks a turn, without really waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)
        await asyncio.sleep(0.01)


@pytest.fixture
async def publisher(topic: str) -> AsyncIterator[KafkaPublisher]:
    async with kafka_publisher(IngestSettings(topic_events=topic)) as real:
        yield real


def make_relay(pool: AsyncConnectionPool, publisher: Any, batch_size: int = BATCH) -> Relay:
    return Relay(pool, publisher, batch_size=batch_size, poll_interval_s=0.1, sleep=Quiet())


def published_flags(dsn: str) -> list[bool]:
    with psycopg.connect(dsn) as conn:
        rows = conn.execute("select published_at is not null from outbox order by id").fetchall()
    return [flag for (flag,) in rows]


# the happy path


async def test_rows_reach_kafka_with_key_value_and_headers_and_are_marked_published(
    pool: AsyncConnectionPool, db: Database, topic: str, publisher: KafkaPublisher
) -> None:
    add_events(db.direct_dsn, 3, "D1")
    relay = make_relay(pool, publisher)
    assert await relay.publish_batch() == 3
    messages = await read_topic(topic, 3)
    assert {m.key for m in messages} == {b"D1"}
    assert [json.loads(m.value)["version"] for m in messages] == [1, 2, 3]
    headers = [dict(m.headers) for m in messages]
    assert [h["request_id"] for h in headers] == [b"req-x-1", b"req-x-2", b"req-x-3"]
    assert {h["schema"] for h in headers} == {b"doc_event_v1"}
    assert published_flags(db.direct_dsn) == [True, True, True]
    assert await relay.publish_batch() == 0
    assert await relay.backlog() == 0


async def test_a_big_backlog_goes_out_in_batches_in_id_order(
    pool: AsyncConnectionPool, db: Database, topic: str, publisher: KafkaPublisher
) -> None:
    add_events(db.direct_dsn, 250, "ORDER")
    relay = make_relay(pool, publisher)
    assert [await relay.publish_batch() for _ in range(4)] == [100, 100, 50, 0]
    messages = await read_topic(topic, 250)
    # one document is one key is one partition, so the offsets show the order they were sent in
    assert [json.loads(m.value)["version"] for m in messages] == list(range(1, 251))


async def test_events_of_one_document_stay_in_order_between_other_documents(
    pool: AsyncConnectionPool, db: Database, topic: str, publisher: KafkaPublisher
) -> None:
    for version in range(1, 5):
        add_events(db.direct_dsn, 1, "D1", first_version=version)
        add_events(db.direct_dsn, 1, "D2", first_version=version)
    await make_relay(pool, publisher).publish_batch()
    messages = await read_topic(topic, 8)
    for key in (b"D1", b"D2"):
        assert [json.loads(m.value)["version"] for m in messages if m.key == key] == [1, 2, 3, 4]


async def test_rows_are_claimed_in_id_order_even_when_a_row_was_rewritten(
    pool: AsyncConnectionPool, db: Database, publisher: KafkaPublisher
) -> None:
    ids = add_events(db.direct_dsn, 5)
    # an update writes a new copy of the row at the end of the table, so the table order is no
    # longer the id order, and only the order by keeps the oldest event first
    with psycopg.connect(db.direct_dsn, autocommit=True) as conn:
        conn.execute("update outbox set request_id = 'rewritten' where id = %s", (ids[0],))
    recording = Recording(publisher)
    await make_relay(pool, recording, batch_size=2).publish_batch()
    assert recording.sent == ids[:2]


async def test_the_backlog_counts_only_unpublished_rows(
    pool: AsyncConnectionPool, db: Database, publisher: KafkaPublisher
) -> None:
    add_events(db.direct_dsn, 5)
    relay = make_relay(pool, publisher, batch_size=2)
    assert await relay.backlog() == 5
    await relay.publish_batch()
    assert await relay.backlog() == 3


# a crash between the send and the mark: duplicates, never a loss


class CrashAfterSend:
    def __init__(self, real: KafkaPublisher) -> None:
        self._real = real

    async def publish(self, messages: Sequence[OutboxMessage]) -> None:
        await self._real.publish(messages)
        raise RuntimeError("the relay died after the send and before the mark")


async def test_a_crash_between_the_send_and_the_mark_means_duplicates_but_no_loss(
    pool: AsyncConnectionPool, db: Database, topic: str, publisher: KafkaPublisher
) -> None:
    add_events(db.direct_dsn, 4)
    with pytest.raises(RuntimeError, match="died after the send"):
        await make_relay(pool, CrashAfterSend(publisher)).publish_batch()
    assert published_flags(db.direct_dsn) == [False] * 4  # the transaction rolled back
    assert await make_relay(pool, publisher).publish_batch() == 4
    messages = await read_topic(topic, 8)
    counts = Counter(json.loads(m.value)["version"] for m in messages)
    assert counts == {1: 2, 2: 2, 3: 2, 4: 2}  # every event arrived, each one twice
    assert published_flags(db.direct_dsn) == [True] * 4


# two relays at once


class HoldInsidePublish:
    """The real publisher, except that publish waits for the test to say go."""

    def __init__(self, real: KafkaPublisher) -> None:
        self._real = real
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.claimed: list[int] = []

    async def publish(self, messages: Sequence[OutboxMessage]) -> None:
        self.claimed = [message.id for message in messages]
        self.entered.set()
        await self.release.wait()
        await self._real.publish(messages)


class Recording:
    def __init__(self, real: KafkaPublisher) -> None:
        self._real = real
        self.sent: list[int] = []

    async def publish(self, messages: Sequence[OutboxMessage]) -> None:
        self.sent = [message.id for message in messages]
        await self._real.publish(messages)


async def test_two_relays_take_different_rows_and_neither_waits_for_the_other(
    pool: AsyncConnectionPool, db: Database, topic: str, publisher: KafkaPublisher
) -> None:
    ids = add_events(db.direct_dsn, 6)
    first = HoldInsidePublish(publisher)
    second = Recording(publisher)
    first_task = asyncio.create_task(make_relay(pool, first, batch_size=3).publish_batch())
    await asyncio.wait_for(first.entered.wait(), timeout=10)
    # the first relay holds three rows open in a transaction, the second must take the others
    assert await asyncio.wait_for(make_relay(pool, second, batch_size=3).publish_batch(), 10) == 3
    assert first.claimed == ids[:3]
    assert second.sent == ids[3:]
    # and a new event arriving meanwhile is not blocked by the relay's locks
    late = add_events(db.direct_dsn, 1, "LATE")
    first.release.set()
    assert await asyncio.wait_for(first_task, 10) == 3
    messages = await read_topic(topic, 6)
    assert len(messages) == 6  # every row went out exactly once, none twice
    assert published_flags(db.direct_dsn) == [True] * 6 + [False]
    assert len(late) == 1
    assert await make_relay(pool, publisher).backlog() == 1


# kafka saying no


class FlakyThenReal:
    def __init__(self, real: KafkaPublisher, failures: int) -> None:
        self._real = real
        self.failures_left = failures

    async def publish(self, messages: Sequence[OutboxMessage]) -> None:
        if self.failures_left > 0:
            self.failures_left -= 1
            raise KafkaTimeoutError("no acknowledgement")
        await self._real.publish(messages)


async def test_while_kafka_fails_the_rows_stay_the_loop_backs_off_and_recovers(
    pool: AsyncConnectionPool, db: Database, topic: str, publisher: KafkaPublisher
) -> None:
    add_events(db.direct_dsn, 5)
    sleeps = Quiet()
    flaky = FlakyThenReal(publisher, failures=3)
    relay = Relay(pool, flaky, batch_size=BATCH, poll_interval_s=0.1, sleep=sleeps)
    stop = asyncio.Event()
    task = asyncio.create_task(relay.run(stop))
    try:
        # the sleeps are the relay's own, so the poll below is the only waiting the test does
        async with asyncio.timeout(20):
            while await relay.backlog() > 0:  # noqa: ASYNC110
                await asyncio.sleep(0.05)
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=10)
    waits = sleeps.delays[:3]  # three failures, three growing waits, each under its ceiling
    assert all(0 <= wait <= 0.1 * 2**n for n, wait in enumerate(waits))
    messages = await read_topic(topic, 5)
    assert len(messages) == 5  # the failures happened before any send, so nothing is duplicated


async def test_the_server_keeps_retrying_when_kafka_is_unreachable_and_stops_when_told(
    pool: AsyncConnectionPool, db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    add_events(db.direct_dsn, 2)
    settings = IngestSettings(
        database_dsn=db.pooled_dsn, kafka_bootstrap="127.0.0.1:9", relay_produce_timeout_s=0.5
    )
    stop = asyncio.Event()
    task = asyncio.create_task(serve(settings, stop, sleep=Quiet()))
    await asyncio.sleep(2.0)
    assert not task.done()  # still alive, trying again and again
    stop.set()
    await asyncio.wait_for(task, timeout=10)  # and it ends without an error
    assert published_flags(db.direct_dsn) == [False, False]
    attempts = [r.fields["attempt"] for r in caplog.records if hasattr(r, "fields")]  # type: ignore[attr-defined]
    # the attempt counter grows, which is what makes the waits between attempts grow
    assert attempts[:3] == [1, 2, 3]


async def test_the_server_publishes_a_new_row_and_shuts_down_cleanly(
    db: Database, topic: str
) -> None:
    settings = IngestSettings(
        database_dsn=db.pooled_dsn, topic_events=topic, relay_poll_interval_s=0.1
    )
    stop = asyncio.Event()
    task = asyncio.create_task(serve(settings, stop))
    try:
        add_events(db.direct_dsn, 1, "SERVED")
        messages = await read_topic(topic, 1)
        assert messages[0].key == b"SERVED"
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=10)
    assert published_flags(db.direct_dsn) == [True]


async def test_the_producer_is_built_for_safe_delivery(publisher: KafkaPublisher) -> None:
    producer = publisher._producer
    # a transaction manager exists only when idempotence is on, and it makes the broker drop
    # a message the producer resent after a lost acknowledgement
    assert producer._txn_manager is not None
    assert producer.client._client_id == "outbox-relay"


# from the api to kafka


async def test_a_write_through_the_api_arrives_on_kafka_matching_what_is_in_s3(
    api: Api, pool: AsyncConnectionPool, topic: str, publisher: KafkaPublisher
) -> None:
    response = await api.post("D1", 3, title="End To End")
    assert response.status_code == 202
    assert await make_relay(pool, publisher).publish_batch() == 1
    (message,) = await read_topic(topic, 1)
    event = doc_event_adapter.validate_json(message.value)
    assert isinstance(event, UpsertEvent)
    assert message.key == b"D1"
    assert dict(message.headers)["request_id"] == response.headers["x-request-id"].encode()
    stored = api.object_bytes(event.s3_key)
    assert (event.content_sha, event.size_bytes) == (content_sha(stored), len(stored))
