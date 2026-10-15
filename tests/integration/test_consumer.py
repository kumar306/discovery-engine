"""The consumer loop against real Redpanda: offsets, the dead letter queue, and held partitions.

The pipeline underneath is the real one on the real stores, or, where a test needs a specific
failure at a specific moment, a handler that follows a script. Every test has its own topics and
its own consumer group, so nothing leaks from one test into another.
"""

import asyncio
import random
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import pytest
from aiokafka.errors import KafkaError

from libs.schemas import DocEvent, UpsertEvent
from services.embed.consumer import (
    DeadLetter,
    DeadLetterSink,
    EventConsumer,
    EventHandler,
    kafka_consumer,
    kafka_dead_letter_sink,
)
from services.embed.embedder import EmbedTransient
from services.embed.pipeline import KafkaEventResult
from services.embed.settings import EmbedSettings
from tests.support.crash import SimulatedCrash

from .conftest import NOW, Rig, committed_offsets, eventually, produce, read_topic

APPLIED = KafkaEventResult.KAFKA_EVENT_APPLIED


def settings_for(topic: str, dlq_topic: str) -> EmbedSettings:
    return EmbedSettings(
        topic_events=topic,
        topic_dlq=dlq_topic,
        consumer_group=f"test-{uuid.uuid4().hex[:10]}",
        consumer_poll_ms=100,
        stall_base_delay_s=0.05,
        stall_max_delay_s=0.2,
    )


async def no_wait(seconds: float) -> None:
    # the retries between attempts do not need to take real time in a test
    await asyncio.sleep(0)


@dataclass
class Worker:
    consumer: EventConsumer
    task: "asyncio.Task[None]"
    stop: asyncio.Event
    settings: EmbedSettings

    async def committed(self) -> dict[int, int]:
        return await committed_offsets(self.settings.consumer_group)

    async def finish(self) -> None:
        self.stop.set()
        await asyncio.wait_for(self.task, timeout=15)


@asynccontextmanager
async def running_worker(
    handler: EventHandler,
    settings: EmbedSettings,
    wrap_sink: Callable[[DeadLetterSink], DeadLetterSink] | None = None,
) -> AsyncIterator[Worker]:
    stop = asyncio.Event()
    async with kafka_consumer(settings) as consumer, kafka_dead_letter_sink(settings) as sink:
        event_consumer = EventConsumer(
            consumer,
            wrap_sink(sink) if wrap_sink else sink,
            handler,
            settings,
            sleep=no_wait,
            rng=random.Random(1),
        )
        task = asyncio.create_task(event_consumer.run(stop))
        worker = Worker(event_consumer, task, stop, settings)
        try:
            yield worker
        finally:
            stop.set()
            if not task.done():
                await asyncio.wait_for(task, timeout=15)


class Scripted:
    """A handler that records what it was asked and behaves as the test says."""

    def __init__(self, behave: Callable[[DocEvent, int], Awaitable[KafkaEventResult]]) -> None:
        self.calls: list[tuple[str, int]] = []
        self._behave = behave

    async def handle(self, event: DocEvent) -> KafkaEventResult:
        self.calls.append((event.doc_id, event.version))
        return await self._behave(event, len(self.calls))


async def always_applied(event: DocEvent, call: int) -> KafkaEventResult:
    return APPLIED


def event_bytes(doc_id: str, version: int) -> bytes:
    """A valid event for a handler that never reads the document, so no s3 object is needed."""
    event = UpsertEvent(
        schema_version=1,
        corpus="esci",
        doc_id=doc_id,
        version=version,
        produced_at=NOW,
        op="upsert",
        s3_key=f"esci/{doc_id}/v{version}.json",
        content_sha="a" * 64,
        size_bytes=1,
    )
    return event.model_dump_json().encode()


def headers_of(message: Any) -> dict[str, bytes]:
    return dict(message.headers)


# the normal road


async def test_a_message_is_applied_and_its_offset_is_committed_after_it(
    rig: Rig, topic: str, dlq_topic: str
) -> None:
    event = await rig.ingest("W1", 1)
    await produce(topic, event.model_dump_json().encode(), b"W1", partition=0)
    async with running_worker(rig.pipeline, settings_for(topic, dlq_topic)) as worker:
        await eventually(lambda: _is({0: 1}, worker))
    assert rig.row("W1")["indexed"] == 1
    assert {p["state"] for p in (await rig.points("W1")).values()} == {"live"}
    assert await read_topic(dlq_topic, 0) == []


async def _is(expected: dict[int, int], worker: Worker) -> bool:
    return await worker.committed() == expected


async def test_messages_of_one_partition_are_applied_in_order(
    rig: Rig, topic: str, dlq_topic: str
) -> None:
    first = await rig.ingest("W1", 1)
    second = await rig.ingest("W1", 2, description="a changed description")
    for event in (first, second):
        await produce(topic, event.model_dump_json().encode(), b"W1", partition=0)
    async with running_worker(rig.pipeline, settings_for(topic, dlq_topic)) as worker:
        await eventually(lambda: _is({0: 2}, worker))
    assert rig.row("W1")["indexed"] == 2
    assert {p["version"] for p in (await rig.points("W1")).values() if p["state"] == "live"} == {2}


async def test_an_older_message_arriving_late_is_skipped_and_still_committed(
    rig: Rig, topic: str, dlq_topic: str
) -> None:
    first = await rig.ingest("W1", 1)
    second = await rig.ingest("W1", 2, description="a changed description")
    for event in (second, first):  # the newer one arrives first
        await produce(topic, event.model_dump_json().encode(), b"W1", partition=0)
    async with running_worker(rig.pipeline, settings_for(topic, dlq_topic)) as worker:
        await eventually(lambda: _is({0: 2}, worker))
    assert rig.row("W1")["indexed"] == 2  # a skip is a normal answer, so nothing went to the queue
    assert await read_topic(dlq_topic, 0) == []


# messages that can never be applied go to the dead letter queue, and the loop moves on


async def test_a_message_that_is_not_an_event_is_dead_lettered_and_the_next_one_still_runs(
    rig: Rig, topic: str, dlq_topic: str
) -> None:
    good = await rig.ingest("W1", 1)
    await produce(topic, b"this is not json", b"W0", partition=0)
    await produce(topic, good.model_dump_json().encode(), b"W1", partition=0)
    async with running_worker(rig.pipeline, settings_for(topic, dlq_topic)) as worker:
        await eventually(lambda: _is({0: 2}, worker))
        group = worker.settings.consumer_group
    (record,) = await read_topic(dlq_topic, 1)
    assert record.value == b"this is not json"  # the original bytes, untouched
    assert record.key == b"W0"
    headers = headers_of(record)
    assert headers["dlq_error_type"] == b"PermanentError"
    assert headers["dlq_attempts"] == b"0"
    assert headers["dlq_source"] == f"{topic}:0:0".encode()
    assert rig.row("W1")["indexed"] == 1
    assert group


async def test_a_message_without_a_body_is_dead_lettered(
    rig: Rig, topic: str, dlq_topic: str
) -> None:
    await produce(topic, None, b"W0", partition=0)
    async with running_worker(rig.pipeline, settings_for(topic, dlq_topic)) as worker:
        await eventually(lambda: _is({0: 1}, worker))
    (record,) = await read_topic(dlq_topic, 1)
    assert record.value is None
    # the reason is what a person reads when replaying, so it is pinned
    assert headers_of(record)["dlq_error"] == b"an event without a body"


async def test_an_event_whose_document_does_not_match_is_dead_lettered_with_the_reason(
    rig: Rig, topic: str, dlq_topic: str
) -> None:
    event = await rig.ingest("W1", 1)
    assert isinstance(event, UpsertEvent)
    tampered = event.model_copy(update={"content_sha": "0" * 64})
    await produce(topic, tampered.model_dump_json().encode(), b"W1", partition=0)
    async with running_worker(rig.pipeline, settings_for(topic, dlq_topic)) as worker:
        await eventually(lambda: _is({0: 1}, worker))
    (record,) = await read_topic(dlq_topic, 1)
    headers = headers_of(record)
    assert headers["dlq_attempts"] == b"1"  # a permanent error is not tried again
    assert b"sha" in headers["dlq_error"]
    assert rig.row("W1")["current"] is None  # nothing was committed for it


# trouble that is not the message's fault is retried, then the partition is held, and the
# message is never dead lettered for it


async def test_a_transient_error_is_retried_and_the_message_then_goes_through(
    topic: str, dlq_topic: str
) -> None:
    async def fails_twice(event: DocEvent, call: int) -> KafkaEventResult:
        if call <= 2:
            raise EmbedTransient("the machine is short of memory")
        return APPLIED

    handler = Scripted(fails_twice)
    await produce(topic, event_bytes("W1", 1), b"W1", partition=0)
    async with running_worker(handler, settings_for(topic, dlq_topic)) as worker:
        await eventually(lambda: _is({0: 1}, worker))
    assert len(handler.calls) == 3
    assert await read_topic(dlq_topic, 0) == []


async def test_a_partition_that_keeps_failing_is_held_while_the_others_carry_on(
    topic: str, dlq_topic: str
) -> None:
    recovered = asyncio.Event()

    async def partition_zero_is_down_until_it_recovers(
        event: DocEvent, call: int
    ) -> KafkaEventResult:
        if event.doc_id == "A" and not recovered.is_set():
            raise EmbedTransient("qdrant is not answering")
        return APPLIED

    handler = Scripted(partition_zero_is_down_until_it_recovers)
    await produce(topic, event_bytes("A", 1), b"A", partition=0)
    await produce(topic, event_bytes("A", 2), b"A", partition=0)
    await produce(topic, event_bytes("B", 1), b"B", partition=1)
    async with running_worker(handler, settings_for(topic, dlq_topic)) as worker:
        await eventually(lambda: _has({1: 1}, worker))  # the healthy partition is not blocked
        await asyncio.sleep(1.0)  # and the failing one is tried again and again, not committed
        assert 0 not in await worker.committed()
        assert [c for c in handler.calls if c[0] == "A"].count(("A", 1)) > 3
        assert ("A", 2) not in handler.calls  # the message behind it waits for it
        recovered.set()
        await eventually(lambda: _is({0: 2, 1: 1}, worker))
    # it was never given up on, so nothing was moved, and the order of the partition held
    assert await read_topic(dlq_topic, 0) == []
    a_calls = [c for c in handler.calls if c[0] == "A"]
    assert a_calls.index(("A", 2)) > max(i for i, c in enumerate(a_calls) if c == ("A", 1))


async def _has(expected: dict[int, int], worker: Worker) -> bool:
    committed = await worker.committed()
    return all(committed.get(p) == o for p, o in expected.items())


# the dead letter record is written before the offset moves, so nothing is ever dropped


class FlakySink:
    def __init__(self, real: DeadLetterSink, failures: int) -> None:
        self._real = real
        self.failures_left = failures

    async def publish(self, record: DeadLetter) -> None:
        if self.failures_left > 0:
            self.failures_left -= 1
            raise KafkaError("the broker did not answer")
        await self._real.publish(record)


async def test_when_the_dead_letter_record_cannot_be_written_the_message_is_held_not_dropped(
    topic: str, dlq_topic: str
) -> None:
    flaky: list[FlakySink] = []

    def wrap(real: DeadLetterSink) -> DeadLetterSink:
        flaky.append(FlakySink(real, failures=3))
        return flaky[0]

    await produce(topic, b"not json", b"W0", partition=0)
    handler = Scripted(always_applied)
    async with running_worker(handler, settings_for(topic, dlq_topic), wrap) as worker:
        await asyncio.sleep(0.4)
        assert await worker.committed() == {}  # not committed while the record is not written
        await eventually(lambda: _is({0: 1}, worker))
    records = await read_topic(dlq_topic, 1)
    assert len(records) == 1  # written once, after the sink recovered
    assert flaky[0].failures_left == 0


# at least once: a worker that dies before the commit leaves the message for the next one


async def test_a_worker_that_dies_mid_message_leaves_it_for_the_next_worker(
    rig: Rig, topic: str, dlq_topic: str
) -> None:
    event = await rig.ingest("W1", 1)
    await produce(topic, event.model_dump_json().encode(), b"W1", partition=0)
    settings = settings_for(topic, dlq_topic)

    async def dies(event: DocEvent, call: int) -> KafkaEventResult:
        raise SimulatedCrash("the process died inside the pipeline")

    with pytest.raises(SimulatedCrash):
        async with running_worker(Scripted(dies), settings) as worker:
            await asyncio.wait_for(worker.task, timeout=20)
    assert await committed_offsets(settings.consumer_group) == {}

    # a new worker of the same group takes over and finishes the job
    async with running_worker(rig.pipeline, settings) as worker:
        await eventually(lambda: _is({0: 1}, worker))
    assert rig.row("W1")["indexed"] == 1


async def test_a_worker_started_again_does_not_handle_what_was_already_committed(
    topic: str, dlq_topic: str
) -> None:
    settings = settings_for(topic, dlq_topic)
    await produce(topic, event_bytes("W1", 1), b"W1", partition=0)
    await produce(topic, event_bytes("W1", 2), b"W1", partition=0)
    async with running_worker(Scripted(always_applied), settings) as worker:
        await eventually(lambda: _is({0: 2}, worker))
    second = Scripted(always_applied)
    async with running_worker(second, settings):
        await asyncio.sleep(1.5)
    assert second.calls == []


async def test_a_stop_while_idle_ends_the_loop_promptly(topic: str, dlq_topic: str) -> None:
    async with running_worker(Scripted(always_applied), settings_for(topic, dlq_topic)) as worker:
        await asyncio.sleep(0.5)
        started = asyncio.get_running_loop().time()
        await worker.finish()
    assert asyncio.get_running_loop().time() - started < 3
