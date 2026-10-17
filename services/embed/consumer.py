import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, TopicPartition
from aiokafka.errors import KafkaError
from prometheus_client import CollectorRegistry
from pydantic import ValidationError

from libs.observability.health import Heartbeat
from libs.observability.metrics import EmbedMetrics
from libs.resilience.retry import (
    FailureAction,
    PermanentError,
    Rejected,
    RetryPolicy,
    after_dead_letter,
    backoff_delay,
    call_with_retries,
)
from libs.schemas import DocEvent, doc_event_adapter
from services.embed.pipeline import KafkaEventResult
from services.embed.settings import EmbedSettings

log = logging.getLogger("embed.consumer")

# an error message goes into the dead letter record, so it is cut, the record must stay small
_MAX_ERROR_CHARS = 500


class ConsumedEventResult(StrEnum):
    """How the consumer finished with one message, the values are those of the result label."""

    CONSUMED_EVENT_APPLIED = "applied"
    CONSUMED_EVENT_SKIPPED = "skipped"
    CONSUMED_EVENT_DEAD_LETTERED = "dead_lettered"
    CONSUMED_EVENT_HELD = "held"


class EventHandler(Protocol):
    async def handle(self, event: DocEvent) -> KafkaEventResult: ...


@dataclass(frozen=True)
class DeadLetter:
    key: bytes | None
    value: bytes | None
    headers: list[tuple[str, bytes]]


class DeadLetterSink(Protocol):
    async def publish(self, record: DeadLetter) -> None:
        """Write the record and return, or raise when it could not be written."""
        ...


def dead_letter_for(message: Any, error: Exception, attempts: int) -> DeadLetter:
    # the original bytes go through untouched, so a replay after a fix sends exactly what came in
    # what went wrong rides in headers, and where the message came from, to find it again
    source = f"{message.topic}:{message.partition}:{message.offset}"
    return DeadLetter(
        key=message.key,
        value=message.value,
        headers=[
            *(message.headers or []),
            ("dlq_error_type", type(error).__name__.encode()),
            ("dlq_error", str(error)[:_MAX_ERROR_CHARS].encode()),
            ("dlq_attempts", str(attempts).encode()),
            ("dlq_source", source.encode()),
        ],
    )


class KafkaDeadLetterSink:
    def __init__(self, producer: AIOKafkaProducer, topic: str, timeout_s: float) -> None:
        self._producer = producer
        self._topic = topic
        self._timeout_s = timeout_s

    async def publish(self, record: DeadLetter) -> None:
        # the timeout keeps a produce from hanging forever when the broker never answers
        async with asyncio.timeout(self._timeout_s):
            future = await self._producer.send(
                self._topic, value=record.value, key=record.key, headers=record.headers
            )
            # send only queues the record, the future is the broker saying it is written
            await future


class EventConsumer:
    def __init__(
        self,
        kafka_consumer: AIOKafkaConsumer,
        dlq_sink: DeadLetterSink,
        handler: EventHandler,
        settings: EmbedSettings,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
        heartbeat: Heartbeat | None = None,
        metrics: EmbedMetrics | None = None,
    ) -> None:
        self._kafka_consumer = kafka_consumer
        # where a message that can never be applied is written, a kafka producer that sends it
        # to the dead letter topic, the topic name is a setting and it lives inside the sink
        self._dlq_sink = dlq_sink
        self._handler = handler
        self._poll_ms = settings.consumer_poll_ms
        self._sleep = sleep
        # the rng only spreads waits, it protects nothing, so a plain generator is fine
        self._rng = rng or random.Random()  # noqa: S311
        self._clock = clock
        # the loop beats every turn, the health endpoint reads the age of the last beat
        self._heartbeat = heartbeat or Heartbeat(clock)
        self._metrics = metrics or EmbedMetrics(CollectorRegistry())
        # when each partition was first held for the message it is stuck on, it leaves here only
        # when that message goes through, so its age is the whole time it has been stuck and not
        # the time since the latest retry
        self._held_since: dict[TopicPartition, float] = {}
        # tries for one message before it is given up on
        self._attempt_policy = RetryPolicy(max_attempts=settings.message_attempts)
        # the wait before a held partition is tried again, only its delays are used
        self._hold_policy = RetryPolicy(
            max_attempts=1,
            base_delay=settings.stall_base_delay_s,
            max_delay=settings.stall_max_delay_s,
        )
        # when each held partition may be tried again, a partition is in here only while held
        self._resume_at: dict[TopicPartition, float] = {}
        # how many times in a row each partition was held, it makes the wait grow, and a
        # message that goes through clears it
        self._times_held_in_a_row: dict[TopicPartition, int] = {}

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            self._heartbeat.beat()
            self._resume_the_partitions_that_are_due()
            self._report_holds()
            # one message at a time, and a poll that finds nothing returns after the timeout, so
            # the loop keeps polling while partitions are held, which keeps the group membership
            # alive and gives the timers and the stop signal their turn
            batch = await self._kafka_consumer.getmany(timeout_ms=self._poll_ms, max_records=1)
            for partition, messages in batch.items():
                for message in messages:
                    await self._process(partition, message)

    async def _process(self, partition: TopicPartition, message: Any) -> None:
        started = self._clock()
        result = await self._handle(message)
        self._metrics.event_seconds.observe(self._clock() - started)
        self._metrics.events.labels(result=result.value).inc()
        if result is ConsumedEventResult.CONSUMED_EVENT_HELD:
            self._hold(partition, message)
        else:
            await self._commit(partition, message.offset + 1)
            self._times_held_in_a_row.pop(partition, None)
            self._held_since.pop(partition, None)

    async def _handle(self, message: Any) -> ConsumedEventResult:
        """Finished (applied, skipped, safely dead lettered) or held, which leaves the offset."""
        where = {"topic": message.topic, "partition": message.partition, "offset": message.offset}
        if message.value is None:
            return await self._dead_letter(message, PermanentError("an event without a body"), 0)
        try:
            event = doc_event_adapter.validate_json(message.value)
        except ValidationError as error:
            # the text of the error is not kept, it could echo what was sent
            problem = PermanentError(f"not a valid event, {error.error_count()} problem(s)")
            return await self._dead_letter(message, problem, 0)
        try:
            result = await call_with_retries(
                lambda: self._handler.handle(event),
                self._attempt_policy,
                self._sleep,
                self._rng,
            )
        except Rejected as rejected:
            if rejected.action is FailureAction.FAILURE_DEAD_LETTER:
                return await self._dead_letter(message, rejected.error, rejected.attempts)
            # the message is fine, the system around it is not, so it is held and not moved anywhere
            log.warning(
                "message held",
                extra={
                    "fields": {
                        **where,
                        "action": rejected.action.value,
                        "error": repr(rejected.error),
                    }
                },
            )
            return ConsumedEventResult.CONSUMED_EVENT_HELD
        log.info("message handled", extra={"fields": {**where, "result": result.value}})
        if result is KafkaEventResult.KAFKA_EVENT_APPLIED:
            return ConsumedEventResult.CONSUMED_EVENT_APPLIED
        return ConsumedEventResult.CONSUMED_EVENT_SKIPPED

    async def _dead_letter(
        self, message: Any, error: Exception, attempts: int
    ) -> ConsumedEventResult:
        try:
            await self._dlq_sink.publish(dead_letter_for(message, error, attempts))
        except Exception:
            # the record is not written, so the offset must not move, the message is never dropped
            log.exception("dead letter record not written, the message is held")
            return self._settled(after_dead_letter(published=False))
        log.warning(
            "message dead lettered",
            extra={"fields": {"offset": message.offset, "error": type(error).__name__}},
        )
        self._metrics.dead_letters.labels(error_type=type(error).__name__).inc()
        return self._settled(after_dead_letter(published=True))

    @staticmethod
    def _settled(action: FailureAction) -> ConsumedEventResult:
        if action is FailureAction.FAILURE_COMMIT_OFFSET:
            return ConsumedEventResult.CONSUMED_EVENT_DEAD_LETTERED
        return ConsumedEventResult.CONSUMED_EVENT_HELD

    async def _commit(self, partition: TopicPartition, offset: int) -> None:
        try:
            await self._kafka_consumer.commit({partition: offset})
        except KafkaError:
            # a rebalance can take the partition away, the message comes back to whoever has it
            # next, and handling it again is safe, so this is logged and not fatal
            log.exception("offset not committed, the message may come back")

    def _hold(self, partition: TopicPartition, message: Any) -> None:
        # stop fetching this partition and go back to the message, so it is the next one when
        # the partition resumes, the other partitions keep going and the order of this one stays
        self._kafka_consumer.pause(partition)
        self._kafka_consumer.seek(partition, message.offset)
        self._held_since.setdefault(partition, self._clock())
        self._times_held_in_a_row[partition] = self._times_held_in_a_row.get(partition, 0) + 1
        wait = backoff_delay(self._times_held_in_a_row[partition], self._hold_policy, self._rng)
        self._resume_at[partition] = self._clock() + wait

    def _report_holds(self) -> None:
        # set on every turn, so the gauges are at most one poll interval old when prometheus reads
        now = self._clock()
        self._metrics.held_partitions.set(len(self._held_since))
        self._metrics.oldest_hold_age.set(
            max((now - since for since in self._held_since.values()), default=0.0)
        )

    def _resume_the_partitions_that_are_due(self) -> None:
        # not a timer: it runs once per turn of the loop, and a turn lasts at most one poll
        # interval, so a held partition is let go within a poll interval of its wait ending
        # resuming only lets the partition be fetched again, the held message then comes back
        # through the poll like any other and goes through the normal processing
        now = self._clock()
        for partition in [p for p, until in self._resume_at.items() if until <= now]:
            self._kafka_consumer.resume(partition)
            del self._resume_at[partition]


@asynccontextmanager
async def kafka_consumer(settings: EmbedSettings) -> AsyncIterator[AIOKafkaConsumer]:
    consumer = AIOKafkaConsumer(
        settings.topic_events,
        bootstrap_servers=settings.kafka_bootstrap,
        group_id=settings.consumer_group,
        client_id="embed-worker",
        # offsets move only when a message is finished, never on a timer
        enable_auto_commit=False,
        # a group that has never committed starts from the first message
        auto_offset_reset="earliest",
    )
    await consumer.start()
    try:
        yield consumer
    finally:
        await consumer.stop()


@asynccontextmanager
async def kafka_dead_letter_sink(settings: EmbedSettings) -> AsyncIterator[KafkaDeadLetterSink]:
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap,
        acks="all",
        enable_idempotence=True,
        client_id="embed-worker-dlq",
        request_timeout_ms=int(settings.dlq_produce_timeout_s * 1000),
    )
    await producer.start()
    try:
        yield KafkaDeadLetterSink(producer, settings.topic_dlq, settings.dlq_produce_timeout_s)
    finally:
        await producer.stop()
