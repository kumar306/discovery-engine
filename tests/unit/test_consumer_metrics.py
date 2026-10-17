"""What the consumer loop counts and the heartbeat it gives. No broker, a stand in client."""

import asyncio
import random
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from aiokafka import TopicPartition
from prometheus_client import CollectorRegistry

from libs.observability.health import Heartbeat
from libs.observability.metrics import EmbedMetrics
from libs.schemas import DocEvent
from services.embed.consumer import DeadLetter, EventConsumer
from services.embed.pipeline import KafkaEventResult
from services.embed.settings import EmbedSettings
from tests.support.events import upsert_event_bytes

APPLIED = KafkaEventResult.KAFKA_EVENT_APPLIED
SKIPPED = KafkaEventResult.KAFKA_EVENT_SKIPPED
P0 = TopicPartition("doc.events", 0)
P1 = TopicPartition("doc.events", 1)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class FakeKafka:
    """Records what the loop asks of the client and serves scripted batches."""

    def __init__(self, batches: list[dict[TopicPartition, list[Any]]] | None = None) -> None:
        self.batches = list(batches or [])
        self.committed: list[dict[TopicPartition, int]] = []
        self.paused: list[int] = []
        self.on_poll: Callable[[], None] = lambda: None
        self.stop = asyncio.Event()

    async def getmany(self, timeout_ms: int, max_records: int) -> dict[TopicPartition, list[Any]]:
        self.on_poll()
        if len(self.batches) <= 1:
            self.stop.set()  # the last turn
        return self.batches.pop(0) if self.batches else {}

    async def commit(self, offsets: dict[TopicPartition, int]) -> None:
        self.committed.append(offsets)

    def pause(self, partition: TopicPartition) -> None:
        self.paused.append(partition.partition)

    def resume(self, partition: TopicPartition) -> None:
        pass

    def seek(self, partition: TopicPartition, offset: int) -> None:
        pass


class Handler:
    """Each call takes the next scripted outcome: a result is returned, an exception is raised."""

    def __init__(self, *script: KafkaEventResult | Exception) -> None:
        self.script = list(script)

    async def handle(self, event: DocEvent) -> KafkaEventResult:
        outcome = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Sink:
    def __init__(self, fail: bool = False) -> None:
        self.records: list[DeadLetter] = []
        self._fail = fail

    async def publish(self, record: DeadLetter) -> None:
        if self._fail:
            raise ConnectionError("the broker did not answer")
        self.records.append(record)


async def no_sleep(seconds: float) -> None:
    return None


def message(offset: int, value: bytes | None = None, partition: int = 0) -> Any:
    return SimpleNamespace(
        topic="doc.events",
        partition=partition,
        offset=offset,
        key=b"D1",
        value=upsert_event_bytes("D1", offset + 1) if value is None else value,
        headers=[],
    )


class Rig:
    def __init__(self, handler: Handler, sink: Sink | None = None) -> None:
        self.clock = Clock()
        self.kafka = FakeKafka()
        self.registry = CollectorRegistry()
        self.heartbeat = Heartbeat(self.clock)
        self.sink = sink or Sink()
        self.consumer = EventConsumer(
            self.kafka,  # type: ignore[arg-type]
            self.sink,
            handler,
            EmbedSettings(message_attempts=2, stall_base_delay_s=1.0, stall_max_delay_s=5.0),
            sleep=no_sleep,
            rng=random.Random(1),
            clock=self.clock,
            heartbeat=self.heartbeat,
            metrics=EmbedMetrics(self.registry),
        )

    def value(self, name: str, **labels: str) -> float | None:
        return self.registry.get_sample_value(name, labels)

    def gauges(self) -> tuple[float | None, float | None]:
        self.consumer._report_holds()
        return self.value("embed_held_partitions"), self.value("embed_oldest_hold_age_seconds")


@pytest.fixture
def down() -> Exception:
    return RuntimeError("qdrant is not answering")


# what is counted


async def test_applied_and_skipped_events_are_counted_and_timed() -> None:
    # two applied and one skipped, so a swap of the two labels cannot look the same
    rig = Rig(Handler(APPLIED, APPLIED, SKIPPED))
    for offset in range(3):
        await rig.consumer._process(P0, message(offset))
    assert rig.value("embed_events_total", result="applied") == 2
    assert rig.value("embed_events_total", result="skipped") == 1
    assert rig.value("embed_event_seconds_count") == 3
    assert rig.kafka.committed == [{P0: 1}, {P0: 2}, {P0: 3}]


async def test_an_event_that_cannot_be_applied_is_dead_lettered_with_its_error_type() -> None:
    rig = Rig(Handler(APPLIED))
    await rig.consumer._process(P0, message(0, value=b"this is not json"))
    assert rig.value("embed_events_total", result="dead_lettered") == 1
    assert rig.value("embed_dead_letters_total", error_type="PermanentError") == 1
    assert len(rig.sink.records) == 1


async def test_a_dead_letter_that_could_not_be_written_is_a_hold_and_not_a_dead_letter() -> None:
    rig = Rig(Handler(APPLIED), Sink(fail=True))
    await rig.consumer._process(P0, message(0, value=b"this is not json"))
    assert rig.value("embed_events_total", result="held") == 1
    assert rig.value("embed_events_total", result="dead_lettered") in (None, 0)
    assert rig.value("embed_dead_letters_total", error_type="PermanentError") in (None, 0)
    assert rig.kafka.committed == []


async def test_a_message_that_keeps_failing_is_counted_as_held_and_not_committed(
    down: Exception,
) -> None:
    rig = Rig(Handler(down))
    await rig.consumer._process(P0, message(0))
    assert rig.value("embed_events_total", result="held") == 1
    assert rig.kafka.committed == []
    assert rig.kafka.paused == [0]


# the held partition gauges, the alert is written against these


async def test_nothing_held_reads_zero() -> None:
    rig = Rig(Handler(APPLIED))
    assert rig.gauges() == (0, 0)


async def test_a_held_partition_is_counted_and_its_age_grows_with_time(down: Exception) -> None:
    rig = Rig(Handler(down))
    await rig.consumer._process(P0, message(0))
    assert rig.gauges() == (1, 0)
    rig.clock.now += 40
    assert rig.gauges() == (1, 40)


async def test_the_age_counts_from_the_first_hold_not_from_the_latest_retry(
    down: Exception,
) -> None:
    rig = Rig(Handler(down))
    await rig.consumer._process(P0, message(0))  # held at t=100
    rig.clock.now += 30
    await rig.consumer._process(P0, message(0))  # tried again, failed again, held again at t=130
    rig.clock.now += 10
    assert rig.gauges() == (1, 40)  # stuck for 40 seconds, not 10


async def test_the_gauges_return_to_zero_when_the_message_finally_goes_through(
    down: Exception,
) -> None:
    handler = Handler(down, down, APPLIED)
    rig = Rig(handler)
    await rig.consumer._process(P0, message(0))
    rig.clock.now += 50
    assert rig.gauges() == (1, 50)
    handler.script = [APPLIED]
    await rig.consumer._process(P0, message(0))
    assert rig.gauges() == (0, 0)


async def test_with_two_held_partitions_the_oldest_hold_is_reported(down: Exception) -> None:
    rig = Rig(Handler(down))
    await rig.consumer._process(P0, message(0, partition=0))
    rig.clock.now += 25
    await rig.consumer._process(P1, message(0, partition=1))
    rig.clock.now += 5
    assert rig.gauges() == (2, 30)


# the heartbeat


async def test_the_loop_beats_on_every_turn() -> None:
    rig = Rig(Handler(APPLIED))
    rig.kafka.batches = [{}, {}, {}]
    ages: list[float] = []

    def look_at_the_heartbeat_and_let_time_pass() -> None:
        ages.append(rig.heartbeat.age())  # this is the moment just after the loop beat
        rig.clock.now += 10

    rig.kafka.on_poll = look_at_the_heartbeat_and_let_time_pass
    rig.clock.now += 500  # a long time since the heartbeat was created
    await rig.consumer.run(rig.kafka.stop)
    assert ages == [0, 0, 0]


async def test_a_loop_that_is_stuck_in_a_message_does_not_beat() -> None:
    rig = Rig(Handler(APPLIED))
    rig.clock.now += 200
    assert rig.heartbeat.age() == 200  # nobody beat, the age is the time since the start
