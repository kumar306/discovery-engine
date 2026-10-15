"""How long a held partition waits, and when it is let go. A stand in for the kafka client records
what the loop asked of it, the timing is the loop's own and does not need a broker to be checked.
"""

import random
from types import SimpleNamespace
from typing import Any

from aiokafka import TopicPartition

from services.embed.consumer import EventConsumer
from services.embed.settings import EmbedSettings

PARTITION = TopicPartition("doc.events", 0)


class MaxRandom(random.Random):
    """Always draws the top of the range, so the ceiling of each wait can be asserted."""

    def uniform(self, a: float, b: float) -> float:
        return b


class RecordingKafka:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def pause(self, partition: TopicPartition) -> None:
        self.calls.append(("pause", partition.partition))

    def resume(self, partition: TopicPartition) -> None:
        self.calls.append(("resume", partition.partition))

    def seek(self, partition: TopicPartition, offset: int) -> None:
        self.calls.append(("seek", offset))


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def make(kafka: RecordingKafka, clock: Clock) -> EventConsumer:
    settings = EmbedSettings(stall_base_delay_s=1.0, stall_max_delay_s=5.0)
    return EventConsumer(
        kafka,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        settings,
        rng=MaxRandom(),
        clock=clock,
    )


def message(offset: int) -> Any:
    return SimpleNamespace(offset=offset)


def test_a_held_partition_is_paused_and_taken_back_to_the_message() -> None:
    kafka, clock = RecordingKafka(), Clock()
    make(kafka, clock)._hold(PARTITION, message(7))
    assert kafka.calls == [("pause", 0), ("seek", 7)]


def test_a_held_partition_is_resumed_only_when_its_wait_is_over() -> None:
    kafka, clock = RecordingKafka(), Clock()
    consumer = make(kafka, clock)
    consumer._hold(PARTITION, message(7))  # the first wait is 1 s
    clock.now = 100.9
    consumer._resume_the_partitions_that_are_due()
    assert ("resume", 0) not in kafka.calls
    clock.now = 101.0
    consumer._resume_the_partitions_that_are_due()
    assert kafka.calls[-1] == ("resume", 0)
    consumer._resume_the_partitions_that_are_due()
    assert kafka.calls.count(("resume", 0)) == 1  # and only once


def test_the_wait_doubles_each_time_the_same_partition_is_held_again_up_to_the_cap() -> None:
    kafka, clock = RecordingKafka(), Clock()
    consumer = make(kafka, clock)
    waits = []
    for _ in range(5):
        consumer._hold(PARTITION, message(7))
        waits.append(consumer._resume_at[PARTITION] - clock.now)
        clock.now += 100  # long enough for the wait to be over
        consumer._resume_the_partitions_that_are_due()
    assert waits == [1.0, 2.0, 4.0, 5.0, 5.0]


def test_two_partitions_wait_on_their_own_clocks() -> None:
    kafka, clock = RecordingKafka(), Clock()
    consumer = make(kafka, clock)
    other = TopicPartition("doc.events", 1)
    consumer._hold(PARTITION, message(7))
    clock.now = 100.5
    consumer._hold(other, message(3))
    clock.now = 101.0
    consumer._resume_the_partitions_that_are_due()
    assert [c for c in kafka.calls if c[0] == "resume"] == [("resume", 0)]  # only the first is due
