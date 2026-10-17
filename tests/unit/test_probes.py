"""The three readiness probes of the worker. A probe returns when healthy and raises when not."""

import pytest
from aiokafka import TopicPartition

from services.embed.probes import KafkaAssignmentProbe, qdrant_probe


class FakeQdrant:
    def __init__(self, error: Exception | None = None) -> None:
        self.asked_for: list[str] = []
        self._error = error

    async def get_collection(self, name: str) -> None:
        self.asked_for.append(name)
        if self._error:
            raise self._error


class FakeConsumer:
    def __init__(self, assignment: set[TopicPartition]) -> None:
        self._assignment = assignment

    def assignment(self) -> set[TopicPartition]:
        return self._assignment


async def test_qdrant_is_ready_when_it_knows_the_collection_of_the_worker() -> None:
    client = FakeQdrant()
    await qdrant_probe(client, "chunks_esci")()  # type: ignore[arg-type]
    assert client.asked_for == ["chunks_esci"]


async def test_qdrant_is_not_ready_when_the_call_fails() -> None:
    probe = qdrant_probe(FakeQdrant(OSError("refused")), "chunks_esci")  # type: ignore[arg-type]
    with pytest.raises(OSError, match="refused"):
        await probe()


async def test_kafka_is_not_ready_before_the_consumer_exists() -> None:
    with pytest.raises(RuntimeError, match="not started"):
        await KafkaAssignmentProbe()()


async def test_kafka_is_not_ready_while_the_group_has_given_no_partition() -> None:
    probe = KafkaAssignmentProbe()
    probe.attach(FakeConsumer(set()))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="no partitions"):
        await probe()


async def test_kafka_is_ready_once_the_group_has_given_partitions() -> None:
    probe = KafkaAssignmentProbe()
    probe.attach(FakeConsumer({TopicPartition("doc.events", 3)}))  # type: ignore[arg-type]
    assert await probe() is None
