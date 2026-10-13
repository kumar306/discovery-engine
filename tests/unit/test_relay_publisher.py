"""KafkaPublisher against a fake producer: what it sends, in which order, and how it fails."""

import asyncio
import time
from typing import Any

import pytest
from aiokafka.errors import KafkaTimeoutError

from services.ingest.relay import KafkaPublisher, OutboxMessage


def message(number: int) -> OutboxMessage:
    return OutboxMessage(
        id=number, key=f"D{number}", request_id=f"req-{number}", schema="doc_event_v1", value="{}"
    )


class FakeProducer:
    """send hands back a future like the real one, and each test decides how that future ends."""

    def __init__(self, outcomes: dict[int, Any] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.sent: list[dict[str, Any]] = []

    async def send(
        self, topic: str, value: bytes, key: bytes, headers: list[tuple[str, bytes]]
    ) -> asyncio.Future[Any]:
        self.sent.append({"topic": topic, "value": value, "key": key, "headers": headers})
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        outcome = self.outcomes.get(len(self.sent))
        if outcome == "never":
            return future  # the broker never answers
        if isinstance(outcome, BaseException):
            future.set_exception(outcome)
        else:
            future.set_result(("partition", len(self.sent)))
        return future


def publisher(producer: FakeProducer, timeout_s: float = 1.0) -> KafkaPublisher:
    return KafkaPublisher(producer, "doc.events", timeout_s)  # type: ignore[arg-type]


async def test_every_message_is_sent_in_order_with_its_key_value_and_headers() -> None:
    producer = FakeProducer()
    await publisher(producer).publish([message(1), message(2), message(3)])
    assert [sent["key"] for sent in producer.sent] == [b"D1", b"D2", b"D3"]
    assert {sent["topic"] for sent in producer.sent} == {"doc.events"}
    assert producer.sent[0]["value"] == b"{}"
    assert producer.sent[1]["headers"] == [
        ("request_id", b"req-2"),
        ("schema", b"doc_event_v1"),
    ]


async def test_one_failed_acknowledgement_fails_the_whole_batch() -> None:
    # the batch is all or nothing, so the relay must not mark rows that kafka did not accept
    producer = FakeProducer({2: KafkaTimeoutError("no ack")})
    with pytest.raises(KafkaTimeoutError):
        await publisher(producer).publish([message(1), message(2), message(3)])
    assert len(producer.sent) == 3  # the others were still sent, they go out again as duplicates


async def test_a_broker_that_never_answers_ends_in_a_timeout_not_a_hang() -> None:
    producer = FakeProducer({1: "never"})
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        # the outer limit is only a safety net, the publisher's own limit must fire much sooner
        await asyncio.wait_for(publisher(producer, timeout_s=0.1).publish([message(1)]), 5)
    assert time.monotonic() - started < 2
