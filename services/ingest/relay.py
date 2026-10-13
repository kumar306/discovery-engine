import asyncio
import logging
import random
import signal
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Protocol

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError
from psycopg_pool import AsyncConnectionPool

from libs.db.pool import create_pool
from libs.observability.logging import configure_logging
from libs.resilience.retry import RetryPolicy, backoff_delay
from services.ingest.settings import IngestSettings

log = logging.getLogger("ingest.relay")

# why an outbox: embedding is expensive and must not run while a database lock is held
# the event is stored in the same transaction as the document, so it cannot be lost between
# the commit and the send, and this relay carries it to kafka afterwards
# why kafka: a message holds no lock while its embedding is computed, and the log is a shock
# buffer that absorbs a burst of requests while the embeddings catch up
# a poison event can be retried a few times and then moved to the dead letter topic instead of
# failing forever, and spark can read the topic from offset zero to build the evaluation log

# the claim and the mark are two statements of one transaction
# skip locked skips the rows another relay is processing right now, so two relays never wait
# on each other and never send the same row at the same time
_CLAIM = """
select id, doc_id, request_id, event_schema, payload::text
from outbox
where published_at is null
order by id
limit %s
for update skip locked
"""

_MARK = "update outbox set published_at = now() where id = any(%s)"

_BACKLOG = "select count(*) from outbox where published_at is null"


@dataclass(frozen=True)
class OutboxMessage:
    id: int
    # the document id is the kafka key, so every event of one document lands on one partition
    key: str
    request_id: str
    schema: str
    value: str


class Publisher(Protocol):
    async def publish(self, messages: Sequence[OutboxMessage]) -> None: ...


class KafkaPublisher:
    def __init__(self, producer: AIOKafkaProducer, topic: str, timeout_s: float) -> None:
        self._producer = producer
        self._topic = topic
        self._timeout_s = timeout_s

    async def publish(self, messages: Sequence[OutboxMessage]) -> None:
        # the timeout keeps a produce from hanging forever when the broker never answers
        async with asyncio.timeout(self._timeout_s):
            # send queues a message and hands back a future, sending in order keeps the order
            futures = [
                await self._producer.send(
                    self._topic,
                    value=message.value.encode(),
                    key=message.key.encode(),
                    headers=[
                        ("request_id", message.request_id.encode()),
                        ("schema", message.schema.encode()),
                    ],
                )
                for message in messages
            ]
            # every future must be read, otherwise a failed one is reported as never retrieved
            results = await asyncio.gather(*futures, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result


class Relay:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        publisher: Publisher,
        batch_size: int,
        poll_interval_s: float,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
        policy: RetryPolicy | None = None,
    ) -> None:
        self._pool = pool
        self._publisher = publisher
        self._batch_size = batch_size
        self._poll_interval_s = poll_interval_s
        self._sleep = sleep
        # the rng only spreads retry waits, it protects nothing, so a plain generator is fine
        self._rng = rng or random.Random()  # noqa: S311
        self._policy = policy or RetryPolicy()

    async def publish_batch(self) -> int:
        # the transaction stays open from the claim until the mark
        # rows it holds are skipped by every other relay, and a crash anywhere in between
        # rolls it back, so the rows stay unpublished and go out again later
        async with self._pool.connection() as conn, conn.transaction():
            rows = await (await conn.execute(_CLAIM, (self._batch_size,))).fetchall()
            if not rows:
                return 0
            messages = [OutboxMessage(*row) for row in rows]
            await self._publisher.publish(messages)
            await conn.execute(_MARK, ([message.id for message in messages],))
        return len(messages)

    async def backlog(self) -> int:
        async with self._pool.connection() as conn:
            row = await (await conn.execute(_BACKLOG)).fetchone()
        return int(row[0]) if row else 0

    async def run(self, stop: asyncio.Event) -> None:
        failures = 0
        while not stop.is_set():
            try:
                published = await self.publish_batch()
            except Exception:
                # kafka or the database said no, the rows are still there, so wait and try again
                failures += 1
                log.exception(
                    "relay batch failed", extra={"fields": {"failures_in_a_row": failures}}
                )
                await self._sleep(backoff_delay(failures, self._policy, self._rng))
                continue
            failures = 0
            if published == 0:
                # nothing waiting, so sleep instead of asking the database in a tight loop
                await self._sleep(self._poll_interval_s)
            else:
                log.info("relay batch published", extra={"fields": {"published": published}})


# this function produces an async context: async setup before the yield, async teardown after it
# the instance is created first and yielded, and the caller consumes it with
# async with kafka_publisher(settings) as publisher, so the publisher exists only inside that block
# the finally runs on exit even if the block raised, so the producer is always stopped
@asynccontextmanager
async def kafka_publisher(settings: IngestSettings) -> AsyncIterator[KafkaPublisher]:
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap,
        # acks all makes the broker wait for every replica before it confirms a message
        acks="all",
        # idempotence only guards against redelivery of the same packet, if the producer itself
        # tries one, a new send of the same row after a rollback is a new packet, so duplicates
        # still reach the topic and the worker gate drops them later by version
        enable_idempotence=True,
        client_id="outbox-relay",
        request_timeout_ms=int(settings.relay_produce_timeout_s * 1000),
    )
    await producer.start()
    try:
        yield KafkaPublisher(producer, settings.topic_events, settings.relay_produce_timeout_s)
    finally:
        await producer.stop()


async def serve(
    settings: IngestSettings,
    stop: asyncio.Event,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    pool = create_pool(settings.database_dsn, settings.db_pool_max, settings.db_pool_timeout_s)
    await pool.open(wait=True, timeout=10)
    policy = RetryPolicy()
    rng = random.Random()  # noqa: S311
    failures = 0
    try:
        while not stop.is_set():
            try:
                async with kafka_publisher(settings) as publisher:
                    failures = 0
                    relay = Relay(
                        pool,
                        publisher,
                        settings.relay_batch_size,
                        settings.relay_poll_interval_s,
                        sleep,
                    )
                    await relay.run(stop)
            except KafkaError:
                # the broker was unreachable when the producer started, keep trying until stopped
                failures += 1
                log.warning(
                    "kafka not reachable, retrying", extra={"fields": {"attempt": failures}}
                )
                await sleep(backoff_delay(failures, policy, rng))
    finally:
        await pool.close()


def _stop_aware_sleep(stop: asyncio.Event) -> Callable[[float], Awaitable[None]]:
    async def pause(seconds: float) -> None:
        # stop is an event that is false until set, and stop.wait() freezes this coroutine until
        # it is true, the coroutine wakes in one of two ways
        # the timeout passes, wait_for raises TimeoutError and suppress swallows it, so this
        # works as a plain sleep, or SIGTERM or SIGINT sets stop and wakes it early
        # then the loop check not stop.is_set() fails and the graceful shutdown goes on
        with suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=seconds)

    return pause


async def main() -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for name in (signal.SIGTERM, signal.SIGINT):
        # windows cannot add signal handlers to the loop, there ctrl c raises instead
        with suppress(NotImplementedError):
            loop.add_signal_handler(name, stop.set)
    await serve(IngestSettings(), stop, sleep=_stop_aware_sleep(stop))


if __name__ == "__main__":
    configure_logging()
    if sys.platform == "win32":
        # psycopg async cannot use the default windows loop, linux needs nothing
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
