from aiokafka import AIOKafkaConsumer
from qdrant_client import AsyncQdrantClient

from libs.observability.health import Probe

# a probe is a function that returns when the dependency is fine and raises when it is not
# the health app runs them, cuts them off after a timeout and shows only the type of the error
# the postgres probe is shared by every process and lives in libs/observability/probes.py


def qdrant_probe(qdrant_client: AsyncQdrantClient, collection: str) -> Probe:
    async def probe() -> None:
        # qdrant answers and still has the collection this worker was started for
        await qdrant_client.get_collection(collection)

    return probe


class KafkaAssignmentProbe:
    """Ready once the group has given this worker partitions.

    The consumer is created after the health server starts, so it is attached later, and until
    then the worker is not ready. This shows that the worker is in the group, a broker that
    died a moment ago shows up when the group is lost, not at once.
    """

    def __init__(self) -> None:
        self._kafka_consumer: AIOKafkaConsumer | None = None

    def attach(self, kafka_consumer: AIOKafkaConsumer) -> None:
        self._kafka_consumer = kafka_consumer

    async def __call__(self) -> None:
        if self._kafka_consumer is None:
            raise RuntimeError("the consumer is not started")
        if not self._kafka_consumer.assignment():
            raise RuntimeError("no partitions assigned")
