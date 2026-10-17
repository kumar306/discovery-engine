class PublisherReadyProbe:
    """Ready once the relay has started a producer.

    The health app is built before the producer exists, and the serve loop makes a new producer
    each time kafka was unreachable, so the probe starts out not ready and the loop marks it
    when a producer has started. The probe never calls the producer, so a flag is all it needs.
    This shows that the producer connected, a broker that dies a moment later shows up when a
    send fails and the relay backs off.
    """

    def __init__(self) -> None:
        self._connected = False

    def mark_connected(self) -> None:
        self._connected = True

    async def __call__(self) -> None:
        if not self._connected:
            raise RuntimeError("the producer is not connected")
