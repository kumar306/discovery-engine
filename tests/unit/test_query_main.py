"""The wiring of the query service: what build() hands to the embedder."""

from typing import Any

import httpx

from services.query import main
from services.query.settings import QuerySettings


class StubEmbedder:
    queued_jobs = 5

    def close(self) -> None:
        return None


async def test_the_encoder_wait_of_the_embedder_reaches_the_metrics_endpoint(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    def fake_build_onnx_embedder(settings: Any, workers: int = 1, wait_observer: Any = None) -> Any:
        captured["workers"] = workers
        captured["wait_observer"] = wait_observer
        return StubEmbedder()

    monkeypatch.setattr(main, "build_onnx_embedder", fake_build_onnx_embedder)
    # the real client would try to reach a server when it is made
    monkeypatch.setattr(main, "AsyncQdrantClient", lambda **kwargs: object())
    settings = QuerySettings(encode_workers=3)
    app = main.build(settings)
    assert captured["workers"] == 3
    # what the embedder calls from its model thread lands in the histogram of this service
    captured["wait_observer"](0.02)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://q") as client:
        text = (await client.get("/metrics")).text
    assert "query_encode_wait_seconds_count 1.0" in text
    assert "query_encode_wait_seconds_sum 0.02" in text
    # the queue depth is read from the embedder at the moment of the scrape
    assert "query_encode_queued 5.0" in text
