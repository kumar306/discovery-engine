"""The whole chain with the real model: api, outbox, relay, kafka, worker, qdrant, then searches.

Nothing here is a stand in except the clock of the api: the relay and the worker are the real
processes' code, the worker loads the real ONNX model, and the collection has the real shape
(384 floats and the bm25 vector). The searches use the same filter every reader will use,
state is live, so what they find is exactly what a searcher would see.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastembed import SparseTextEmbedding
from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient, models

from eval.dataset import Product
from eval.loader import LoadConfig, LoadReport, load_dataset
from libs.embedding.collection import CollectionMismatch, verify_collection
from libs.embedding.model import OnnxEmbedder, build_onnx_embedder
from services.embed.qdrant import ensure_collection
from services.embed.settings import EmbedSettings
from services.embed.worker import serve
from services.ingest.relay import Relay, kafka_publisher
from services.ingest.settings import IngestSettings
from services.query.encoder import OnnxQueryEncoder
from services.query.main import build as build_query_service
from services.query.settings import QuerySettings
from tests.unit.test_eval_loader import REVISION, dataset

from .conftest import Api, Database, committed_offsets, document, eventually, outbox, read_topic

PRODUCTS: dict[str, dict[str, Any]] = {
    "mouse": {
        "title": "Acme wireless mouse",
        "brand": "Acme",
        "color": "black",
        "bullets": ["Ergonomic design with silent clicks", "USB receiver included"],
        "description": "A comfortable cordless computer mouse for office work.",
    },
    "keyboard": {
        "title": "Acme mechanical keyboard",
        "brand": "Acme",
        "color": "white",
        "bullets": ["Hot swappable switches", "Tenkeyless layout"],
        "description": "Tactile typing for long coding sessions.",
    },
    "lamp": {
        "title": "Lumo desk lamp",
        "brand": "Lumo",
        "color": "white",
        "bullets": ["Dimmable LED", "Built in USB charging port"],
        "description": "Warm light for reading at your desk in the evening.",
    },
    "headset": {
        "title": "Zeta wireless headset",
        "brand": "Zeta",
        "color": "blue",
        "bullets": ["Over ear comfort", "Noise cancelling microphone"],
        "description": "Clear calls and immersive music.",
    },
    "cable": {
        "title": "Zeta braided USB cable",
        "brand": "Zeta",
        "color": "black",
        "bullets": ["Fast charging", "One metre long"],
        "description": "A durable cable for phones and tablets.",
    },
}


@pytest.fixture(autouse=True)
def moto_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    # moto accepts any credentials, in a pod these come from the role
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")


@dataclass
class System:
    api: Api
    settings: EmbedSettings
    qdrant_client: AsyncQdrantClient
    query_embedder: OnnxEmbedder
    sparse_queries: SparseTextEmbedding
    db: Database

    async def post(self, doc_id: str, version: int, **fields: Any) -> None:
        response = await self.api.post(doc_id, version, **fields)
        assert response.status_code == 202, response.text

    async def delete(self, doc_id: str, version: int) -> None:
        assert (await self.api.delete(doc_id, version)).status_code == 202

    async def wait_until_indexed(self, doc_id: str, version: int) -> None:
        async def indexed() -> bool:
            row = document(self.db.direct_dsn, doc_id)
            return row is not None and row["indexed"] == version

        await eventually(indexed, limit_s=60)

    async def dense_search(self, text: str, limit: int = 5) -> list[str]:
        (embedding,) = await self.query_embedder.embed([text], timeout_s=60)
        return await self._search(embedding.dense, "dense", limit)

    async def sparse_search(self, text: str, limit: int = 5) -> list[str]:
        query = next(iter(self.sparse_queries.query_embed(text)))
        vector = models.SparseVector(indices=query.indices.tolist(), values=query.values.tolist())
        return await self._search(vector, "bm25", limit)

    async def _search(self, vector: Any, using: str, limit: int) -> list[str]:
        live = models.Filter(
            must=[models.FieldCondition(key="state", match=models.MatchValue(value="live"))]
        )
        result = await self.qdrant_client.query_points(
            self.settings.qdrant_collection,
            query=vector,
            using=using,
            query_filter=live,
            limit=limit,
            with_payload=True,
        )
        found: list[str] = []
        for point in result.points:
            doc_id = str((point.payload or {})["doc_id"])
            if doc_id not in found:
                found.append(doc_id)
        return found


@pytest.fixture
async def system(
    api: Api,
    db: Database,
    pool: AsyncConnectionPool,
    qdrant: AsyncQdrantClient,
    topic: str,
    dlq_topic: str,
) -> AsyncIterator[System]:
    settings = EmbedSettings(
        qdrant_collection=f"search_test_{uuid.uuid4().hex[:10]}",
        database_dsn=db.pooled_dsn,
        s3_bucket=api.bucket,
        topic_events=topic,
        topic_dlq=dlq_topic,
        consumer_group=f"test-{uuid.uuid4().hex[:10]}",
        consumer_poll_ms=100,
    )
    await ensure_collection(qdrant, settings)
    query_embedder = build_onnx_embedder(settings)
    stop = asyncio.Event()
    publisher_settings = IngestSettings(topic_events=topic, relay_poll_interval_s=0.1)
    async with kafka_publisher(publisher_settings) as publisher:
        relay = Relay(pool, publisher, batch_size=100, poll_interval_s=0.1)
        relay_task = asyncio.create_task(relay.run(stop))
        worker_task = asyncio.create_task(serve(settings, stop))  # loads the real model itself
        try:
            yield System(
                api,
                settings,
                qdrant,
                query_embedder,
                SparseTextEmbedding("Qdrant/bm25", avg_len=settings.bm25_avg_len),
                db,
            )
        finally:
            stop.set()
            await asyncio.wait_for(asyncio.gather(relay_task, worker_task), timeout=30)
            query_embedder.close()
            await qdrant.delete_collection(settings.qdrant_collection)


async def test_products_travel_from_the_api_to_real_searches_and_changes_follow(
    system: System, dlq_topic: str
) -> None:
    # 1. five products go in through the api, and nothing else is done by hand
    for doc_id, fields in PRODUCTS.items():
        await system.post(doc_id, 1, **fields)
    for doc_id in PRODUCTS:
        await system.wait_until_indexed(doc_id, 1)

    # 2. a keyword search finds what contains the words, and only that
    assert (await system.sparse_search("noise cancelling"))[0] == "headset"
    assert (await system.sparse_search("dimmable"))[0] == "lamp"
    assert (await system.sparse_search("tenkeyless"))[0] == "keyboard"
    assert await system.sparse_search("xylophone") == []

    # 3. a meaning search finds the product even with no word in common
    assert (await system.dense_search("cordless computer mouse"))[0] == "mouse"
    assert (await system.dense_search("light for reading at night"))[0] == "lamp"
    assert (await system.dense_search("phone charging cable"))[0] == "cable"

    # 4. a changed product: the new words are found, and the old words are gone for good
    changed = {
        **PRODUCTS["mouse"],
        "description": "A vertical trackball designed to relieve wrist strain.",
    }
    await system.post("mouse", 2, **changed)
    await system.wait_until_indexed("mouse", 2)
    assert await system.sparse_search("trackball") == ["mouse"]
    assert await system.sparse_search("cordless") == []
    assert (await system.dense_search("vertical trackball for wrist pain"))[0] == "mouse"

    # 5. a deleted product disappears from both kinds of search
    await system.delete("lamp", 2)
    await system.wait_until_indexed("lamp", 2)
    assert await system.sparse_search("dimmable") == []
    assert "lamp" not in await system.dense_search("light for reading at night")

    # 6. nothing was dead lettered, and every message was committed (5 uploads, 1 update, 1 delete)
    assert await read_topic(dlq_topic, 0) == []
    offsets = await committed_offsets(system.settings.consumer_group)
    assert sum(offsets.values()) == 7


class ForcedOverload(httpx.AsyncBaseTransport):
    """Answers the first posts with 503 and passes everything else to the real api."""

    def __init__(self, inner: httpx.AsyncBaseTransport, refusals: int) -> None:
        self.inner = inner
        self.refusals = refusals

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/documents" and self.refusals > 0:
            self.refusals -= 1
            return httpx.Response(
                503, headers={"Retry-After": "0"}, json={"error": "overloaded"}, request=request
            )
        return await self.inner.handle_async_request(request)


async def test_the_loader_puts_a_dataset_through_the_api_and_a_second_run_changes_nothing(
    system: System, tmp_path: Path
) -> None:
    items = [
        Product(f"L{n}", "us", f"widget zorkunit{n}", None, None, None, None) for n in range(20)
    ]
    dataset(tmp_path, items)
    # the one private access: the real transport of the api client, to put a forced 503 in front
    real = system.api.client._transport
    async with httpx.AsyncClient(
        transport=ForcedOverload(real, refusals=3), base_url=system.api.client.base_url
    ) as client:
        config = LoadConfig(poll_interval_s=0.5, wait_timeout_s=120)
        report = await load_dataset(client, tmp_path, config)
        assert report == LoadReport(documents=20, failures={})
        for item in items:
            row = document(system.db.direct_dsn, item.product_id)
            assert row is not None
            assert row["indexed"] == REVISION
        # the loader said done, so a search must already find a product by its own word
        assert await system.sparse_search("zorkunit7") == ["L7"]
        outbox_rows = len(outbox(system.db.direct_dsn))
        assert (await load_dataset(client, tmp_path, config)).failures == {}
        assert len(outbox(system.db.direct_dsn)) == outbox_rows


# ---------------- the query service on top of the same chain


@asynccontextmanager
async def query_client(system: System) -> AsyncIterator[httpx.AsyncClient]:
    """The real query service (built the way main builds it), served in process."""
    settings = QuerySettings(
        qdrant_collection=system.settings.qdrant_collection, retriever_timeout_s=10.0
    )
    app = build_query_service(settings)
    # the lifespan verifies the collection and the model, as a real start does
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://query") as client:
            yield client


async def load_the_five_products(system: System) -> None:
    for doc_id, fields in PRODUCTS.items():
        await system.post(doc_id, 1, **fields)
    for doc_id in PRODUCTS:
        await system.wait_until_indexed(doc_id, 1)


async def test_the_search_flow_finds_products_whatever_the_capitalisation_and_hides_deleted_ones(
    system: System,
) -> None:
    await load_the_five_products(system)
    async with query_client(system) as client:

        async def search(**body: Any) -> dict[str, Any]:
            response = await client.post("/v1/search", json=body)
            assert response.status_code == 200, response.text
            answer: dict[str, Any] = response.json()
            assert answer["degraded"] == []
            return answer

        # hybrid retrieval puts the right product first, in words and in meaning
        for text, expected in (
            ("cordless computer mouse", "mouse"),
            ("light for reading at night", "lamp"),
            ("phone charging cable", "cable"),
            ("noise cancelling microphone", "headset"),
        ):
            assert (await search(query=text))["results"][0]["doc_id"] == expected
        found = (await search(query="wireless mouse"))["results"][0]
        assert (found["doc_id"], found["title"], found["brand"], found["color"]) == (
            "mouse",
            "Acme wireless mouse",
            "Acme",
            "black",
        )

        # capitalisation changes nothing: same products, same scores
        lower = await search(query="wireless mouse", explain=True)
        upper = await search(query="WIRELESS   MOUSE", explain=True)
        assert [(r["doc_id"], r["score"]) for r in lower["results"]] == [
            (r["doc_id"], r["score"]) for r in upper["results"]
        ]

        # filters are optional, exact on the folded value, and a filter nothing satisfies is empty
        acme = await search(query="wireless", filters={"brand": "ACME"})
        assert {r["doc_id"] for r in acme["results"]} == {"mouse", "keyboard"}
        black_zeta = await search(query="usb", filters={"brand": " zeta ", "color": "BLACK"})
        assert [r["doc_id"] for r in black_zeta["results"]] == ["cable"]
        assert (await search(query="wireless", filters={"brand": "Puma"}))["results"] == []

        # a deleted product is gone from the answer
        await system.delete("lamp", 2)
        await system.wait_until_indexed("lamp", 2)
        after = await search(query="light for reading at night")
        assert "lamp" not in {r["doc_id"] for r in after["results"]}


async def test_the_chunk_flow_returns_chunks_and_can_be_restricted_to_shown_products(
    system: System,
) -> None:
    await load_the_five_products(system)
    async with query_client(system) as client:
        everything = await client.post("/v1/chunks", json={"query": "wireless mouse"})
        assert everything.status_code == 200
        body = everything.json()
        assert body["degraded"] == []
        assert body["chunks"][0]["doc_id"] == "mouse"
        assert all(chunk["text"] and chunk["title"] for chunk in body["chunks"])
        only = await client.post(
            "/v1/chunks", json={"query": "wireless mouse", "doc_ids": ["lamp", "cable"]}
        )
        assert {chunk["doc_id"] for chunk in only.json()["chunks"]} <= {"lamp", "cable"}
        assert only.json()["chunks"]
        one_each = await client.post(
            "/v1/chunks", json={"query": "wireless mouse", "per_retriever_limit": 1}
        )
        assert len(one_each.json()["chunks"]) <= 2


async def test_nike_and_NIKE_become_the_same_vectors_with_the_real_models(
    system: System,
) -> None:
    encoder = OnnxQueryEncoder(system.query_embedder, timeout_s=30.0)
    assert await encoder.encode_dense("NIKE   Running SHOES") == await encoder.encode_dense(
        "nike running shoes"
    )
    assert await encoder.encode_sparse("NIKE   Running SHOES") == await encoder.encode_sparse(
        "nike running shoes"
    )


async def test_the_query_service_refuses_a_collection_built_for_another_model(
    system: System,
) -> None:
    other = QuerySettings(qdrant_collection=system.settings.qdrant_collection, model_revision=2)
    with pytest.raises(CollectionMismatch, match="model_version"):
        await verify_collection(system.qdrant_client, other)
