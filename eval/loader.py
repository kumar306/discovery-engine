"""Send the processed ESCI products through the real ingest api and wait until each is searchable.

The loader is a client like any other: it posts to /v1/documents with a limit on requests in
flight, backs off when the api says 503, and asks /v1/document-statuses, in batches, until the
worker has published the version. It finishes only when every product is reported indexed, and
then it writes loaded.json next to the dataset, which the retrieval evaluations check.
The products file is only ever read line by line, a chunk at a time, never whole.
"""

import asyncio
import hashlib
import itertools
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from eval.dataset import Product, verify_dataset
from libs.schemas import MAX_STATUS_BATCH, UpsertRequest

# the file that says: every product of this dataset directory was reported indexed
LOADED_MARKER = "loaded.json"


class LoaderNotReady(Exception):
    """A readiness url of the system the loader sends to did not answer 200."""


@dataclass(frozen=True)
class LoadConfig:
    # requests on the wire at once, the api sheds load above 32 (max_inflight_requests)
    max_in_flight: int = 16
    # lines of products.jsonl handled at a time, this is what bounds the memory of the loader
    chunk_size: int = 1000
    # tries for one post when the api says 503 or does not answer
    max_post_attempts: int = 10
    # seconds to wait for every product to be indexed, counted from the end of the send phase
    wait_timeout_s: float = 1800.0
    # seconds between two rounds of asking for the status
    poll_interval_s: float = 2.0
    # urls that must answer 200 before anything is sent: the api, the worker, the relay
    ready_urls: tuple[str, ...] = ()


# the settings used when the caller passes none, the class is frozen so sharing one is safe
DEFAULT_CONFIG = LoadConfig()


@dataclass(frozen=True)
class LoadReport:
    documents: int
    # doc_id -> why it failed, empty when every document was posted and reported indexed
    failures: dict[str, str]


def product_to_request(product: Product, version: int) -> dict[str, Any]:
    """The body of POST /v1/documents for one product, checked against the api's own rules."""
    body: dict[str, Any] = {
        "corpus": "esci",
        "doc_id": product.product_id,
        "version": version,
        "title": product.title,
        "locale": product.locale,
    }
    for name, value in (
        ("brand", product.brand),
        ("color", product.color),
        ("description", product.description),
    ):
        if value is not None and value.strip():
            body[name] = value.strip()
    if product.bullet_point is not None:
        # the dataset joins the bullets of a product with newlines
        bullets = [line.strip() for line in product.bullet_point.split("\n") if line.strip()]
        if bullets:
            body["bullets"] = bullets
    try:
        UpsertRequest.model_validate(body)
    except ValidationError as error:
        fields = sorted({str(part) for item in error.errors() for part in item["loc"]})
        raise ValueError(
            f"product {product.product_id} breaks the api rules in: {', '.join(fields)}"
        ) from error
    return body


async def post_document(
    client: httpx.AsyncClient,
    gate: asyncio.Semaphore,
    body: dict[str, Any],
    config: LoadConfig,
    sleep: Callable[[float], Awaitable[None]],
) -> str | None:
    """Post one document. None when the api accepted it (202), else the reason it did not."""
    payload = json.dumps(body).encode()
    for _ in range(config.max_post_attempts):
        try:
            # the gate counts requests on the wire, a retry that is sleeping holds no place
            async with gate:
                response = await client.post("/v1/documents", content=payload)
        except httpx.TransportError:
            # no answer or a timeout, the post may or may not have arrived, posting again is
            # safe because the same version with the same content is answered 202 as a duplicate
            await sleep(1.0)
            continue
        if response.status_code == 202:
            return None
        if response.status_code == 503:
            await sleep(float(response.headers.get("retry-after", "1")))
            continue
        # 409 (a newer version is held, or this version has other content) and 422 never get
        # better by waiting
        return f"{response.status_code} {response.text[:200]}"
    return f"no 202 after {config.max_post_attempts} attempts"


async def ask_status_batch(
    client: httpx.AsyncClient, gate: asyncio.Semaphore, batch: Sequence[str], version: int
) -> dict[str, str]:
    """One status request for up to MAX_STATUS_BATCH documents.

    Returns doc_id -> why this version is not indexed yet. An indexed document is left out.
    """
    try:
        async with gate:
            response = await client.post("/v1/document-statuses", json={"doc_ids": list(batch)})
    except httpx.TransportError as error:
        return dict.fromkeys(batch, f"no answer ({type(error).__name__})")
    if response.status_code != 200:
        return dict.fromkeys(batch, f"status answered {response.status_code}")
    indexed_version_by_doc_id = {
        item["doc_id"]: item["indexed_version"] for item in response.json()["statuses"]
    }
    reasons: dict[str, str] = {}
    for doc_id in batch:
        if doc_id not in indexed_version_by_doc_id:
            reasons[doc_id] = "the api does not know this document yet"
            continue
        indexed_version = indexed_version_by_doc_id[doc_id]
        if indexed_version is None or indexed_version < version:
            reasons[doc_id] = f"indexed_version is {indexed_version}, waiting for {version}"
    return reasons


async def reasons_not_indexed(
    client: httpx.AsyncClient, gate: asyncio.Semaphore, doc_ids: Sequence[str], version: int
) -> dict[str, str]:
    """doc_id -> why this version is not indexed yet, for any number of documents."""
    answers = await asyncio.gather(
        *(
            ask_status_batch(client, gate, batch, version)
            for batch in itertools.batched(doc_ids, MAX_STATUS_BATCH)
        )
    )
    return {doc_id: reason for answer in answers for doc_id, reason in answer.items()}


async def load_dataset(
    client: httpx.AsyncClient,
    directory: Path,
    config: LoadConfig = DEFAULT_CONFIG,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> LoadReport:
    # an old marker must never vouch for a load that failed halfway
    marker_path = directory / LOADED_MARKER
    marker_path.unlink(missing_ok=True)
    verify_dataset(directory)
    # the system must be up before anything is sent, a dead worker would otherwise be found
    # only after the whole wait timeout
    for url in config.ready_urls:
        try:
            response = await client.get(url)
        except httpx.TransportError as error:
            raise LoaderNotReady(f"{url} did not answer ({type(error).__name__})") from error
        if response.status_code != 200:
            raise LoaderNotReady(f"{url} answered {response.status_code}")
    manifest_bytes = (directory / "manifest.json").read_bytes()
    version: int = json.loads(manifest_bytes)["dataset_revision"]
    products_path = directory / "products.jsonl"

    # pass 1 over the file: a product the api would refuse stops the load here, before the
    # first request is sent, only the ids are kept
    doc_ids: list[str] = []
    with products_path.open(encoding="utf-8") as products_file:
        for line in products_file:
            doc_ids.append(product_to_request(Product(**json.loads(line)), version)["doc_id"])

    # what is already indexed at this version is not sent again, a second run sends nothing
    gate = asyncio.Semaphore(config.max_in_flight)
    reason_by_doc_id = await reasons_not_indexed(client, gate, doc_ids, version)
    doc_ids_to_send = set(reason_by_doc_id)

    # pass 2 over the file, send phase: one chunk of lines at a time, at most max_in_flight of
    # a chunk's posts on the wire
    failure_by_doc_id: dict[str, str] = {}
    with products_path.open(encoding="utf-8") as products_file:
        for lines in itertools.batched(products_file, config.chunk_size):
            bodies = [product_to_request(Product(**json.loads(line)), version) for line in lines]
            bodies = [body for body in bodies if body["doc_id"] in doc_ids_to_send]
            post_results = await asyncio.gather(
                *(post_document(client, gate, body, config, sleep) for body in bodies)
            )
            for body, reason in zip(bodies, post_results, strict=True):
                if reason is not None:
                    failure_by_doc_id[body["doc_id"]] = reason

    # wait phase, a round asks about every document that is not indexed yet
    doc_ids_waiting = sorted(doc_ids_to_send - failure_by_doc_id.keys())
    deadline = clock() + config.wait_timeout_s
    while doc_ids_waiting:
        reason_by_doc_id = await reasons_not_indexed(client, gate, doc_ids_waiting, version)
        doc_ids_waiting = sorted(reason_by_doc_id)
        if not doc_ids_waiting:
            break
        if clock() >= deadline:
            for doc_id in doc_ids_waiting:
                failure_by_doc_id[doc_id] = (
                    f"not indexed after {config.wait_timeout_s} s: {reason_by_doc_id[doc_id]}"
                )
            break
        await sleep(config.poll_interval_s)

    report = LoadReport(documents=len(doc_ids), failures=failure_by_doc_id)
    if not report.failures:
        marker = {
            "dataset_revision": version,
            "documents": report.documents,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "loaded_at": datetime.now(UTC).isoformat(),
        }
        marker_path.write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report
