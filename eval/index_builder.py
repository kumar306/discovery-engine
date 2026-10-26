"""Put a dataset into a Qdrant collection for one model, without Kafka, Postgres or S3.

The content of a search index depends on three things only: how a product is cut into chunks
(compose), which vectors the model makes for them, and which payload each point carries. All three
are the production code here, the worker's own functions. Kafka, Postgres and S3 only deliver a
product to the worker; skipping them makes a build of the mini set take minutes instead of the 24
minutes of the real pipeline, and lets the quality gate and the model bake off run with nothing but
Qdrant. tests/integration/test_index_builder.py builds the same products through the real worker
and through this builder and compares the two collections point by point, so the two paths cannot
drift apart unnoticed.

Order, as in the worker: a batch of products is embedded, its points are written as staging
(invisible to every search), and only then flipped to live. A crash leaves staged points that no
search sees, and a rerun writes only what is missing.

    python -m eval.index_builder [--dataset data/processed/mini] [--collection NAME]
"""

import argparse
import asyncio
import json
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from eval.dataset import Product
from eval.embedding_cache import EmbeddingCache
from eval.loader import product_to_request
from libs.embedding.collection import normalise
from libs.embedding.model import Embedder, EmbedPermanent, build_onnx_embedder
from libs.schemas.documents import UpsertRequest
from libs.schemas.hashing import chunk_id
from services.embed.compose import Chunk, ComposeError, compose
from services.embed.qdrant import (
    ChunkIndex,
    ChunkPayload,
    IndexedChunk,
    LiveChunk,
    ensure_collection,
    new_client,
    publish_operations,
)
from services.embed.settings import EmbedSettings


class BuildStopped(Exception):
    """A product cannot be indexed, so the build stops and the collection must be discarded.

    Models are compared on the same products. A product that one model could index and another
    could not, left out quietly, would make the comparison unfair, so there is no skipping: the
    build stops at the first such product, names it, and the dataset is fixed for every model.
    """

    def __init__(self, product_id: str, reason: str) -> None:
        super().__init__(f"product {product_id} cannot be indexed: {reason}")
        self.product_id = product_id


@dataclass(frozen=True)
class BuildReport:
    # products that were written, and the chunks they became
    products: int
    chunks: int


@dataclass(frozen=True)
class _Composed:
    request: UpsertRequest
    chunks: tuple[Chunk, ...]


async def build_index(
    dataset_directory: Path,
    settings: EmbedSettings,
    qdrant: AsyncQdrantClient,
    embedder: Embedder,
    *,
    batch_chunks: int = 64,
    progress: Callable[[int, int], None] | None = None,
    cache: EmbeddingCache | None = None,
    product_ids: Collection[str] | None = None,
) -> BuildReport:
    """Write the products of the dataset into the collection of `settings`, live.

    product_ids, when given, limits the build to those products (the others in the file are read
    past without a check): the distractor ladder builds the judged products plus the first
    distractors. cache, when given, supplies the embeddings it holds and stores the ones it had to
    compute, so a build can be repeated or resumed without embedding the same chunk twice.
    """
    # every document got this version when the dataset was loaded through the api, so it is the
    # version of every point here too
    manifest = json.loads((dataset_directory / "manifest.json").read_text(encoding="utf-8"))
    version: int = manifest["dataset_revision"]

    # creates the collection for the model of the settings, or checks that an existing one is
    # right and stops when it is not
    await ensure_collection(qdrant, settings)
    index = ChunkIndex(qdrant, settings.qdrant_collection)

    products_written = 0
    chunks_written = 0
    batch: list[_Composed] = []

    async def write_batch() -> None:
        # embeds the chunks of the batch, stages them, then flips them to live
        nonlocal batch, products_written, chunks_written
        if not batch:
            return
        # every chunk of the batch with its id, in order
        entries = [
            (
                composed,
                chunk,
                chunk_id(composed.request.doc_id, settings.model_version, chunk.content_hash),
            )
            for composed in batch
            for chunk in composed.chunks
        ]
        # what the cache holds is not embedded again
        embedding_of = (
            cache.get_many([str(identifier) for _, _, identifier in entries]) if cache else {}
        )
        missing = [entry for entry in entries if str(entry[2]) not in embedding_of]
        if missing:
            try:
                fresh = await embedder.embed(
                    [chunk.text for _, chunk, _ in missing], settings.embed_timeout_s
                )
            except EmbedPermanent as refused:
                # the model refuses one text: the build stops and names the product that owns it
                # (the worker would send such a document to the dead letter queue and go on, a
                # comparison between models cannot)
                if 0 <= refused.index < len(missing):
                    raise BuildStopped(
                        missing[refused.index][0].request.doc_id,
                        f"the model refused a text: {refused}",
                    ) from refused
                raise
            computed = {
                str(identifier): embedding
                for (_, _, identifier), embedding in zip(missing, fresh, strict=True)
            }
            if cache is not None:
                cache.put_many(computed)
            embedding_of = {**embedding_of, **computed}
        indexed: list[IndexedChunk] = []
        live: list[LiveChunk] = []
        for composed, chunk, identifier in entries:
            request = composed.request
            embedding = embedding_of[str(identifier)]
            indexed.append(
                IndexedChunk(
                    identifier,
                    embedding.dense,
                    embedding.sparse_indices,
                    embedding.sparse_values,
                    ChunkPayload(
                        doc_id=request.doc_id,
                        version=version,
                        chunk_idx=chunk.positions[0],
                        content_hash=chunk.content_hash,
                        model_version=settings.model_version,
                        locale=request.locale,
                        text=chunk.text,
                        title=normalise(request.title),
                        brand=request.brand,
                        color=request.color,
                    ),
                )
            )
            live.append(LiveChunk(identifier, chunk.positions[0]))
        await index.stage(indexed)
        await qdrant.batch_update_points(
            settings.qdrant_collection,
            update_operations=publish_operations(version, live, []),
            wait=True,
        )
        products_written += len(batch)
        chunks_written += len(indexed)
        batch = []
        if progress is not None:
            progress(products_written, chunks_written)

    # the file is read line by line from the file, never with str.splitlines: that also breaks a
    # line at U+2028, which real product texts contain
    with (dataset_directory / "products.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            product = Product(**json.loads(line))
            if product_ids is not None and product.product_id not in product_ids:
                continue
            try:
                # the same conversion the bulk loader applies, with the api's own rules
                request = UpsertRequest.model_validate(product_to_request(product, version))
                chunks = compose(request, embedder.tokenizer, embedder.limits)
            except (ValueError, ComposeError) as error:
                raise BuildStopped(product.product_id, str(error)) from error
            batch.append(_Composed(request, chunks))
            # a batch is closed at a product boundary, so the chunks of one product always go live
            # together
            if sum(len(composed.chunks) for composed in batch) >= batch_chunks:
                await write_batch()
    await write_batch()
    return BuildReport(products_written, chunks_written)


async def build_on_dev_stack(
    dataset_directory: Path, settings: EmbedSettings, progress: Callable[[int, int], None]
) -> BuildReport:
    """Open Qdrant and the model, build, close both."""
    qdrant = new_client(settings)
    embedder = build_onnx_embedder(settings)
    try:
        return await build_index(dataset_directory, settings, qdrant, embedder, progress=progress)
    finally:
        embedder.close()
        await qdrant.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="build a Qdrant collection from a dataset")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/mini"))
    parser.add_argument("--collection", default=None, help="default: EMBED_QDRANT_COLLECTION")
    parser.add_argument("--dense-model", default=None, help="default: EMBED_DENSE_MODEL")
    parser.add_argument("--dense-dim", type=int, default=None, help="default: EMBED_DENSE_DIM")
    # the service default is 2 threads so that several workers share a machine, a build owns the
    # machine, measured on real chunks: 4.9 chunks per second with 2 threads, 9.4 with 8
    parser.add_argument("--threads", type=int, default=8, help="onnx threads for the dense model")
    arguments = parser.parse_args(argv)
    overrides: dict[str, object] = {"embed_threads": arguments.threads}
    if arguments.collection is not None:
        overrides["qdrant_collection"] = arguments.collection
    if arguments.dense_model is not None:
        overrides["dense_model"] = arguments.dense_model
    if arguments.dense_dim is not None:
        overrides["dense_dim"] = arguments.dense_dim
    settings = EmbedSettings(**overrides)  # type: ignore[arg-type]
    print(
        f"building {settings.qdrant_collection} ({settings.model_version}) from {arguments.dataset}"
    )
    try:
        report = asyncio.run(
            build_on_dev_stack(
                arguments.dataset,
                settings,
                lambda products, chunks: print(
                    f"  {products} products, {chunks} chunks", flush=True
                ),
            )
        )
    except BuildStopped as stopped:
        # the collection holds the products before this one and nothing is marked as complete
        raise SystemExit(
            f"{stopped}\ncollection {settings.qdrant_collection} is incomplete, delete it, "
            "fix the dataset for every model, and build again"
        ) from stopped
    print(f"done: {report.products} products, {report.chunks} chunks")


if __name__ == "__main__":
    main()
