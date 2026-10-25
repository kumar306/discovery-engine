"""The quality job in one command: make sure the index exists, run the ladder, run the gate.

This is what CI runs (make quality) and what a developer can run on a machine that has Qdrant.
It never deletes or overwrites a collection that is there: if the collection holds exactly the
products of the dataset it is reused, if it is missing it is built (eval/index_builder.py), and
if it holds anything else the job stops and says so, because that collection is somebody's.

    python -m eval.quality_job [--dataset data/processed/mini] [--threads N]
exit code: 0 the gate passed, 1 it failed, 2 the job could not run (wrong index, a product that
cannot be indexed, no baseline, a baseline for another dataset or model)
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Literal

from qdrant_client import AsyncQdrantClient

from eval.gate import main as gate
from eval.index_builder import BuildStopped, build_index
from eval.ladder import LadderConfig, live_product_ids, run_ladder
from libs.embedding.collection import verify_collection
from libs.embedding.model import Embedder, build_onnx_embedder
from services.embed.settings import EmbedSettings
from services.query.encoder import OnnxQueryEncoder
from services.query.search import QdrantSearch
from services.query.settings import QuerySettings


class IndexMismatch(Exception):
    """The collection exists but does not hold exactly the products of the dataset."""


async def ensure_index(
    dataset_directory: Path,
    settings: EmbedSettings,
    qdrant: AsyncQdrantClient,
    embedder: Embedder,
) -> Literal["reused", "built"]:
    """Build the collection when it is missing, reuse it when it holds the dataset, else refuse."""
    # line by line from the file, never splitlines: real product texts hold U+2028
    with (dataset_directory / "products.jsonl").open(encoding="utf-8") as handle:
        dataset_product_ids = {json.loads(line)["product_id"] for line in handle}
    if not await qdrant.collection_exists(settings.qdrant_collection):
        await build_index(dataset_directory, settings, qdrant, embedder)
        return "built"
    indexed = await live_product_ids(qdrant, settings.qdrant_collection)
    if indexed != dataset_product_ids:
        raise IndexMismatch(
            f"collection {settings.qdrant_collection} exists and holds other products than "
            f"{dataset_directory}: {len(dataset_product_ids - indexed)} of the dataset are "
            f"missing, "
            f"{len(indexed - dataset_product_ids)} in the collection are not in the dataset; "
            "it is not touched, delete it yourself or use another collection name"
        )
    return "reused"


async def run_quality_job(dataset_directory: Path, threads: int, results_root: Path) -> int:
    """Index, ladder, gate. Returns the exit code."""
    settings = QuerySettings()
    # the builder reads the same index settings as the query service, and the threads given
    builder_settings = EmbedSettings(
        qdrant_url=settings.qdrant_url,
        qdrant_collection=settings.qdrant_collection,
        dense_model=settings.dense_model,
        dense_dim=settings.dense_dim,
        bm25_avg_len=settings.bm25_avg_len,
        model_revision=settings.model_revision,
        embed_threads=threads,
    )
    qdrant = AsyncQdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_s)
    embedder = build_onnx_embedder(builder_settings)
    try:
        started = time.monotonic()
        try:
            how = await ensure_index(dataset_directory, builder_settings, qdrant, embedder)
        except (IndexMismatch, BuildStopped) as refused:
            print(refused)
            return 2
        print(f"index {how} in {time.monotonic() - started:.0f} s", flush=True)
        # the collection must be the one built for the model version of these settings
        await verify_collection(qdrant, settings)
        started = time.monotonic()
        folder = await run_ladder(
            LadderConfig(
                dataset_directory,
                settings,
                results_root=results_root,
                environment_kind="ci" if os.environ.get("GITHUB_ACTIONS") == "true" else "local",
            ),
            OnnxQueryEncoder(embedder, settings.encode_timeout_s),
            QdrantSearch(qdrant, settings.qdrant_collection),
            await live_product_ids(qdrant, settings.qdrant_collection),
        )
        print(f"ladder in {time.monotonic() - started:.0f} s: {folder}", flush=True)
    finally:
        embedder.close()
        await qdrant.close()
    return gate(["check", "--result", str(folder)])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="index, ladder and gate in one command")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/mini"))
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    arguments = parser.parse_args(argv)
    raise SystemExit(
        asyncio.run(run_quality_job(arguments.dataset, arguments.threads, arguments.results))
    )


if __name__ == "__main__":
    main()
