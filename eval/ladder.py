"""G6, the retrieval ladder: four ways of searching, run on the same queries and scored.

A rung is one way of answering a query with a list of products:
    bm25         keywords only, the best chunk of each product by the sparse vector
    dense        meaning only, the same with the dense vector
    hybrid       the two product lists fused by rank (what POST /v1/search does)
    hybrid_chunk the two CHUNK lists fused, then each product kept at its first place
                 (what POST /v1/chunks gives the assistant, collapsed to products)
Each rung is asked of Qdrant on its own, the hybrid rungs only fuse what the single rungs
returned. The rankings are scored against the human judgements with nDCG, recall and MRR, and
the hybrid rung is compared with the two single retrievers by a paired bootstrap, because 50
queries are a noisy sample (is the fusion needed at all?). The result goes to
benchmarks/results/g6-ladder-<dataset>/<time>/.

    python -m eval.ladder [--dataset data/processed/mini]
"""

import argparse
import asyncio
import json
import tempfile
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from qdrant_client import AsyncQdrantClient, models

from eval.metrics import evaluate, paired_bootstrap, score_query
from eval.results import write_result
from libs.embedding.collection import QdrantPointState, verify_collection
from libs.embedding.model import OnnxEmbedder, build_onnx_embedder
from services.query.encoder import OnnxQueryEncoder, QueryEncoder
from services.query.primitives import rrf
from services.query.search import Filters, QdrantSearch
from services.query.settings import QuerySettings

type Rung = Literal["bm25", "dense", "hybrid", "hybrid_chunk"]
RUNGS: tuple[Rung, ...] = ("bm25", "dense", "hybrid", "hybrid_chunk")


class LadderRefused(Exception):
    """The index does not hold exactly the products of this dataset, a ladder on it would lie."""


@dataclass(frozen=True)
class RankedProduct:
    product_id: str
    # the score of the rung: the chunk score for bm25 and dense, the fusion score for the hybrids
    score: float


@dataclass(frozen=True)
class LadderConfig:
    dataset_directory: Path
    settings: QuerySettings
    results_root: Path = Path("benchmarks/results")
    # products asked of each single rung, recall at 100 needs 100
    depth: int = 100
    # chunks asked for the chunk rung: several chunks of one product are collapsed afterwards, so
    # it needs more than depth to still reach depth products
    chunk_depth: int = 200
    resamples: int = 1000
    seed: int = 7
    environment_kind: Literal["local", "ci", "aws"] = "local"
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))


async def rank_query(
    text: str,
    encoder: QueryEncoder,
    search: QdrantSearch,
    *,
    rrf_k: int,
    depth: int,
    chunk_depth: int,
) -> dict[Rung, list[RankedProduct]]:
    """The four rankings of one query, each a list of products, best first."""
    # 1. the query as the two vectors, with the same encoder as the service
    dense_vector = await encoder.encode_dense(text)
    sparse_vector = await encoder.encode_sparse(text)
    # a query whose words bm25 does not know has no terms, and the oracle test also skips such a
    # query, so bm25 answers nothing for it and the hybrid rungs are the dense list alone, which
    # is also what the service answers when its sparse retriever fails
    has_terms = bool(sparse_vector.indices)

    # 2. four searches, one per rung that needs Qdrant, no filters: the whole catalogue
    dense_products = await search.grouped_search("dense", dense_vector, Filters(), depth)
    dense_chunks = await search.chunk_search("dense", dense_vector, Filters(), chunk_depth)
    bm25_products = (
        await search.grouped_search("bm25", sparse_vector, Filters(), depth) if has_terms else []
    )
    bm25_chunks = (
        await search.chunk_search("bm25", sparse_vector, Filters(), chunk_depth)
        if has_terms
        else []
    )

    # 3. hybrid: fuse the two product lists, the same function and constant as the service
    fused_products = rrf(
        {
            "dense": [hit.doc_id for hit in dense_products],
            "bm25": [hit.doc_id for hit in bm25_products],
        },
        k=rrf_k,
    )

    # 4. hybrid_chunk: fuse the two chunk lists, then walk the fused chunks best first and keep
    # a product at the place of its best chunk, the chunks after it that belong to the same
    # product are dropped
    product_of_chunk = {hit.point_id: hit.doc_id for hit in dense_chunks + bm25_chunks}
    fused_chunks = rrf(
        {
            "dense": [hit.point_id for hit in dense_chunks],
            "bm25": [hit.point_id for hit in bm25_chunks],
        },
        k=rrf_k,
    )
    chunk_rung: list[RankedProduct] = []
    seen: set[str] = set()
    for fused in fused_chunks:
        product_id = product_of_chunk[fused.key]
        if product_id in seen:
            continue
        seen.add(product_id)
        chunk_rung.append(RankedProduct(product_id, fused.score))
        if len(chunk_rung) == depth:
            break

    return {
        "bm25": [RankedProduct(hit.doc_id, hit.score) for hit in bm25_products],
        "dense": [RankedProduct(hit.doc_id, hit.score) for hit in dense_products],
        "hybrid": [RankedProduct(item.key, item.score) for item in fused_products[:depth]],
        "hybrid_chunk": chunk_rung,
    }


async def run_ladder(
    config: LadderConfig,
    encoder: QueryEncoder,
    search: QdrantSearch,
    indexed_product_ids: Collection[str],
    *,
    expected_product_ids: Collection[str] | None = None,
) -> Path:
    """Run every query through the four rungs, score them, and write the result folder."""
    directory = config.dataset_directory

    # 1. the index must hold exactly the products of this dataset
    # a receipt written at load time cannot say which collection it is for, and an index that is
    # incomplete or holds other products would be trusted, so the live product ids of the
    # collection itself (read by the caller) are compared with the products of the dataset
    # the files are read line by line from the file, never with str.splitlines: that also breaks a
    # line at U+2028 and a few other characters, which real product texts contain
    # expected_product_ids names a smaller corpus than the file holds (the judged products plus the
    # first distractors of the distractor ladder), without it the whole file is expected
    if expected_product_ids is None:
        with (directory / "products.jsonl").open(encoding="utf-8") as handle:
            dataset_product_ids = {json.loads(line)["product_id"] for line in handle}
    else:
        dataset_product_ids = set(expected_product_ids)
    indexed = set(indexed_product_ids)
    if indexed != dataset_product_ids:
        raise LadderRefused(
            f"the index and {directory} hold different products: "
            f"{len(dataset_product_ids - indexed)} of the dataset are missing from the index, "
            f"{len(indexed - dataset_product_ids)} in the index are not in the dataset"
        )

    # 2. the queries, and what the judges said: query id -> (product id -> letter)
    # ids are kept as text, because a run and the judgements must use the same key
    with (directory / "queries.jsonl").open(encoding="utf-8") as handle:
        queries = [json.loads(line) for line in handle]
    judgements: dict[str, dict[str, str]] = {}
    with (directory / "judgements.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            judgements.setdefault(str(row["query_id"]), {})[row["product_id"]] = row["label"]

    # 3. every judged query through the four rungs, one at a time (a measurement, not a load test)
    # a query nobody judged cannot be scored and is only counted
    rankings: dict[str, dict[Rung, list[RankedProduct]]] = {}
    query_text: dict[str, str] = {}
    for query in queries:
        query_id = str(query["query_id"])
        if query_id not in judgements:
            continue
        query_text[query_id] = query["query"]
        rankings[query_id] = await rank_query(
            query["query"],
            encoder,
            search,
            rrf_k=config.settings.rrf_k,
            depth=config.depth,
            chunk_depth=config.chunk_depth,
        )
    scored_ids = list(rankings)

    # 4. score each rung: the means for the result, the per query nDCG for the bootstrap
    metrics: dict[str, float] = {
        "queries_scored": float(len(scored_ids)),
        "queries_unjudged": float(len(queries) - len(scored_ids)),
    }
    ndcg_of_query: dict[Rung, list[float]] = {}
    for rung in RUNGS:
        run = {
            query_id: [item.product_id for item in rankings[query_id][rung]]
            for query_id in scored_ids
        }
        scored = evaluate(run, {query_id: judgements[query_id] for query_id in scored_ids})
        metrics[f"{rung}_ndcg_at_10"] = scored.ndcg_at_10
        metrics[f"{rung}_recall_at_100"] = scored.recall_at_100
        metrics[f"{rung}_mrr_at_10"] = scored.mrr_at_10
        ndcg_of_query[rung] = [
            score_query(run[query_id], judgements[query_id]).ndcg_at_10 for query_id in scored_ids
        ]

    # 5. is the hybrid rung really better than each of the others, or is it the noise of 50 queries
    for other in ("bm25", "dense"):
        gain = paired_bootstrap(
            ndcg_of_query[other],
            ndcg_of_query["hybrid"],
            resamples=config.resamples,
            seed=config.seed,
        )
        metrics[f"hybrid_minus_{other}_ndcg_mean"] = gain.mean
        metrics[f"hybrid_minus_{other}_ndcg_low"] = gain.low
        metrics[f"hybrid_minus_{other}_ndcg_high"] = gain.high
        metrics[f"hybrid_minus_{other}_ndcg_sd"] = gain.sd

    # 6. the rankings as TREC run files and the per query numbers, then one result folder
    with tempfile.TemporaryDirectory() as scratch:
        raw_files: list[Path] = []
        for rung in RUNGS:
            path = Path(scratch) / f"{rung}.txt"
            lines = [
                f"{query_id} Q0 {item.product_id} {rank} {item.score} {rung}"
                for query_id in scored_ids
                for rank, item in enumerate(rankings[query_id][rung], start=1)
            ]
            # unix line endings, the checksum in result.json must survive a commit on windows
            path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            raw_files.append(path)
        per_query_path = Path(scratch) / "per_query.jsonl"
        per_query_rows = []
        for position, query_id in enumerate(scored_ids):
            numbers: dict[str, object] = {"query_id": query_id, "query": query_text[query_id]}
            for rung in RUNGS:
                numbers[f"{rung}_results"] = len(rankings[query_id][rung])
                numbers[f"{rung}_ndcg_at_10"] = ndcg_of_query[rung][position]
            per_query_rows.append(numbers)
        per_query_path.write_text(
            "".join(json.dumps(numbers, sort_keys=True) + "\n" for numbers in per_query_rows),
            encoding="utf-8",
            newline="\n",
        )
        raw_files.append(per_query_path)
        return write_result(
            config.results_root,
            f"g6-ladder-{directory.name}",
            command="make eval",
            environment_kind=config.environment_kind,
            dataset_directory=directory,
            model_version=config.settings.model_version,
            config={
                "collection": config.settings.qdrant_collection,
                "dense_model": config.settings.dense_model,
                "depth": config.depth,
                "chunk_depth": config.chunk_depth,
                "rrf_k": config.settings.rrf_k,
                "resamples": config.resamples,
                "seed": config.seed,
                "queries": len(queries),
            },
            metrics=metrics,
            clock=config.clock,
            raw_files=raw_files,
        )


async def live_product_ids(qdrant_client: AsyncQdrantClient, collection: str) -> set[str]:
    """The product id of every live chunk of the collection, read page by page."""
    live = models.Filter(
        must=[
            models.FieldCondition(
                key="state",
                match=models.MatchValue(value=QdrantPointState.QDRANT_POINT_LIVE.value),
            )
        ]
    )
    product_ids: set[str] = set()
    cursor = None
    while True:
        records, cursor = await qdrant_client.scroll(
            collection,
            scroll_filter=live,
            limit=1000,
            with_payload=["doc_id"],
            with_vectors=False,
            offset=cursor,
        )
        product_ids.update(str((record.payload or {})["doc_id"]) for record in records)
        if cursor is None:
            return product_ids


async def run_on_dev_stack(config: LadderConfig) -> Path:
    """Open Qdrant and the model, check the collection, run the ladder, close everything."""
    settings = config.settings
    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_s)
    embedder: OnnxEmbedder | None = None
    try:
        # the collection must be the one built for the model version of these settings
        await verify_collection(qdrant_client, settings)
        # the product ids of every live chunk, to prove that the collection holds this dataset
        indexed_product_ids = await live_product_ids(qdrant_client, settings.qdrant_collection)
        embedder = build_onnx_embedder(settings)
        return await run_ladder(
            config,
            OnnxQueryEncoder(embedder, settings.encode_timeout_s),
            QdrantSearch(qdrant_client, settings.qdrant_collection),
            indexed_product_ids,
        )
    finally:
        if embedder is not None:
            embedder.close()
        await qdrant_client.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="run the retrieval ladder on a loaded dataset")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/mini"))
    arguments = parser.parse_args(argv)
    folder = asyncio.run(run_on_dev_stack(LadderConfig(arguments.dataset, QuerySettings())))
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    print(f"result written to {folder}")
    for name, value in sorted(result["metrics"].items()):
        print(f"  {name}: {value:.4g}")


if __name__ == "__main__":
    main()
