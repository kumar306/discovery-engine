"""The avg_len preview: does the length constant of BM25 change the ranking? Offline, no index.

The sparse index is built with a fixed average chunk length (256, part of the model version),
while the real mean of our chunks is lower. BM25 divides the length of a chunk by that average, so
the constant decides how much a long chunk is punished. Re-indexing for every candidate value
would take hours, but the reference BM25 of eval/bm25.py (the oracle) can rank the same chunks with
any value in seconds. For each value it ranks every judged query over the chunks of the corpus,
turns the chunk ranking into a product ranking (a product sits where its best chunk sits, as in
Qdrant's grouping) and scores nDCG@10 against the judgements. Each value is compared with the
baseline (256) by the paired bootstrap on the per query scores.

Limit stated up front: this measures BM25 ALONE, not the hybrid. A gain here may shrink in the
hybrid, so a value that wins only says "a second experiment is worth its cost". The corpus is the
judged products plus the first `distractors` of the seeded order the distractor ladder uses.

    python -m eval.avg_len_preview [--dataset data/processed/validation] [--distractors 4000]
"""

import argparse
import json
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from eval.bm25 import Bm25Corpus, tokenise
from eval.dataset import Product
from eval.distractor_ladder import rank_distractors
from eval.loader import product_to_request
from eval.metrics import Judgements, PairedGain, ndcg_at_k, paired_bootstrap
from eval.results import write_result
from libs.embedding.collection import normalise
from libs.embedding.model import Limits, Tokenizer, build_onnx_embedder
from libs.schemas.documents import UpsertRequest
from services.embed.compose import ComposeError, compose
from services.query.settings import QuerySettings

type Verdict = Literal["baseline", "keep the baseline", "worth a hybrid experiment"]


@dataclass(frozen=True)
class AvgLenPreviewConfig:
    dataset_directory: Path
    # how many distractors join the judged products (the same seeded order as the distractor ladder)
    distractors: int
    # k1 and b of the BM25 model, read from the installed model by the caller, not typed here
    k1: float
    b: float
    fixed_values: tuple[float, ...] = (64.0, 256.0, 512.0)
    baseline: float = 256.0
    # a value must beat the baseline by this much nDCG@10 AND have an interval above 0
    min_gain: float = 0.01
    chunk_depth: int = 1000
    product_depth: int = 100
    resamples: int = 1000
    seed: int = 7
    distractor_seed: int = 7
    model_version: str = "unspecified"
    results_root: Path = Path("benchmarks/results")
    environment_kind: Literal["local", "ci", "aws"] = "local"
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class AvgLenRow:
    average_length: float
    # "mean" for the true mean of the corpus, else the number
    label: str
    # nDCG@10 of each query, in the order of the sorted query ids
    ndcg_per_query: tuple[float, ...]
    ndcg_at_10: float
    gain: PairedGain | None  # against the baseline, None for the baseline itself
    verdict: Verdict


def verdict_of(gain: PairedGain, min_gain: float) -> Verdict:
    """Worth a second experiment only when the mean gain reaches the minimum and the whole interval
    is above zero. Exactly the minimum counts, an interval that touches zero does not."""
    if gain.mean >= min_gain and gain.low > 0.0:
        return "worth a hybrid experiment"
    return "keep the baseline"


def preview_avg_len(
    config: AvgLenPreviewConfig,
    chunk_tokens: Mapping[str, Sequence[str]],
    product_of_chunk: Mapping[str, str],
    query_tokens: Mapping[str, Sequence[str]],
    judgements: Judgements,
) -> list[AvgLenRow]:
    """One row per length constant: the fixed values plus the true mean of these chunks.

    chunk_tokens: chunk id -> its tokens; product_of_chunk: chunk id -> product id;
    query_tokens: query id -> tokens; judgements: query id -> (product id -> letter).
    """
    for query_id in judgements:
        if query_id not in query_tokens:
            raise ValueError(f"query {query_id} is judged but has no tokens")
    corpus = Bm25Corpus(chunk_tokens)
    true_mean = corpus.average_length
    values = sorted({*config.fixed_values, true_mean})
    if config.baseline not in values:
        raise ValueError(f"the baseline {config.baseline} is not one of the values {values}")
    query_ids = sorted(judgements)

    # every value: every query through the oracle, chunks first, then products by their best chunk
    scores_of_value: dict[float, list[float]] = {}
    for value in values:
        scores: list[float] = []
        for query_id in query_ids:
            ranked_chunks = corpus.rank(
                query_tokens[query_id],
                k1=config.k1,
                b=config.b,
                average_length=value,
                limit=config.chunk_depth,
            )
            # a product is listed once, where its best chunk is: the dict keeps the first insert
            products: dict[str, None] = {}
            for scored in ranked_chunks:
                products.setdefault(product_of_chunk[scored.doc_id], None)
            scores.append(ndcg_at_k(list(products)[: config.product_depth], judgements[query_id]))
        scores_of_value[value] = scores

    rows: list[AvgLenRow] = []
    baseline_scores = scores_of_value[config.baseline]
    for value in values:
        scores = scores_of_value[value]
        label = "mean" if value == true_mean else f"{value:g}"
        gain: PairedGain | None = None
        verdict: Verdict = "baseline"
        if value != config.baseline:
            gain = paired_bootstrap(
                baseline_scores, scores, resamples=config.resamples, seed=config.seed
            )
            verdict = verdict_of(gain, config.min_gain)
        rows.append(
            AvgLenRow(value, label, tuple(scores), sum(scores) / len(scores), gain, verdict)
        )
    return rows


def run_avg_len_preview(
    config: AvgLenPreviewConfig,
    *,
    tokenizer: Tokenizer,
    limits: Limits,
    tokenise_text: Callable[[str], list[str]],
) -> Path:
    """Compose the chunks of the corpus, run the preview and write one result folder.

    tokenizer and limits are the embedding model's (the chunks are cut the way the builder cuts
    them), tokenise_text turns a text into BM25 tokens (the oracle's `tokenise` of the installed
    BM25 model; a plain function in the tests)."""
    directory = config.dataset_directory
    # 1. the corpus: every judged product and the first `distractors` of the seeded order
    judged_product_ids, ranked_distractors = rank_distractors(directory, config.distractor_seed)
    if config.distractors > len(ranked_distractors):
        raise ValueError(
            f"{config.distractors} distractors asked, the dataset has {len(ranked_distractors)}"
        )
    corpus_product_ids = judged_product_ids | set(ranked_distractors[: config.distractors])

    # 2. their chunks, cut by the builder's own composer, as BM25 tokens
    chunk_tokens: dict[str, list[str]] = {}
    product_of_chunk: dict[str, str] = {}
    with (directory / "products.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            product = Product(**json.loads(line))
            if product.product_id not in corpus_product_ids:
                continue
            try:
                request = UpsertRequest.model_validate(product_to_request(product, 1))
                chunks = compose(request, tokenizer, limits)
            except (ValueError, ComposeError) as error:
                raise ValueError(f"product {product.product_id}: {error}") from error
            for index, chunk in enumerate(chunks):
                chunk_id = f"{product.product_id}#{index}"
                chunk_tokens[chunk_id] = tokenise_text(chunk.text)
                product_of_chunk[chunk_id] = product.product_id

    # 3. the queries (folded as the query encoder folds them) and the judgements
    judgements: dict[str, dict[str, str]] = {}
    with (directory / "judgements.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            judgements.setdefault(str(row["query_id"]), {})[row["product_id"]] = row["label"]
    query_tokens: dict[str, list[str]] = {}
    with (directory / "queries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            query = json.loads(line)
            if str(query["query_id"]) in judgements:
                query_tokens[str(query["query_id"])] = tokenise_text(
                    normalise(query["query"]).casefold()
                )

    rows = preview_avg_len(config, chunk_tokens, product_of_chunk, query_tokens, judgements)

    # 4. the result: means and gains as metrics, rows and the per query table as raw files
    metrics: dict[str, float] = {
        "products": float(len(corpus_product_ids)),
        "chunks": float(len(chunk_tokens)),
        "queries": float(len(judgements)),
        "corpus_average_length": next(r.average_length for r in rows if r.label == "mean"),
    }
    for row in rows:
        metrics[f"avglen_{row.label}_ndcg_at_10"] = row.ndcg_at_10
        if row.gain is not None:
            metrics[f"avglen_{row.label}_gain_mean"] = row.gain.mean
            metrics[f"avglen_{row.label}_gain_low"] = row.gain.low
            metrics[f"avglen_{row.label}_gain_high"] = row.gain.high
    query_ids = sorted(judgements)
    with tempfile.TemporaryDirectory() as scratch:
        rows_path = Path(scratch) / "rows.jsonl"
        per_query_path = Path(scratch) / "per_query.jsonl"
        row_lines = [
            {
                "label": row.label,
                "average_length": row.average_length,
                "ndcg_at_10": row.ndcg_at_10,
                "gain_mean": None if row.gain is None else row.gain.mean,
                "gain_low": None if row.gain is None else row.gain.low,
                "gain_high": None if row.gain is None else row.gain.high,
                "gain_sd": None if row.gain is None else row.gain.sd,
                "verdict": row.verdict,
            }
            for row in rows
        ]
        query_lines = [
            {"query_id": query_id, **{row.label: row.ndcg_per_query[i] for row in rows}}
            for i, query_id in enumerate(query_ids)
        ]
        for path, lines in ((rows_path, row_lines), (per_query_path, query_lines)):
            # unix line endings, the checksum in result.json must survive a commit on windows
            path.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in lines),
                encoding="utf-8",
                newline="\n",
            )
        return write_result(
            config.results_root,
            "g12-avglen-preview",
            command="make avg-len-preview",
            environment_kind=config.environment_kind,
            dataset_directory=directory,
            model_version=config.model_version,
            config={
                "distractors": config.distractors,
                "k1": config.k1,
                "b": config.b,
                "baseline": config.baseline,
                "min_gain": config.min_gain,
                "chunk_depth": config.chunk_depth,
                "product_depth": config.product_depth,
                "resamples": config.resamples,
                "seed": config.seed,
                "distractor_seed": config.distractor_seed,
            },
            metrics=metrics,
            clock=config.clock,
            raw_files=[rows_path, per_query_path],
            notes="BM25 alone, not the hybrid: a gain here may shrink in the hybrid",
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="the avg_len preview with the BM25 oracle")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/validation"))
    parser.add_argument("--distractors", type=int, default=4000)
    parser.add_argument("--baseline", type=float, default=256.0)
    arguments = parser.parse_args(argv)
    # imported here, not at the top: fastembed loads only when the command runs
    from fastembed import SparseTextEmbedding

    settings = QuerySettings()
    # fastembed types it as a base class without k and b, the BM25 one has them
    bm25_model: Any = SparseTextEmbedding("Qdrant/bm25", avg_len=settings.bm25_avg_len).model
    embedder = build_onnx_embedder(settings)
    try:
        folder = run_avg_len_preview(
            AvgLenPreviewConfig(
                dataset_directory=arguments.dataset,
                distractors=arguments.distractors,
                # read from the installed model: the same defaults the real pipeline uses
                k1=bm25_model.k,
                b=bm25_model.b,
                baseline=arguments.baseline,
                model_version=settings.model_version,
            ),
            tokenizer=embedder.tokenizer,
            limits=embedder.limits,
            tokenise_text=lambda text: tokenise(bm25_model, text),
        )
    finally:
        embedder.close()
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    print(f"result written to {folder}  (BM25 alone, not the hybrid)")
    print(
        f"corpus: {result['metrics']['products']:.0f} products, "
        f"{result['metrics']['chunks']:.0f} chunks, "
        f"true mean length {result['metrics']['corpus_average_length']:.1f}"
    )
    for line in (folder / "raw" / "rows.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        gain = (
            ""
            if row["gain_mean"] is None
            else (
                f"  gain {row['gain_mean']:+.4f} [{row['gain_low']:+.4f}, {row['gain_high']:+.4f}]"
            )
        )
        print(
            f"  avg_len {row['label']:>6} ({row['average_length']:7.1f}): "
            f"nDCG@10 {row['ndcg_at_10']:.4f}{gain}  {row['verdict']}"
        )


if __name__ == "__main__":
    main()
