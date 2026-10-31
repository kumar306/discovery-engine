"""G11, the embedding bake off: which dense model, judged on hybrid nDCG@10, and what it costs.

Each model gets its own collection (the same judged products plus the first distractors of the
distractor ladder), the 500 validation queries are ranked through the ladder's own code, and the
rankings are scored and read by eval.ranking_diagnostics. The models are then compared with the
first one, the current model, by a paired bootstrap on the hybrid nDCG@10 per query.

THE VERDICT IS COMPUTED ON TODAY'S FUSION: the hybrid list is reciprocal rank fusion of the dense
and the bm25 lists, the way the product code does it. If the fusion is replaced later, a model
verdict taken here may change; the result says so. The product ids of the rankings are private:
they are written under benchmarks/private/ (ignored by git), never into the committed result, and
the result holds only their sha256, so a reader who has the private file can prove it is the one
this result was made from.

    python -m eval.bakeoff [--dataset data/processed/validation] [--distractors 500]
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from qdrant_client import AsyncQdrantClient

from eval.distractor_ladder import rank_distractors
from eval.embedding_cache import EmbeddingCache, embedding_cache_path
from eval.index_builder import build_index
from eval.ladder import LadderConfig, live_product_ids, run_ladder
from eval.metrics import paired_bootstrap, queries_needed
from eval.ranking_diagnostics import (
    HARD_QUERY_MAX_JUDGED,
    Diagnosis,
    RankedLists,
    diagnose,
    hard_query_ids,
)
from eval.results import write_result
from libs.embedding.model import OnnxEmbedder, build_onnx_embedder
from services.embed.settings import EmbedSettings
from services.query.encoder import OnnxQueryEncoder, QueryEncoder, SparseQuery
from services.query.search import QdrantSearch
from services.query.settings import QuerySettings

# the three rungs whose rankings are kept, in the order of the files of the ladder
_RUNGS = ("dense", "bm25", "hybrid")
# a model name becomes a file name, a collection name and a key of result.json
_NAME = re.compile(r"[a-z0-9_]{1,32}")
# the price of one doubling of the cost, in nDCG (a policy of the project, not derived from data)
PRICE_PER_DOUBLING = 0.01
# a gain smaller than this is not worth resolving, so "queries needed" never asks for more
MIN_TARGET_GAIN = 0.01
# queries per block of the stability check: 500 queries give ten blocks
BLOCK = 50
# the chunk texts and the queries timed to price a model, a sample is enough for a ratio
SPEED_CHUNKS = 200
SPEED_QUERIES = 100


@dataclass(frozen=True)
class ModelSpec:
    name: str
    dense_model: str
    dense_dim: int
    model_revision: int
    # text put in front of the QUERY for the dense model only (the stored chunks never get it).
    # Some models are trained that way: snowflake-arctic-embed says "prefixes: necessary" and
    # fastembed does not add it, so without it the model is measured below what it can do
    query_prefix: str = ""

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"model name {self.name!r} must be 1 to 32 of a-z, 0-9 and _")


# the sentence the bge and the arctic model cards give for a search query
RETRIEVAL_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# the current model first (plain bge_small), every other entry is compared with it. Each of the
# three models is run plain and with the query prefix (the name ends in _p): the plain entry is what
# production does today, the prefixed one is what the model card asks for
DEFAULT_MODELS = (
    ModelSpec("bge_small", "BAAI/bge-small-en-v1.5", 384, 1),
    ModelSpec("bge_small_p", "BAAI/bge-small-en-v1.5", 384, 1, RETRIEVAL_QUERY_PREFIX),
    ModelSpec("arctic_s", "snowflake/snowflake-arctic-embed-s", 384, 1),
    ModelSpec("arctic_s_p", "snowflake/snowflake-arctic-embed-s", 384, 1, RETRIEVAL_QUERY_PREFIX),
    ModelSpec("bge_base", "BAAI/bge-base-en-v1.5", 768, 1),
    ModelSpec("bge_base_p", "BAAI/bge-base-en-v1.5", 768, 1, RETRIEVAL_QUERY_PREFIX),
)


class PrefixedQueryEncoder:
    """A query encoder that puts the model prefix in front of the text the DENSE model sees."""

    def __init__(self, inner: QueryEncoder, prefix: str) -> None:
        self._inner = inner
        self._prefix = prefix

    async def encode_dense(self, query: str) -> list[float]:
        return await self._inner.encode_dense(self._prefix + query)

    async def encode_sparse(self, query: str) -> SparseQuery:
        # bm25 matches words, a sentence in front of the query would add words nobody typed
        return await self._inner.encode_sparse(query)


@dataclass(frozen=True)
class Speed:
    # wall time to embed one chunk text and to encode one query, on this machine (functional)
    embed_seconds_per_chunk: float
    query_seconds: float


@dataclass(frozen=True)
class ModelRun:
    spec: ModelSpec
    # query id -> the three lists, in the order of the queries
    rankings: Mapping[str, RankedLists]
    products: int
    chunks: int
    speed: Speed


@dataclass(frozen=True)
class ModelVerdict:
    model: str
    mean_gain: float
    low: float
    high: float
    price: float
    verdict: Literal["switch", "keep", "undecided"]
    # for an undecided model: the queries it would take to settle it, else None
    queries_needed: int | None
    # the mean gain of each run of BLOCK consecutive queries (the last one may be shorter)
    block_gains: tuple[float, ...]

    @property
    def net_value(self) -> float:
        return self.mean_gain - self.price


@dataclass(frozen=True)
class BakeoffConfig:
    dataset_directory: Path
    settings: QuerySettings
    distractors: int
    results_root: Path = Path("benchmarks/results")
    # the rankings (product ids) go here, ignored by git, never into the committed result
    private_root: Path = Path("benchmarks/private")
    draws: int = 200
    resamples: int = 1000
    seed: int = 7
    # 98.3 percent: the validation queries may be looked at three times (500, 1000, 2000)
    confidence: float = 0.983
    per_doubling: float = PRICE_PER_DOUBLING
    environment_kind: Literal["local", "ci", "aws"] = "local"
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))


type Measure = Callable[[ModelSpec], Awaitable[ModelRun]]


def cost_price(
    embed_time_ratio: float, query_latency_ratio: float, per_doubling: float = PRICE_PER_DOUBLING
) -> float:
    """The nDCG a model must gain to pay for what it costs; a cheaper model gets a credit."""
    # the cost of a model is the geometric mean of two ratios against the current model: the time
    # to embed a chunk and the time to answer a query. Every doubling of that cost is priced at
    # per_doubling nDCG. Examples: one ratio 4 and one 1 give a mean of 2, one doubling, +0.01;
    # both 0.25 give -0.02 (a credit: such a model only has to be not worse)
    if embed_time_ratio <= 0.0 or query_latency_ratio <= 0.0:
        raise ValueError("a cost ratio must be above zero")
    return per_doubling * math.log2(math.sqrt(embed_time_ratio * query_latency_ratio))


def compare_models(
    per_query: Mapping[str, Sequence[float]],
    current: str,
    price: Mapping[str, float],
    *,
    resamples: int,
    seed: int,
    confidence: float,
) -> list[ModelVerdict]:
    """A verdict for every model except the current one, best first.

    SWITCH when the low end of the interval of the gain is above the price (a cheaper model has a
    negative price, so it only has to be not worse by more than its credit), KEEP when the high end
    is below the price, UNDECIDED when the interval holds the price.
    """
    if current not in per_query:
        raise ValueError(f"the current model {current!r} has no scores")
    reference = per_query[current]
    for model, scores in per_query.items():
        if len(scores) != len(reference):
            raise ValueError("every model needs scores for the same queries")
        if model not in price:
            raise ValueError(f"model {model!r} has no price")
    verdicts: list[ModelVerdict] = []
    for model, scores in per_query.items():
        if model == current:
            continue
        gain = paired_bootstrap(
            reference, scores, resamples=resamples, seed=seed, confidence=confidence
        )
        gains = [after - before for before, after in zip(reference, scores, strict=True)]
        cost = price[model]
        verdict: Literal["switch", "keep", "undecided"]
        needed: int | None = None
        if gain.low > cost:
            verdict = "switch"
        elif gain.high < cost:
            verdict = "keep"
        else:
            verdict = "undecided"
            # the queries it takes for the interval to clear the price if the gain stays as it is
            needed = queries_needed(gains, max(abs(gain.mean - cost), MIN_TARGET_GAIN))
        verdicts.append(
            ModelVerdict(
                model=model,
                mean_gain=gain.mean,
                low=gain.low,
                high=gain.high,
                price=cost,
                verdict=verdict,
                queries_needed=needed,
                block_gains=tuple(
                    statistics.fmean(gains[start : start + BLOCK])
                    for start in range(0, len(gains), BLOCK)
                ),
            )
        )
    order = {"switch": 0, "undecided": 1, "keep": 2}
    return sorted(verdicts, key=lambda v: (order[v.verdict], -v.net_value))


def read_trec_rankings(
    query_ids: Sequence[str], files: Mapping[str, str]
) -> dict[str, RankedLists]:
    """The run files of the ladder (one per rung) as the lists of every query, in query order."""
    # a TREC line is  query_id Q0 product_id rank score rung  and the ladder writes them best first
    wanted = set(query_ids)
    found: dict[str, dict[str, list[tuple[int, str]]]] = {
        rung: {query_id: [] for query_id in query_ids} for rung in _RUNGS
    }
    for rung in _RUNGS:
        for number, line in enumerate(files[rung].split("\n"), start=1):
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) != 6:
                raise ValueError(f"{rung} file, line {number}: expected 6 fields, got {len(parts)}")
            query_id, _, product_id, rank, _, name = parts
            if name != rung:
                raise ValueError(f"{rung} file, line {number}: the line names {name!r}")
            if query_id not in wanted:
                raise ValueError(f"{rung} file, line {number}: query {query_id} was not scored")
            found[rung][query_id].append((int(rank), product_id))
    return {
        query_id: RankedLists(
            query_id,
            *(tuple(product for _, product in sorted(found[rung][query_id])) for rung in _RUNGS),
        )
        for query_id in query_ids
    }


def _rankings_file(rankings: Mapping[str, RankedLists]) -> bytes:
    """The private file of one model: a line per query with the three lists of product ids."""
    lines = [
        json.dumps(
            {
                "query_id": lists.query_id,
                "dense": list(lists.dense),
                "bm25": list(lists.bm25),
                "hybrid": list(lists.hybrid),
            },
            sort_keys=True,
        )
        for lists in rankings.values()
    ]
    return ("".join(line + "\n" for line in lines)).encode("utf-8")


def _read_judgements(dataset_directory: Path) -> dict[str, dict[str, str]]:
    # line by line from the file, never splitlines: real texts hold U+2028
    judgements: dict[str, dict[str, str]] = {}
    with (dataset_directory / "judgements.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            judgements.setdefault(str(row["query_id"]), {})[row["product_id"]] = row["label"]
    return judgements


def _metrics_of(
    name: str,
    run: ModelRun,
    diagnosis: Diagnosis,
    embed_ratio: float,
    query_ratio: float,
    price: float,
) -> dict[str, float]:
    metrics = {
        f"{name}_hybrid_ndcg_at_10": diagnosis.hybrid_ndcg,
        f"{name}_dense_ndcg_at_10": diagnosis.dense_ndcg,
        f"{name}_bm25_ndcg_at_10": diagnosis.bm25_ndcg,
        f"{name}_hybrid_recall_at_100": diagnosis.hybrid_recall_at_100,
        f"{name}_dense_recall_at_100": diagnosis.dense_recall_at_100,
        f"{name}_hard_queries": float(diagnosis.hard_queries),
        f"{name}_judged_share_top10": diagnosis.judged_share_top10,
        f"{name}_random_order_ndcg": diagnosis.random_order_ndcg,
        f"{name}_ceiling_top20": diagnosis.ceiling_top20,
        f"{name}_ceiling_top100": diagnosis.ceiling_top100,
        f"{name}_fusion_losses": float(diagnosis.fusion_losses),
        f"{name}_best_of_three": diagnosis.best_of_three,
        f"{name}_products": float(run.products),
        f"{name}_chunks": float(run.chunks),
        f"{name}_embed_seconds_per_chunk": run.speed.embed_seconds_per_chunk,
        f"{name}_query_seconds": run.speed.query_seconds,
        f"{name}_embed_time_ratio": embed_ratio,
        f"{name}_query_latency_ratio": query_ratio,
        f"{name}_price": price,
    }
    for rung, value in (
        ("hybrid", diagnosis.hybrid_ndcg_without_hard),
        ("dense", diagnosis.dense_ndcg_without_hard),
        ("bm25", diagnosis.bm25_ndcg_without_hard),
    ):
        if value is not None:
            metrics[f"{name}_{rung}_ndcg_at_10_without_hard"] = value
    for group in diagnosis.groups:
        key = f"{name}_group_{group.label.replace(' ', '_')}"
        metrics[f"{key}_queries"] = float(group.queries)
        metrics[f"{key}_hybrid"] = group.hybrid
        metrics[f"{key}_dense"] = group.dense
        metrics[f"{key}_bm25"] = group.bm25
    return metrics


def _notes(config: BakeoffConfig, candidates: int) -> str:
    many = candidates != 1
    return (
        "The verdict is computed on hybrid nDCG@10 under today's fusion, reciprocal rank fusion "
        f"with k={config.settings.rrf_k}; a different fusion may change it. The hard queries are "
        "fixed once from the hybrid lists of the current model and every model is scored on that "
        f"same set. {candidates} candidate{'s are each' if many else ' is'} compared with the "
        f"current model at {config.confidence:.1%}, so a small false SWITCH is possible. The "
        "rankings (product ids) are private: they are not in this folder, only the sha256 of each "
        "private file is in the config. The cost of a model is the ratio of its measured "
        "embedding time per chunk and of its measured query time to those of the current model "
        "(<model>_embed_time_ratio and <model>_query_latency_ratio in the metrics). Speeds were "
        "measured one model after the other in one process, with nothing else running by design "
        "(the run does not check this); they are functional numbers from this machine, not "
        "benchmark results."
    )


async def run_bakeoff(config: BakeoffConfig, specs: Sequence[ModelSpec], measure: Measure) -> Path:
    """Measure every model, compare with the first one, write the result and the private files."""
    if not specs:
        raise ValueError("at least one model is needed")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("a model name appears twice")
    current = names[0]

    # 1. one model after the other: each builds and deletes its own collection
    runs = {spec.name: await measure(spec) for spec in specs}

    # 2. score and read the rankings of every model
    # scores are paired by position, so every model must have been asked the same queries IN THE
    # SAME ORDER (checked before anything is scored: the hard queries are those of the current one)
    query_ids = list(runs[current].rankings)
    for name, run in runs.items():
        if list(run.rankings) != query_ids:
            raise ValueError(f"model {name!r} was not asked the same queries as {current!r}")
    judgements = _read_judgements(config.dataset_directory)
    # the hard queries are fixed ONCE, from the hybrid lists of the current model, and every model
    # is scored on that same set, so "without the hard queries" means the same queries for all
    hard = hard_query_ids(runs[current].rankings, judgements)
    diagnoses = {
        name: diagnose(
            run.rankings, judgements, hard_queries=hard, draws=config.draws, seed=config.seed
        )
        for name, run in runs.items()
    }

    # 3. the price of every model against the current one, then the verdicts
    base = runs[current].speed
    embed_ratio = {
        name: run.speed.embed_seconds_per_chunk / base.embed_seconds_per_chunk
        for name, run in runs.items()
    }
    query_ratio = {name: run.speed.query_seconds / base.query_seconds for name, run in runs.items()}
    price = {
        name: cost_price(embed_ratio[name], query_ratio[name], config.per_doubling) for name in runs
    }
    verdicts = compare_models(
        {name: [row.hybrid for row in d.per_query] for name, d in diagnoses.items()},
        current,
        price,
        resamples=config.resamples,
        seed=config.seed,
        confidence=config.confidence,
    )

    # 4. the numbers
    metrics: dict[str, float] = {}
    for name, run in runs.items():
        metrics.update(
            _metrics_of(
                name, run, diagnoses[name], embed_ratio[name], query_ratio[name], price[name]
            )
        )
    for verdict in verdicts:
        metrics[f"{verdict.model}_gain_mean"] = verdict.mean_gain
        metrics[f"{verdict.model}_gain_low"] = verdict.low
        metrics[f"{verdict.model}_gain_high"] = verdict.high

    # 5. the private files are written first under a temporary name; the committed result gets
    # their sha256 and nothing else about them
    benchmark_id = f"g11-bakeoff-{config.dataset_directory.name}"
    private_directory = config.private_root / benchmark_id
    pending = private_directory / f".pending-{uuid.uuid4().hex}"
    pending.mkdir(parents=True)
    result_config: dict[str, str | int | float | bool | None] = {
        "verdict_metric": "hybrid_ndcg_at_10",
        "fusion": "rrf",
        "rrf_k": config.settings.rrf_k,
        "current_model": current,
        "models": ",".join(names),
        "distractors": config.distractors,
        "queries": len(query_ids),
        "hard_query_max_judged": HARD_QUERY_MAX_JUDGED,
        "hard_query_source": current,
        "hard_queries": len(hard),
        "confidence": config.confidence,
        "resamples": config.resamples,
        "seed": config.seed,
        "draws": config.draws,
        "price_per_doubling": config.per_doubling,
    }
    for name, run in runs.items():
        result_config[f"query_prefix_{name}"] = run.spec.query_prefix
    for verdict in verdicts:
        result_config[f"verdict_{verdict.model}"] = verdict.verdict
        result_config[f"queries_needed_{verdict.model}"] = verdict.queries_needed
    try:
        for name, run in runs.items():
            data = _rankings_file(run.rankings)
            (pending / f"rankings_{name}.jsonl").write_bytes(data)
            result_config[f"private_rankings_{name}_sha256"] = hashlib.sha256(data).hexdigest()

        # 6. the raw files that are committed: per query numbers (ids of queries, no product ids)
        # and the comparison
        with tempfile.TemporaryDirectory() as scratch:
            raw_files: list[Path] = []
            for name, diagnosis in diagnoses.items():
                path = Path(scratch) / f"per_query_{name}.jsonl"
                path.write_text(
                    "".join(
                        json.dumps(
                            {
                                "query_id": row.query_id,
                                "judged_top10": row.judged_top10,
                                "hard": row.query_id in hard,
                                "hybrid_ndcg_at_10": row.hybrid,
                                "dense_ndcg_at_10": row.dense,
                                "bm25_ndcg_at_10": row.bm25,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                        for row in diagnosis.per_query
                    ),
                    encoding="utf-8",
                    newline="\n",
                )
                raw_files.append(path)
            comparison = Path(scratch) / "comparison.jsonl"
            comparison.write_text(
                "".join(
                    json.dumps(
                        {
                            "model": v.model,
                            "mean_gain": v.mean_gain,
                            "low": v.low,
                            "high": v.high,
                            "price": v.price,
                            "net_value": v.net_value,
                            "verdict": v.verdict,
                            "queries_needed": v.queries_needed,
                            "block_gains": list(v.block_gains),
                            "embed_time_ratio": embed_ratio[v.model],
                            "query_latency_ratio": query_ratio[v.model],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                    for v in verdicts
                ),
                encoding="utf-8",
                newline="\n",
            )
            raw_files.append(comparison)
            folder = write_result(
                config.results_root,
                benchmark_id,
                command="make bakeoff",
                environment_kind=config.environment_kind,
                dataset_directory=config.dataset_directory,
                model_version=config.settings.model_version,
                config=result_config,
                metrics=metrics,
                clock=config.clock,
                raw_files=raw_files,
                notes=_notes(config, len(verdicts)),
            )
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    # the private folder gets the name of the result folder, so the two belong together
    pending.rename(private_directory / folder.name)
    return folder


# ---------------- the real thing: Qdrant and the models, checked by a real run, not by unit tests


async def _sample_chunk_texts(client: AsyncQdrantClient, collection: str, count: int) -> list[str]:
    records, _ = await client.scroll(
        collection, limit=count, with_payload=["text"], with_vectors=False
    )
    return [str((record.payload or {})["text"]) for record in records]


async def _measure_speed(
    embedder: OnnxEmbedder,
    encoder: QueryEncoder,
    settings: QuerySettings,
    texts: list[str],
    queries: list[str],
) -> Speed:
    """Time the model on the same chunk texts and queries for every model (not the cache)."""
    # the first calls load and warm the model, they are not timed
    await embedder.embed(texts[:8], settings.embed_timeout_s)
    for query in queries[:3]:
        await encoder.encode_dense(query)
    start = time.perf_counter()
    await embedder.embed(texts, settings.embed_timeout_s)
    embed_seconds = (time.perf_counter() - start) / len(texts)
    sample = queries[:SPEED_QUERIES]
    start = time.perf_counter()
    for query in sample:
        await encoder.encode_dense(query)
    return Speed(embed_seconds, (time.perf_counter() - start) / len(sample))


async def run_on_dev_stack(config: BakeoffConfig, specs: Sequence[ModelSpec], threads: int) -> Path:
    settings = config.settings
    qdrant = AsyncQdrantClient(url=settings.qdrant_url, timeout=120)
    judged, ranked = rank_distractors(config.dataset_directory, 7)
    product_ids = judged | set(ranked[: config.distractors])
    with (config.dataset_directory / "queries.jsonl").open(encoding="utf-8") as handle:
        query_texts = [json.loads(line)["query"] for line in handle]

    async def measure(spec: ModelSpec) -> ModelRun:
        collection = f"{settings.qdrant_collection}_bakeoff_{spec.name}"
        build_settings = EmbedSettings(
            qdrant_url=settings.qdrant_url,
            qdrant_collection=collection,
            dense_model=spec.dense_model,
            dense_dim=spec.dense_dim,
            model_revision=spec.model_revision,
            bm25_avg_len=settings.bm25_avg_len,
            embed_threads=threads,
        )
        query_settings = settings.model_copy(
            update={
                "qdrant_collection": collection,
                "dense_model": spec.dense_model,
                "dense_dim": spec.dense_dim,
                "model_revision": spec.model_revision,
            }
        )
        embedder = build_onnx_embedder(build_settings)
        try:
            if await qdrant.collection_exists(collection):
                await qdrant.delete_collection(collection)
            # one cache file per model version: models never share vectors
            with EmbeddingCache(embedding_cache_path(build_settings.model_version)) as cache:
                report = await build_index(
                    config.dataset_directory,
                    build_settings,
                    qdrant,
                    embedder,
                    cache=cache,
                    product_ids=product_ids,
                )
            encoder = PrefixedQueryEncoder(
                OnnxQueryEncoder(embedder, query_settings.encode_timeout_s), spec.query_prefix
            )
            speed = await _measure_speed(
                embedder,
                encoder,
                query_settings,
                await _sample_chunk_texts(qdrant, collection, SPEED_CHUNKS),
                query_texts,
            )
            with tempfile.TemporaryDirectory() as scratch:
                folder = await run_ladder(
                    LadderConfig(
                        config.dataset_directory,
                        query_settings,
                        results_root=Path(scratch),
                        environment_kind=config.environment_kind,
                    ),
                    encoder,
                    QdrantSearch(qdrant, collection),
                    await live_product_ids(qdrant, collection),
                    expected_product_ids=product_ids,
                )
                with (folder / "raw" / "per_query.jsonl").open(encoding="utf-8") as handle:
                    scored = [str(json.loads(line)["query_id"]) for line in handle]
                files = {
                    rung: (folder / "raw" / f"{rung}.txt").read_text(encoding="utf-8")
                    for rung in _RUNGS
                }
            return ModelRun(
                spec, read_trec_rankings(scored, files), report.products, report.chunks, speed
            )
        finally:
            embedder.close()
            if await qdrant.collection_exists(collection):
                await qdrant.delete_collection(collection)

    try:
        return await run_bakeoff(config, specs, measure)
    finally:
        await qdrant.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="the embedding bake off on the validation set")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/validation"))
    parser.add_argument("--distractors", type=int, default=500)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    arguments = parser.parse_args(argv)
    config = BakeoffConfig(
        dataset_directory=arguments.dataset,
        settings=QuerySettings(),
        distractors=arguments.distractors,
    )
    folder = asyncio.run(run_on_dev_stack(config, DEFAULT_MODELS, arguments.threads))
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    print(f"result written to {folder}")
    print(f"the rankings are private: {config.private_root / folder.parent.name / folder.name}")
    print("verdicts are on hybrid nDCG@10 under reciprocal rank fusion")
    for spec in DEFAULT_MODELS:
        m = result["metrics"]
        print(
            f"{spec.name:10} hybrid {m[f'{spec.name}_hybrid_ndcg_at_10']:.4f}  "
            f"dense {m[f'{spec.name}_dense_ndcg_at_10']:.4f}  "
            f"recall@100 {m[f'{spec.name}_hybrid_recall_at_100']:.3f}  "
            f"price {m[f'{spec.name}_price']:+.4f}  "
            f"verdict {result['config'].get(f'verdict_{spec.name}', 'current model')}"
        )


if __name__ == "__main__":
    main()
