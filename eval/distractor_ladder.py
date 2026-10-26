"""The distractor ladder: how does nDCG fall as products that are not answers are added?

A production query is ranked against every live product of the catalogue. The judges labelled
about 20 products per query, so a corpus of the judged products plus a few thousand others is
easier than production and gives a nDCG that is too high. Here the same 500 validation queries are
searched over a corpus that grows by doubling the number of distractors (500, 1,000, 2,000, ...),
each size built as its own collection (the BM25 idf is computed per collection), and the ladder
stops when nDCG has stopped changing twice in a row: that size of corpus is then representative.

A step from one size to the next is FLAT when the WHOLE paired bootstrap interval of the change of
hybrid nDCG@10 lies inside a band around zero: low >= -epsilon and high <= +epsilon. An interval
that holds zero but is wide proves nothing (the change may be a fall of 0.03), so it is not flat.
epsilon is a required setting with no default: the owner writes its value and its reason in
docs/DECISIONS.md before the real run. The plateau is the smaller size of the first two flat steps
in a row; two is a guess, and two consecutive steps share the middle measurement, so they are not
independent evidence.

What a plateau claims, in these words: "doubling the corpus no longer changes hybrid nDCG@10 by
more than epsilon, with the interval inside the band, twice in a row". It does NOT claim that nDCG
has stopped falling: the changes of consecutive steps can add up. The mean of every step is
printed and saved, so the total over the confirming steps can be read.

If the curve never flattens before the distractors run out ("no plateau"),
that is the result: it is reported as it is, the ladder does not guess and stops for the owner to
decide. With 500 queries the intervals are wide, so "no plateau" is a likely and honest result;
the 2,104 queries of the validation bucket are the way to narrow them later.

The distractors of a size are the first of a fixed seeded order over the products that no chosen
query judged, so every corpus is inside the next one. Embedding is the slow part, so chunks are
embedded once in an on disk cache and every later size reuses them.

    python -m eval.distractor_ladder [--dataset data/processed/validation] [--threads N]
"""

import argparse
import asyncio
import json
import os
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from qdrant_client import AsyncQdrantClient

from eval.dataset import stable_rank
from eval.embedding_cache import EmbeddingCache, embedding_cache_path
from eval.index_builder import build_index
from eval.ladder import LadderConfig, live_product_ids, run_ladder
from eval.metrics import PairedGain, paired_bootstrap
from eval.results import write_result
from libs.embedding.model import build_onnx_embedder
from services.embed.settings import EmbedSettings
from services.query.encoder import OnnxQueryEncoder
from services.query.search import QdrantSearch
from services.query.settings import QuerySettings


@dataclass(frozen=True)
class Measurement:
    """The ladder run on a corpus with `size` distractors."""

    size: int
    products: int
    chunks: int
    # hybrid nDCG@10 of each query, always in the order of the queries of the dataset
    hybrid_ndcg_per_query: tuple[float, ...]
    # the means of the run, for example hybrid_ndcg_at_10, dense_ndcg_at_10, bm25_ndcg_at_10
    metrics: Mapping[str, float]


type Measure = Callable[[int], Awaitable[Measurement]]


@dataclass(frozen=True)
class LadderStep:
    from_size: int
    to_size: int
    gain: PairedGain
    flat: bool


@dataclass(frozen=True)
class DistractorCurve:
    measurements: tuple[Measurement, ...]
    steps: tuple[LadderStep, ...]
    # the smaller size of the first two flat steps in a row, None when the curve never flattened
    plateau: int | None
    stopped: Literal["flat twice", "no more distractors"]


@dataclass(frozen=True)
class DistractorLadderConfig:
    dataset_directory: Path
    settings: QuerySettings
    first_size: int
    largest_size: int
    # half width of the band around zero inside which a step is flat; no default on purpose
    epsilon: float
    results_root: Path = Path("benchmarks/results")
    resamples: int = 1000
    seed: int = 7
    environment_kind: Literal["local", "ci", "aws"] = "local"
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))


def rank_distractors(dataset_directory: Path, seed: int) -> tuple[frozenset[str], list[str]]:
    """The judged products, and every other product of the dataset in the seeded order.

    The distractors of a corpus with D distractors are the first D of the list, so a smaller
    corpus is always inside a bigger one.
    """
    # line by line from the file, never splitlines: real product texts hold U+2028
    with (dataset_directory / "judgements.jsonl").open(encoding="utf-8") as handle:
        judged = frozenset(json.loads(line)["product_id"] for line in handle)
    with (dataset_directory / "products.jsonl").open(encoding="utf-8") as handle:
        others = [
            product_id
            for product_id in (json.loads(line)["product_id"] for line in handle)
            if product_id not in judged
        ]
    return judged, sorted(others, key=lambda p: (stable_rank(seed, "distractor", p), p))


def ladder_collection_name(base: str, size: int) -> str:
    """The collection of a size. The ladder deletes the collections it names, and only these."""
    if size < 1:
        raise ValueError(f"size {size} is below 1")
    return f"{base}_ladder_d{size}"


def is_flat(gain: PairedGain, epsilon: float) -> bool:
    """A step is flat when the whole interval lies inside the band, the edges included."""
    return -epsilon <= gain.low and gain.high <= epsilon


async def climb(
    measure: Measure,
    *,
    first_size: int,
    largest_size: int,
    resamples: int,
    seed: int,
    epsilon: float,
) -> DistractorCurve:
    """Measure first_size, 2 x first_size, ... one by one, until two flat steps in a row.

    The rule runs live, and its verdict decides whether the next size (hours of embedding) is
    built, so it is made to err toward continuing: a wide interval is never flat. Running the rule
    again on saved numbers can only cover the sizes that were built, it cannot ask for a larger one.
    """
    if first_size > largest_size:
        raise ValueError(f"the first size {first_size} is above the largest size {largest_size}")
    measurements: list[Measurement] = []
    steps: list[LadderStep] = []
    size = first_size
    while size <= largest_size:
        measurements.append(await measure(size))
        if len(measurements) > 1:
            before, after = measurements[-2], measurements[-1]
            gain = paired_bootstrap(
                before.hybrid_ndcg_per_query,
                after.hybrid_ndcg_per_query,
                resamples=resamples,
                seed=seed,
            )
            steps.append(LadderStep(before.size, after.size, gain, is_flat(gain, epsilon)))
            # two flat steps in a row: nothing changes any more between the first of the three
            # sizes and the others, so that first size is big enough
            if len(steps) > 1 and steps[-1].flat and steps[-2].flat:
                return DistractorCurve(
                    tuple(measurements), tuple(steps), steps[-2].from_size, "flat twice"
                )
        size *= 2
    return DistractorCurve(tuple(measurements), tuple(steps), None, "no more distractors")


async def run_distractor_ladder(config: DistractorLadderConfig, measure: Measure) -> Path:
    """Climb the ladder and write one result folder with the curve, the steps and the plateau."""
    curve = await climb(
        measure,
        first_size=config.first_size,
        largest_size=config.largest_size,
        resamples=config.resamples,
        seed=config.seed,
        epsilon=config.epsilon,
    )
    metrics: dict[str, float] = {}
    for measurement in curve.measurements:
        metrics[f"d{measurement.size}_products"] = float(measurement.products)
        metrics[f"d{measurement.size}_chunks"] = float(measurement.chunks)
        for name, value in measurement.metrics.items():
            metrics[f"d{measurement.size}_{name}"] = value
    for step in curve.steps:
        prefix = f"step_{step.from_size}_{step.to_size}"
        metrics[f"{prefix}_mean"] = step.gain.mean
        metrics[f"{prefix}_low"] = step.gain.low
        metrics[f"{prefix}_high"] = step.gain.high
    if curve.plateau is not None:
        metrics["plateau_distractors"] = float(curve.plateau)

    with tempfile.TemporaryDirectory() as scratch:
        curve_path = Path(scratch) / "curve.jsonl"
        steps_path = Path(scratch) / "steps.jsonl"
        per_query_path = Path(scratch) / "per_query_hybrid.jsonl"
        for path, rows in (
            (
                curve_path,
                [
                    {
                        "size": m.size,
                        "products": m.products,
                        "chunks": m.chunks,
                        **dict(m.metrics),
                    }
                    for m in curve.measurements
                ],
            ),
            (
                steps_path,
                [
                    {
                        "from": s.from_size,
                        "to": s.to_size,
                        "mean": s.gain.mean,
                        "low": s.gain.low,
                        "high": s.gain.high,
                        "sd": s.gain.sd,
                        "flat": s.flat,
                    }
                    for s in curve.steps
                ],
            ),
            (
                per_query_path,
                [
                    {"size": m.size, "hybrid_ndcg_at_10": list(m.hybrid_ndcg_per_query)}
                    for m in curve.measurements
                ],
            ),
        ):
            # unix line endings, the checksum in result.json must survive a commit on windows
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
                newline="\n",
            )
        return write_result(
            config.results_root,
            f"g12-distractor-ladder-{config.dataset_directory.name}",
            command="make distractor-ladder",
            environment_kind=config.environment_kind,
            dataset_directory=config.dataset_directory,
            model_version=config.settings.model_version,
            config={
                "collection": config.settings.qdrant_collection,
                "first_size": config.first_size,
                "largest_size": config.largest_size,
                "resamples": config.resamples,
                "seed": config.seed,
                "epsilon": config.epsilon,
                "plateau": curve.plateau,
                "stopped": curve.stopped,
            },
            metrics=metrics,
            clock=config.clock,
            raw_files=[curve_path, steps_path, per_query_path],
        )


async def run_on_dev_stack(config: DistractorLadderConfig, threads: int) -> Path:
    """The ladder on the real Qdrant and the real model, one collection per size."""
    settings = config.settings
    judged, ranked = rank_distractors(config.dataset_directory, config.seed)
    qdrant = AsyncQdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_s)

    def builder_settings_of(collection: str) -> EmbedSettings:
        # the same index settings as the query service, with the threads of this machine
        return EmbedSettings(
            qdrant_url=settings.qdrant_url,
            qdrant_collection=collection,
            dense_model=settings.dense_model,
            dense_dim=settings.dense_dim,
            bm25_avg_len=settings.bm25_avg_len,
            model_revision=settings.model_revision,
            embed_threads=threads,
        )

    embedder = build_onnx_embedder(builder_settings_of(settings.qdrant_collection))
    try:
        with EmbeddingCache(embedding_cache_path(settings.model_version)) as cache:

            async def measure(size: int) -> Measurement:
                product_ids = judged | set(ranked[:size])
                collection = ladder_collection_name(settings.qdrant_collection, size)
                # these collections belong to the ladder (the name says so): a leftover of an
                # earlier run is replaced
                if await qdrant.collection_exists(collection):
                    await qdrant.delete_collection(collection)
                report = await build_index(
                    config.dataset_directory,
                    builder_settings_of(collection),
                    qdrant,
                    embedder,
                    cache=cache,
                    product_ids=product_ids,
                )
                size_settings = settings.model_copy(update={"qdrant_collection": collection})
                with tempfile.TemporaryDirectory() as scratch:
                    folder = await run_ladder(
                        LadderConfig(
                            config.dataset_directory,
                            size_settings,
                            results_root=Path(scratch),
                            environment_kind=config.environment_kind,
                        ),
                        OnnxQueryEncoder(embedder, settings.encode_timeout_s),
                        QdrantSearch(qdrant, collection),
                        await live_product_ids(qdrant, collection),
                        expected_product_ids=product_ids,
                    )
                    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
                    with (folder / "raw" / "per_query.jsonl").open(encoding="utf-8") as handle:
                        per_query = tuple(json.loads(line)["hybrid_ndcg_at_10"] for line in handle)
                await qdrant.delete_collection(collection)
                wanted = ("hybrid", "dense", "bm25")
                return Measurement(
                    size=size,
                    products=report.products,
                    chunks=report.chunks,
                    hybrid_ndcg_per_query=per_query,
                    metrics={
                        f"{rung}_ndcg_at_10": result["metrics"][f"{rung}_ndcg_at_10"]
                        for rung in wanted
                    },
                )

            return await run_distractor_ladder(config, measure)
    finally:
        embedder.close()
        await qdrant.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="the distractor ladder on the validation set")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/validation"))
    parser.add_argument("--first-size", type=int, default=500)
    parser.add_argument(
        "--largest-size", type=int, default=None, help="default: every distractor of the dataset"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="half width of the band around zero inside which a step is flat (docs/DECISIONS.md)",
    )
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    arguments = parser.parse_args(argv)
    available = len(rank_distractors(arguments.dataset, 7)[1])
    config = DistractorLadderConfig(
        dataset_directory=arguments.dataset,
        settings=QuerySettings(),
        first_size=arguments.first_size,
        largest_size=min(arguments.largest_size or available, available),
        epsilon=arguments.epsilon,
    )
    folder = asyncio.run(run_on_dev_stack(config, arguments.threads))
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    print(f"result written to {folder}")
    for line in (folder / "raw" / "steps.jsonl").read_text(encoding="utf-8").splitlines():
        step = json.loads(line)
        print(
            f"{step['from']:>6} -> {step['to']:<6} mean {step['mean']:+.4f} "
            f"interval [{step['low']:+.4f}, {step['high']:+.4f}] flat: {step['flat']}"
        )
    plateau = result["config"]["plateau"]
    print(f"plateau: {plateau or 'no plateau'}, stopped: {result['config']['stopped']}")


if __name__ == "__main__":
    main()
