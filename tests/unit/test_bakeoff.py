"""The embedding bake off: which dense model, judged on hybrid nDCG, and what it costs."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from eval.bakeoff import (
    DEFAULT_MODELS,
    RETRIEVAL_QUERY_PREFIX,
    BakeoffConfig,
    ModelRun,
    ModelSpec,
    ModelVerdict,
    PrefixedQueryEncoder,
    Speed,
    compare_models,
    cost_price,
    read_trec_rankings,
    run_bakeoff,
)
from eval.dataset import (
    JudgementRow,
    Product,
    QueryRow,
    Selection,
    SelectionConfig,
    write_dataset,
)
from eval.ranking_diagnostics import HARD_QUERY_MAX_JUDGED, RankedLists
from eval.results import ResultExists
from services.query.encoder import SparseQuery
from services.query.settings import QuerySettings

NOW = datetime(2026, 10, 28, 9, 30, tzinfo=UTC)

# ---------------- the price of a model


def test_a_model_as_costly_as_the_current_one_has_no_price() -> None:
    assert cost_price(1.0, 1.0) == 0.0


def test_a_price_is_one_hundredth_per_doubling_of_the_geometric_mean_of_the_two_ratios() -> None:
    # one ratio of 4 and one of 1: the geometric mean is 2, one doubling
    assert cost_price(4.0, 1.0) == pytest.approx(0.01)
    assert cost_price(1.0, 4.0) == pytest.approx(0.01)
    # both 4: the mean is 4, two doublings
    assert cost_price(4.0, 4.0) == pytest.approx(0.02)


def test_a_cheaper_model_has_a_negative_price_a_credit() -> None:
    assert cost_price(0.25, 1.0) == pytest.approx(-0.01)
    assert cost_price(0.25, 0.25) == pytest.approx(-0.02)


def test_a_slower_embedding_and_a_faster_query_cancel_out() -> None:
    assert cost_price(4.0, 0.25) == pytest.approx(0.0)


def test_the_price_per_doubling_is_a_parameter() -> None:
    assert cost_price(4.0, 1.0, per_doubling=0.05) == pytest.approx(0.05)


def test_a_ratio_of_zero_or_less_has_no_price() -> None:
    with pytest.raises(ValueError, match="ratio"):
        cost_price(0.0, 1.0)
    with pytest.raises(ValueError, match="ratio"):
        cost_price(1.0, -2.0)


# ---------------- the verdicts
#
# 100 queries; the current model scores 0.25 and 0.5 in turn. A candidate that is better by a
# constant 0.125 on every query (a number a float holds exactly) has a gain whose interval is
# the single point 0.125, so the edge cases are exact

CURRENT = [0.25, 0.5] * 50


def shifted(delta: float) -> list[float]:
    return [score + delta for score in CURRENT]


def judge(candidates: dict[str, list[float]], price: dict[str, float]) -> dict[str, ModelVerdict]:
    verdicts = compare_models(
        {"cur": CURRENT, **candidates},
        "cur",
        {"cur": 0.0, **price},
        resamples=200,
        seed=1,
        confidence=0.95,
    )
    return {v.model: v for v in verdicts}


def test_a_candidate_clearly_above_its_price_switches() -> None:
    verdict = judge({"x": shifted(0.125)}, {"x": 0.0625})["x"]
    assert verdict.verdict == "switch"
    assert verdict.mean_gain == 0.125
    assert verdict.queries_needed is None


def test_a_gain_exactly_equal_to_the_price_does_not_switch() -> None:
    verdict = judge({"x": shifted(0.125)}, {"x": 0.125})["x"]
    assert verdict.verdict == "undecided"


def test_a_candidate_clearly_below_its_price_is_kept_out() -> None:
    verdict = judge({"x": shifted(0.0625)}, {"x": 0.125})["x"]
    assert verdict.verdict == "keep"
    assert verdict.queries_needed is None


def test_a_candidate_whose_interval_holds_the_price_is_undecided_and_says_how_many_queries() -> (
    None
):
    # gains +0.25 and -0.25 in turn: mean 0, a wide interval
    noisy = [score + (0.25 if index % 2 else -0.25) for index, score in enumerate(CURRENT)]
    verdict = judge({"x": noisy}, {"x": 0.0})["x"]
    assert verdict.verdict == "undecided"
    # the sample standard deviation of the gains is 0.2513, the target is the floor of 0.01
    assert verdict.queries_needed is not None
    assert verdict.queries_needed > 100


def test_a_cheaper_model_that_is_a_bit_worse_but_inside_the_margin_switches() -> None:
    # the price is a credit of 0.0625, the model is worse by 0.03125: not worse than the margin
    verdict = judge({"x": shifted(-0.03125)}, {"x": -0.0625})["x"]
    assert verdict.verdict == "switch"


def test_a_cheaper_model_worse_than_its_credit_is_not_switched_to() -> None:
    assert judge({"x": shifted(-0.0625)}, {"x": -0.0625})["x"].verdict == "undecided"
    assert judge({"x": shifted(-0.125)}, {"x": -0.0625})["x"].verdict == "keep"


def test_the_current_model_gets_no_verdict_and_is_not_compared_with_itself() -> None:
    verdicts = compare_models(
        {"cur": CURRENT, "x": shifted(0.125)},
        "cur",
        {"cur": 0.0, "x": 0.0},
        resamples=100,
        seed=1,
        confidence=0.95,
    )
    assert [v.model for v in verdicts] == ["x"]


def test_switches_come_first_best_net_value_first_then_undecided_then_keep() -> None:
    noisy = [score + (0.25 if index % 2 else -0.25) for index, score in enumerate(CURRENT)]
    verdicts = compare_models(
        {
            "cur": CURRENT,
            "keepme": shifted(0.0),
            "small": shifted(0.125),
            "big": shifted(0.25),
            "unsure": noisy,
        },
        "cur",
        {"cur": 0.0, "keepme": 0.125, "small": 0.0, "big": 0.0625, "unsure": 0.0},
        resamples=100,
        seed=1,
        confidence=0.95,
    )
    # net value: big 0.25 - 0.0625 = 0.1875, small 0.125 - 0 = 0.125
    assert [v.model for v in verdicts] == ["big", "small", "unsure", "keepme"]


def test_the_ten_block_gains_are_the_mean_gain_of_each_run_of_fifty_queries() -> None:
    # first fifty queries better by 0.125, next fifty by 0.25
    candidate = [s + (0.125 if i < 50 else 0.25) for i, s in enumerate(CURRENT)]
    verdict = judge({"x": candidate}, {"x": 0.0})["x"]
    assert verdict.block_gains == (0.125, 0.25)


def test_a_last_block_shorter_than_fifty_is_kept() -> None:
    current = [0.25, 0.5] * 60
    candidate = [s + (0.125 if i < 100 else 0.5) for i, s in enumerate(current)]
    verdicts = compare_models(
        {"cur": current, "x": candidate},
        "cur",
        {"cur": 0.0, "x": 0.0},
        resamples=50,
        seed=1,
        confidence=0.95,
    )
    assert verdicts[0].block_gains == (0.125, 0.125, 0.5)


def test_a_wider_confidence_widens_the_interval_of_the_verdict() -> None:
    noisy = [score + (0.25 if index % 3 else -0.125) for index, score in enumerate(CURRENT)]
    scores = {"cur": CURRENT, "x": noisy}
    prices = {"cur": 0.0, "x": 0.0}
    narrow = compare_models(scores, "cur", prices, resamples=500, seed=1, confidence=0.8)[0]
    wide = compare_models(scores, "cur", prices, resamples=500, seed=1, confidence=0.983)[0]
    assert wide.low < narrow.low
    assert wide.high > narrow.high


def test_scores_for_different_numbers_of_queries_are_refused() -> None:
    with pytest.raises(ValueError, match="same queries"):
        compare_models(
            {"cur": CURRENT, "x": CURRENT[:-1]},
            "cur",
            {"cur": 0.0, "x": 0.0},
            resamples=10,
            seed=1,
            confidence=0.95,
        )


def test_a_missing_current_model_or_a_missing_price_is_refused() -> None:
    with pytest.raises(ValueError, match="current"):
        compare_models({"x": CURRENT}, "cur", {"x": 0.0}, resamples=10, seed=1, confidence=0.95)
    with pytest.raises(ValueError, match="price"):
        compare_models(
            {"cur": CURRENT, "x": CURRENT},
            "cur",
            {"cur": 0.0},
            resamples=10,
            seed=1,
            confidence=0.95,
        )


# ---------------- reading the run files of the ladder


def test_the_run_files_become_ranked_lists_in_the_order_of_the_ranks() -> None:
    files = {
        "dense": "1 Q0 P2 2 0.5 dense\n1 Q0 P1 1 0.9 dense\n2 Q0 P9 1 0.7 dense\n",
        "bm25": "1 Q0 P1 1 3.0 bm25\n",
        "hybrid": "1 Q0 P1 1 0.03 hybrid\n1 Q0 P2 2 0.02 hybrid\n",
    }
    rankings = read_trec_rankings(["1", "2"], files)
    assert rankings["1"] == RankedLists("1", ("P1", "P2"), ("P1",), ("P1", "P2"))
    assert list(rankings) == ["1", "2"]


def test_a_query_a_rung_returned_nothing_for_has_an_empty_list() -> None:
    files = {"dense": "1 Q0 P1 1 0.9 dense\n", "bm25": "", "hybrid": "1 Q0 P1 1 0.03 hybrid\n"}
    rankings = read_trec_rankings(["1", "2"], files)
    assert rankings["2"] == RankedLists("2", (), (), ())
    assert rankings["1"].bm25 == ()


def test_a_line_that_names_another_rung_or_has_the_wrong_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="dense"):
        read_trec_rankings(["1"], {"dense": "1 Q0 P1 1 0.9 bm25\n", "bm25": "", "hybrid": ""})
    with pytest.raises(ValueError, match="line"):
        read_trec_rankings(["1"], {"dense": "1 P1 1\n", "bm25": "", "hybrid": ""})


def test_a_line_for_a_query_that_was_not_scored_is_refused() -> None:
    with pytest.raises(ValueError, match="query"):
        read_trec_rankings(["1"], {"dense": "7 Q0 P1 1 0.9 dense\n", "bm25": "", "hybrid": ""})


# ---------------- the models


def test_a_model_name_is_a_plain_word_because_it_becomes_a_file_name() -> None:
    ModelSpec("bge_small", "BAAI/bge-small-en-v1.5", 384, 1)
    for bad in ("", "a/b", "..", "has space", "UPPER", "x" * 33):
        with pytest.raises(ValueError, match="name"):
            ModelSpec(bad, "BAAI/bge-small-en-v1.5", 384, 1)


# ---------------- the prefix some models need in front of a query


class RecordingEncoder:
    def __init__(self) -> None:
        self.dense: list[str] = []
        self.sparse: list[str] = []

    async def encode_dense(self, query: str) -> list[float]:
        self.dense.append(query)
        return [0.5]

    async def encode_sparse(self, query: str) -> SparseQuery:
        self.sparse.append(query)
        return SparseQuery([1], [1.0])


async def test_the_prefix_goes_in_front_of_the_query_the_dense_model_sees() -> None:
    inner = RecordingEncoder()
    encoder = PrefixedQueryEncoder(inner, "Represent this: ")
    assert await encoder.encode_dense("red shoes") == [0.5]
    assert inner.dense == ["Represent this: red shoes"]


async def test_the_bm25_terms_never_see_the_prefix() -> None:
    inner = RecordingEncoder()
    encoder = PrefixedQueryEncoder(inner, "Represent this: ")
    assert await encoder.encode_sparse("red shoes") == SparseQuery([1], [1.0])
    assert inner.sparse == ["red shoes"]


async def test_an_empty_prefix_changes_nothing() -> None:
    inner = RecordingEncoder()
    encoder = PrefixedQueryEncoder(inner, "")
    await encoder.encode_dense("red shoes")
    assert inner.dense == ["red shoes"]


def test_a_model_has_no_query_prefix_unless_it_is_given_one() -> None:
    assert ModelSpec("a", "m", 8, 1).query_prefix == ""
    assert ModelSpec("a", "m", 8, 1, query_prefix="p: ").query_prefix == "p: "


# ---------------- the whole run on a tiny dataset, a fake stands in for Qdrant and the model
#
# judged: q1 P1 exact, P2 substitute, P6 complement. q2 P3 exact, P4 and P7 irrelevant. q3 P5 exact.
# SECRET is in every product id: no file of the committed result folder may contain it

PRODUCTS = [f"B0SECRET{n}" for n in range(1, 8)]
JUDGEMENT_ROWS = (
    JudgementRow(1, PRODUCTS[0], "E"),
    JudgementRow(1, PRODUCTS[1], "S"),
    JudgementRow(1, PRODUCTS[5], "C"),
    JudgementRow(2, PRODUCTS[2], "E"),
    JudgementRow(2, PRODUCTS[3], "I"),
    JudgementRow(2, PRODUCTS[6], "I"),
    JudgementRow(3, PRODUCTS[4], "E"),
)


def make_dataset(directory: Path) -> Path:
    directory.mkdir(parents=True)
    items = [Product(pid, "us", f"title {pid}", None, None, None, None) for pid in PRODUCTS]
    selection = Selection(
        (
            QueryRow(1, "a", "validation"),
            QueryRow(2, "b", "validation"),
            QueryRow(3, "c", "validation"),
        ),
        JUDGEMENT_ROWS,
        tuple(sorted(PRODUCTS)),
    )
    write_dataset(directory, selection, items, SelectionConfig(seed=1), 1)
    return directory


def rankings_of(quality: str) -> dict[str, RankedLists]:
    p = PRODUCTS
    if quality == "perfect":
        return {
            "1": RankedLists("1", (p[0], p[1], p[5]), (p[0],), (p[0], p[1], p[5])),
            "2": RankedLists("2", (p[2], p[3], p[6]), (p[2],), (p[2], p[3], p[6])),
            "3": RankedLists("3", ("X1",), ("X1",), ("X1",)),
        }
    # the exact product sits last of the three for q1 and q2, q3 is not found at all (by either)
    return {
        "1": RankedLists("1", (p[5], p[1], p[0]), (p[5],), (p[5], p[1], p[0])),
        "2": RankedLists("2", (p[6], p[3], p[2]), (p[6],), (p[6], p[3], p[2])),
        "3": RankedLists("3", ("X1",), ("X1",), ("X1",)),
    }


SPECS = (
    ModelSpec("bge_small", "BAAI/bge-small-en-v1.5", 384, 1),
    ModelSpec("arctic_s", "snowflake/snowflake-arctic-embed-s", 384, 1),
    ModelSpec("bge_base", "BAAI/bge-base-en-v1.5", 768, 1),
)
QUALITY = {"bge_small": "worse", "arctic_s": "perfect", "bge_base": "perfect"}
SPEED = {
    "bge_small": Speed(0.1, 0.01),
    "arctic_s": Speed(0.1, 0.01),
    "bge_base": Speed(0.4, 0.04),
}


class FakeMeasure:
    def __init__(self, quality: dict[str, str] | None = None) -> None:
        self.quality = quality or QUALITY
        self.calls: list[str] = []

    async def __call__(self, spec: ModelSpec) -> ModelRun:
        self.calls.append(spec.name)
        return ModelRun(spec, rankings_of(self.quality[spec.name]), 500, 700, SPEED[spec.name])


def config_for(tmp_path: Path, now: datetime = NOW) -> BakeoffConfig:
    dataset = tmp_path / "validation"
    if not dataset.exists():
        make_dataset(dataset)
    return BakeoffConfig(
        dataset_directory=dataset,
        settings=QuerySettings(),
        distractors=500,
        results_root=tmp_path / "results",
        private_root=tmp_path / "private",
        draws=20,
        resamples=100,
        clock=lambda: now,
    )


async def test_the_models_are_measured_one_by_one_in_the_given_order(tmp_path: Path) -> None:
    fake = FakeMeasure()
    await run_bakeoff(config_for(tmp_path), SPECS, fake)
    assert fake.calls == ["bge_small", "arctic_s", "bge_base"]


async def test_the_result_has_the_numbers_of_every_model_and_says_what_the_verdict_stands_on(
    tmp_path: Path,
) -> None:
    folder = await run_bakeoff(config_for(tmp_path), SPECS, FakeMeasure())
    assert folder.parent.name == "g11-bakeoff-validation"
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    config = result["config"]
    assert config["verdict_metric"] == "hybrid_ndcg_at_10"
    assert config["fusion"] == "rrf"
    assert config["rrf_k"] == QuerySettings().rrf_k
    assert config["distractors"] == 500
    assert config["current_model"] == "bge_small"
    assert config["hard_query_max_judged"] == HARD_QUERY_MAX_JUDGED
    assert "reciprocal rank fusion" in result["notes"]
    assert "private" in result["notes"]
    metrics = result["metrics"]
    # q1 and q2 are perfect for the perfect models, q3 is a miss: (1 + 1 + 0) / 3
    assert metrics["arctic_s_hybrid_ndcg_at_10"] == pytest.approx(2 / 3)
    assert metrics["bge_small_hybrid_ndcg_at_10"] < metrics["arctic_s_hybrid_ndcg_at_10"]
    # q3 is the only hard query (nothing judged in its list); without it the perfect models score 1
    assert metrics["arctic_s_hard_queries"] == 1
    assert metrics["arctic_s_hybrid_ndcg_at_10_without_hard"] == pytest.approx(1.0)
    assert metrics["bge_base_chunks"] == 700


async def test_the_prices_come_from_the_measured_speeds_against_the_current_model(
    tmp_path: Path,
) -> None:
    folder = await run_bakeoff(config_for(tmp_path), SPECS, FakeMeasure())
    metrics = json.loads((folder / "result.json").read_text(encoding="utf-8"))["metrics"]
    assert metrics["bge_small_price"] == 0.0
    assert metrics["arctic_s_price"] == pytest.approx(0.0)
    # bge_base is 4 times slower to embed and 4 times slower to answer: two doublings
    assert metrics["bge_base_price"] == pytest.approx(0.02)
    assert metrics["bge_base_embed_time_ratio"] == pytest.approx(4.0)
    assert metrics["bge_base_query_latency_ratio"] == pytest.approx(4.0)


async def test_the_verdicts_are_in_the_config_and_in_the_comparison_file(tmp_path: Path) -> None:
    folder = await run_bakeoff(config_for(tmp_path), SPECS, FakeMeasure())
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    assert set(result["config"]) >= {"verdict_arctic_s", "verdict_bge_base"}
    assert "verdict_bge_small" not in result["config"]
    rows = [
        json.loads(line)
        for line in (folder / "raw" / "comparison.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sorted(row["model"] for row in rows) == ["arctic_s", "bge_base"]
    assert all(
        set(row) >= {"mean_gain", "low", "high", "price", "verdict", "block_gains"} for row in rows
    )


async def test_the_raw_files_are_the_per_query_numbers_and_the_comparison_never_the_rankings(
    tmp_path: Path,
) -> None:
    folder = await run_bakeoff(config_for(tmp_path), SPECS, FakeMeasure())
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    assert set(result["raw_files"]) == {
        "per_query_bge_small.jsonl",
        "per_query_arctic_s.jsonl",
        "per_query_bge_base.jsonl",
        "comparison.jsonl",
    }
    rows = [
        json.loads(line)
        for line in (folder / "raw" / "per_query_arctic_s.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["query_id"] for row in rows] == ["1", "2", "3"]
    assert [row["judged_top10"] for row in rows] == [3, 3, 0]
    assert rows[0]["hybrid_ndcg_at_10"] == 1.0


async def test_no_product_id_appears_anywhere_in_the_committed_result_folder(
    tmp_path: Path,
) -> None:
    folder = await run_bakeoff(config_for(tmp_path), SPECS, FakeMeasure())
    files = [path for path in folder.rglob("*") if path.is_file()]
    assert files
    for path in files:
        assert b"SECRET" not in path.read_bytes(), path


async def test_the_rankings_are_written_to_the_private_folder_and_the_result_holds_their_hash(
    tmp_path: Path,
) -> None:
    folder = await run_bakeoff(config_for(tmp_path), SPECS, FakeMeasure())
    private = tmp_path / "private" / "g11-bakeoff-validation" / folder.name
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    for spec in SPECS:
        path = private / f"rankings_{spec.name}.jsonl"
        data = path.read_bytes()
        assert b"B0SECRET" in data
        assert (
            result["config"][f"private_rankings_{spec.name}_sha256"]
            == hashlib.sha256(data).hexdigest()
        )
        assert b"\r" not in data
    first = json.loads(
        (private / "rankings_arctic_s.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert first == {
        "query_id": "1",
        "dense": ["B0SECRET1", "B0SECRET2", "B0SECRET6"],
        "bm25": ["B0SECRET1"],
        "hybrid": ["B0SECRET1", "B0SECRET2", "B0SECRET6"],
    }


def test_the_private_folder_is_ignored_by_git() -> None:
    ignored = (Path(__file__).parents[2] / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "benchmarks/private/" in [line.strip() for line in ignored]


async def test_a_second_run_in_the_same_minute_is_refused_and_leaves_no_private_files_behind(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    await run_bakeoff(config, SPECS, FakeMeasure())
    with pytest.raises(ResultExists):
        await run_bakeoff(config, SPECS, FakeMeasure())
    runs = sorted(p.name for p in (tmp_path / "private" / "g11-bakeoff-validation").iterdir())
    assert runs == ["20261028-0930"]


async def test_a_later_run_gets_its_own_private_folder(tmp_path: Path) -> None:
    await run_bakeoff(config_for(tmp_path), SPECS, FakeMeasure())
    await run_bakeoff(config_for(tmp_path, NOW + timedelta(minutes=1)), SPECS, FakeMeasure())
    runs = sorted(p.name for p in (tmp_path / "private" / "g11-bakeoff-validation").iterdir())
    assert runs == ["20261028-0930", "20261028-0931"]


async def test_models_that_were_asked_other_queries_are_refused_and_nothing_is_written(
    tmp_path: Path,
) -> None:
    class Odd(FakeMeasure):
        async def __call__(self, spec: ModelSpec) -> ModelRun:
            run = await super().__call__(spec)
            if spec.name == "arctic_s":
                rankings = dict(run.rankings)
                del rankings["3"]
                return ModelRun(spec, rankings, run.products, run.chunks, run.speed)
            return run

    with pytest.raises(ValueError, match="was not asked the same queries"):
        await run_bakeoff(config_for(tmp_path), SPECS, Odd())
    assert not (tmp_path / "results").exists() or not list(
        (tmp_path / "results").rglob("result.json")
    )
    assert not (tmp_path / "private").exists() or not any((tmp_path / "private").rglob("*.jsonl"))


async def test_no_models_or_the_same_name_twice_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model"):
        await run_bakeoff(config_for(tmp_path), (), FakeMeasure())
    with pytest.raises(ValueError, match="twice"):
        await run_bakeoff(config_for(tmp_path), (SPECS[0], SPECS[0]), FakeMeasure())


async def test_a_single_model_is_measured_and_has_nothing_to_be_compared_with(
    tmp_path: Path,
) -> None:
    folder = await run_bakeoff(config_for(tmp_path), SPECS[:1], FakeMeasure())
    rows = (folder / "raw" / "comparison.jsonl").read_text(encoding="utf-8")
    assert rows == ""


async def test_the_same_queries_in_another_order_are_refused_because_scores_are_paired(
    tmp_path: Path,
) -> None:
    class Shuffled(FakeMeasure):
        async def __call__(self, spec: ModelSpec) -> ModelRun:
            run = await super().__call__(spec)
            if spec.name == "arctic_s":
                reordered = {key: run.rankings[key] for key in ("3", "1", "2")}
                return ModelRun(spec, reordered, run.products, run.chunks, run.speed)
            return run

    with pytest.raises(ValueError, match="was not asked the same queries"):
        await run_bakeoff(config_for(tmp_path), SPECS, Shuffled())


async def test_the_prefix_of_every_model_is_in_the_config_so_a_result_says_what_was_asked(
    tmp_path: Path,
) -> None:
    prefixed = ModelSpec(
        "bge_small_p", "BAAI/bge-small-en-v1.5", 384, 1, query_prefix="Represent: "
    )
    quality = {**QUALITY, "bge_small_p": "perfect"}
    speed = {**SPEED, "bge_small_p": Speed(0.1, 0.01)}

    class Variants(FakeMeasure):
        async def __call__(self, spec: ModelSpec) -> ModelRun:
            self.calls.append(spec.name)
            return ModelRun(spec, rankings_of(quality[spec.name]), 500, 700, speed[spec.name])

    folder = await run_bakeoff(config_for(tmp_path), (SPECS[0], prefixed), Variants())
    config = json.loads((folder / "result.json").read_text(encoding="utf-8"))["config"]
    assert config["query_prefix_bge_small"] == ""
    assert config["query_prefix_bge_small_p"] == "Represent: "


# ---------------- the six models: three models, each plain and with the query prefix


def test_the_default_models_are_three_models_each_plain_and_with_the_prefix_current_one_first() -> (
    None
):
    names = [spec.name for spec in DEFAULT_MODELS]
    assert names == ["bge_small", "bge_small_p", "arctic_s", "arctic_s_p", "bge_base", "bge_base_p"]
    assert [spec.dense_model for spec in DEFAULT_MODELS[0::2]] == [
        "BAAI/bge-small-en-v1.5",
        "snowflake/snowflake-arctic-embed-s",
        "BAAI/bge-base-en-v1.5",
    ]
    for plain, prefixed in zip(DEFAULT_MODELS[0::2], DEFAULT_MODELS[1::2], strict=True):
        assert plain.query_prefix == ""
        assert prefixed.query_prefix == RETRIEVAL_QUERY_PREFIX
        assert prefixed.name == plain.name + "_p"
        # the same model: only the text put in front of the query differs
        assert (prefixed.dense_model, prefixed.dense_dim, prefixed.model_revision) == (
            plain.dense_model,
            plain.dense_dim,
            plain.model_revision,
        )


# ---------------- the hard queries are fixed once, from the current model


class BlindOnQuery1(FakeMeasure):
    """A model whose hybrid list for q1 has nothing judged in it, so q1 is hard FOR IT."""

    async def __call__(self, spec: ModelSpec) -> ModelRun:
        run = await super().__call__(spec)
        if spec.name != "blind":
            return run
        rankings = dict(run.rankings)
        rankings["1"] = RankedLists("1", ("X1",), ("X1",), ("X1",))
        return ModelRun(spec, rankings, run.products, run.chunks, run.speed)


BLIND = ModelSpec("blind", "BAAI/bge-small-en-v1.5", 384, 1)


async def test_every_model_is_scored_on_the_hard_queries_of_the_current_model(
    tmp_path: Path,
) -> None:
    SPEED["blind"] = Speed(0.1, 0.01)
    fake = BlindOnQuery1({**QUALITY, "blind": "perfect"})
    folder = await run_bakeoff(config_for(tmp_path), (*SPECS, BLIND), fake)
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    metrics = result["metrics"]
    # the current model has one hard query, q3. The blind model has two of its own (q1 and q3) but
    # is scored on the one set: one hard query, and the mean without it covers q1 and q2
    assert result["config"]["hard_query_source"] == "bge_small"
    assert result["config"]["hard_queries"] == 1
    assert metrics["bge_small_hard_queries"] == 1
    assert metrics["blind_hard_queries"] == 1
    assert metrics["blind_hybrid_ndcg_at_10_without_hard"] == pytest.approx((0.0 + 1.0) / 2)
    assert metrics["arctic_s_hybrid_ndcg_at_10_without_hard"] == pytest.approx(1.0)


async def test_the_per_query_file_says_which_queries_are_hard(tmp_path: Path) -> None:
    folder = await run_bakeoff(config_for(tmp_path), SPECS, FakeMeasure())
    rows = [
        json.loads(line)
        for line in (folder / "raw" / "per_query_arctic_s.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["hard"] for row in rows] == [False, False, True]


# ---------------- what the notes of the result must say


async def test_the_notes_say_how_many_candidates_are_compared_at_what_confidence(
    tmp_path: Path,
) -> None:
    folder = await run_bakeoff(config_for(tmp_path), SPECS, FakeMeasure())
    notes = json.loads((folder / "result.json").read_text(encoding="utf-8"))["notes"]
    # three models: two candidates against the current one
    assert "2 candidates are each compared with the current model at 98.3%" in notes
    assert "false SWITCH" in notes


async def test_the_notes_speak_of_one_candidate_in_the_singular(tmp_path: Path) -> None:
    folder = await run_bakeoff(config_for(tmp_path), SPECS[:2], FakeMeasure())
    notes = json.loads((folder / "result.json").read_text(encoding="utf-8"))["notes"]
    assert "1 candidate is compared with the current model at 98.3%" in notes


async def test_the_notes_point_to_the_measured_ratios_and_never_claim_equal_sizes(
    tmp_path: Path,
) -> None:
    folder = await run_bakeoff(config_for(tmp_path), SPECS, FakeMeasure())
    notes = json.loads((folder / "result.json").read_text(encoding="utf-8"))["notes"]
    assert "_embed_time_ratio" in notes
    assert "_query_latency_ratio" in notes
    assert "same size" not in notes.lower()
    assert "GB" not in notes


async def test_the_notes_say_how_the_speeds_were_measured(tmp_path: Path) -> None:
    folder = await run_bakeoff(config_for(tmp_path), SPECS, FakeMeasure())
    notes = json.loads((folder / "result.json").read_text(encoding="utf-8"))["notes"]
    assert "one model after the other" in notes
    assert "nothing else running" in notes
    assert "does not check" in notes
