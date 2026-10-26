"""The distractor ladder: how nDCG falls as products that are not answers are added."""

import dataclasses
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval.dataset import (
    JudgementRow,
    Product,
    QueryRow,
    Selection,
    SelectionConfig,
    stable_rank,
    write_dataset,
)
from eval.distractor_ladder import (
    DistractorLadderConfig,
    Measurement,
    climb,
    is_flat,
    ladder_collection_name,
    rank_distractors,
    run_distractor_ladder,
)
from eval.metrics import PairedGain
from services.query.settings import QuerySettings

NOW = datetime(2026, 10, 26, 9, 30, tzinfo=UTC)
QUERIES = 40
# the band used in the tests below, a test value: the real one is written by the owner in
# docs/DECISIONS.md before the real run
EPSILON = 0.01


def make_dataset(directory: Path, distractors: int = 30, judged: int = 3) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    items = [
        Product(f"J{n:02d}", "us", f"judged {n}", None, None, None, None) for n in range(judged)
    ]
    items += [
        Product(f"D{n:03d}", "us", f"distractor {n}", None, None, None, None)
        for n in range(distractors)
    ]
    selection = Selection(
        (QueryRow(1, "mouse", "validation"),),
        tuple(JudgementRow(1, f"J{n:02d}", "E") for n in range(judged)),
        tuple(sorted(item.product_id for item in items)),
    )
    write_dataset(directory, selection, items, SelectionConfig(seed=1), 1)


# ---------------- the order of the distractors


def test_the_distractors_are_every_product_that_is_not_judged_in_the_seeded_order(
    tmp_path: Path,
) -> None:
    make_dataset(tmp_path / "set")
    judged, ranked = rank_distractors(tmp_path / "set", seed=7)
    assert judged == {"J00", "J01", "J02"}
    assert len(ranked) == 30
    assert ranked == sorted(ranked, key=lambda p: (stable_rank(7, "distractor", p), p))
    assert not judged & set(ranked)


def test_another_seed_gives_another_order_and_the_same_seed_the_same_order(tmp_path: Path) -> None:
    make_dataset(tmp_path / "set")
    assert rank_distractors(tmp_path / "set", 7)[1] == rank_distractors(tmp_path / "set", 7)[1]
    assert rank_distractors(tmp_path / "set", 7)[1] != rank_distractors(tmp_path / "set", 8)[1]


def test_a_smaller_corpus_is_inside_a_bigger_one_because_both_take_a_prefix(
    tmp_path: Path,
) -> None:
    make_dataset(tmp_path / "set")
    _, ranked = rank_distractors(tmp_path / "set", 7)
    assert set(ranked[:10]) < set(ranked[:20])


# ---------------- the collection the ladder owns


def test_the_collection_of_a_size_is_the_configured_name_plus_the_ladder_suffix() -> None:
    assert ladder_collection_name("chunks_esci", 500) == "chunks_esci_ladder_d500"
    assert ladder_collection_name("chunks_esci", 64000) == "chunks_esci_ladder_d64000"


def test_the_name_of_a_ladder_collection_is_never_the_configured_collection() -> None:
    # the ladder deletes the collections it names, the one the service reads must never be one
    for size in (1, 500, 64000):
        assert ladder_collection_name("chunks_esci", size) != "chunks_esci"
        assert ladder_collection_name("chunks_esci", size).startswith("chunks_esci_ladder_d")


def test_a_size_below_one_has_no_collection_name() -> None:
    with pytest.raises(ValueError, match="size"):
        ladder_collection_name("chunks_esci", 0)


# ---------------- is a step flat: the whole interval must lie inside the band


def gain(mean: float, low: float, high: float) -> PairedGain:
    return PairedGain(mean=mean, low=low, high=high, sd=0.1)


def test_a_narrow_interval_around_zero_is_flat() -> None:
    assert is_flat(gain(0.0, -0.003, 0.003), epsilon=0.01)


def test_a_wide_interval_that_holds_zero_is_not_flat() -> None:
    # "cannot tell it from zero" is not "no change": this is the case the first rule got wrong,
    # its mean was 0.001 and its interval held zero, so it called a step flat that may fall by 0.03
    assert not is_flat(gain(0.001, -0.03, 0.032), epsilon=0.01)


def test_a_narrow_interval_off_to_one_side_is_not_flat() -> None:
    assert not is_flat(gain(-0.025, -0.03, -0.02), epsilon=0.01)
    assert not is_flat(gain(0.025, 0.02, 0.03), epsilon=0.01)


def test_an_interval_that_ends_exactly_on_the_edge_of_the_band_is_flat() -> None:
    # pinned: the band is closed, low >= -epsilon and high <= +epsilon
    assert is_flat(gain(0.0, -0.01, 0.01), epsilon=0.01)


def test_an_interval_that_goes_a_hair_past_either_edge_is_not_flat() -> None:
    assert not is_flat(gain(0.0, -0.0100001, 0.01), epsilon=0.01)
    assert not is_flat(gain(0.0, -0.01, 0.0100001), epsilon=0.01)


def test_a_tiny_mean_with_a_very_wide_interval_is_not_flat() -> None:
    assert not is_flat(gain(0.0, -0.2, 0.2), epsilon=0.01)


def test_the_band_is_a_setting_a_smaller_band_turns_a_flat_step_into_a_step_that_is_not() -> None:
    narrow = gain(0.0, -0.003, 0.003)
    assert is_flat(narrow, epsilon=0.01)
    assert not is_flat(narrow, epsilon=0.002)


def test_epsilon_has_no_default_anywhere_so_it_cannot_be_forgotten() -> None:
    assert inspect.signature(is_flat).parameters["epsilon"].default is inspect.Parameter.empty
    assert inspect.signature(climb).parameters["epsilon"].default is inspect.Parameter.empty
    (field,) = (f for f in dataclasses.fields(DistractorLadderConfig) if f.name == "epsilon")
    assert field.default is dataclasses.MISSING
    assert field.default_factory is dataclasses.MISSING
    with pytest.raises(TypeError):
        DistractorLadderConfig(  # type: ignore[call-arg]
            dataset_directory=Path("x"),
            settings=QuerySettings(),
            first_size=500,
            largest_size=1000,
        )


# ---------------- climbing the ladder with a made up measurement


def curve_of(kinds: str) -> list[tuple[float, ...]]:
    """nDCG per query at each size, one letter per step.

    d  a clear drop: every query loses 0.05, so the interval is the single point -0.05
    f  flat: +0.01 on even queries and -0.01 on odd ones, mean 0, interval about +-0.003
    w  wide: +0.1 on even queries and -0.1 on odd ones, mean 0, interval about +-0.03
    s  a slow steady fall: every query loses 0.002, the interval is the single point -0.002
    """
    current = [0.8 + 0.001 * i for i in range(QUERIES)]
    arrays = [tuple(current)]
    for kind in kinds:
        if kind == "d":
            current = [value - 0.05 for value in current]
        elif kind == "s":
            current = [value - 0.002 for value in current]
        else:
            size = 0.01 if kind == "f" else 0.1
            current = [value + (size if i % 2 == 0 else -size) for i, value in enumerate(current)]
        arrays.append(tuple(current))
    return arrays


class Fake:
    def __init__(self, kinds: str) -> None:
        self.arrays = curve_of(kinds)
        self.asked: list[int] = []

    async def measure(self, size: int) -> Measurement:
        self.asked.append(size)
        index = len(self.asked) - 1
        return Measurement(
            size=size,
            products=size + 100,
            chunks=int((size + 100) * 1.3),
            hybrid_ndcg_per_query=self.arrays[index],
            metrics={"hybrid_ndcg_at_10": sum(self.arrays[index]) / QUERIES},
        )


async def run_climb(fake: Fake, largest: int = 64000, epsilon: float = EPSILON):  # type: ignore[no-untyped-def]
    return await climb(
        fake.measure, first_size=500, largest_size=largest, resamples=300, seed=3, epsilon=epsilon
    )


async def test_the_climb_stops_after_two_flat_steps_in_a_row_and_names_the_smaller_size() -> None:
    # hand check, epsilon 0.01: steps are drop, drop, flat, flat. A drop has the interval
    # [-0.05, -0.05], outside the band. A flat step has about [-0.003, +0.003], inside it.
    # The two flat steps are 2000 to 4000 and 4000 to 8000, so the plateau is 2000, and 8000 is the
    # last size measured: 16000 is never built.
    fake = Fake("ddff")
    curve = await run_climb(fake)
    assert curve.plateau == 2000
    assert curve.stopped == "flat twice"
    assert fake.asked == [500, 1000, 2000, 4000, 8000]
    assert [step.flat for step in curve.steps] == [False, False, True, True]


async def test_one_flat_step_alone_does_not_stop_the_climb() -> None:
    # hand check: flat, drop, flat, flat: the first flat step is followed by a drop, so only the
    # last two (2000 to 4000, 4000 to 8000) are two in a row, the plateau is 2000 again
    fake = Fake("fdff")
    curve = await run_climb(fake)
    assert curve.plateau == 2000
    assert fake.asked == [500, 1000, 2000, 4000, 8000]
    assert [step.flat for step in curve.steps] == [True, False, True, True]


async def test_wide_steps_whose_mean_is_zero_never_stop_the_climb() -> None:
    # the first rule stopped here: mean 0, interval holds 0. The band rule keeps building,
    # because the interval of about +-0.03 does not fit inside +-0.01
    fake = Fake("wwww")
    curve = await run_climb(fake, largest=8000)
    assert curve.plateau is None
    assert curve.stopped == "no more distractors"
    assert fake.asked == [500, 1000, 2000, 4000, 8000]
    assert not any(step.flat for step in curve.steps)


async def test_a_slow_steady_fall_is_flat_only_if_it_fits_inside_the_band() -> None:
    # a fall of 0.002 per doubling is inside +-0.01 (flat) and outside +-0.001 (a fall that counts)
    assert (await run_climb(Fake("ssss"), epsilon=0.01)).plateau == 500
    never = await run_climb(Fake("ssss"), largest=8000, epsilon=0.001)
    assert never.plateau is None


async def test_a_curve_that_never_flattens_climbs_to_the_largest_size_and_says_so() -> None:
    fake = Fake("ddddddd")
    curve = await run_climb(fake, largest=8000)
    assert curve.plateau is None
    assert curve.stopped == "no more distractors"
    assert fake.asked == [500, 1000, 2000, 4000, 8000]


async def test_the_sizes_double_and_never_pass_the_largest() -> None:
    fake = Fake("dddd")
    await run_climb(fake, largest=3000)
    assert fake.asked == [500, 1000, 2000]


async def test_a_first_size_above_the_largest_is_refused() -> None:
    fake = Fake("d")
    with pytest.raises(ValueError, match="first size"):
        await climb(
            fake.measure, first_size=500, largest_size=100, resamples=100, seed=1, epsilon=EPSILON
        )
    assert fake.asked == []


async def test_a_measurement_over_other_queries_than_the_one_before_is_refused() -> None:
    fake = Fake("d")
    fake.arrays[1] = fake.arrays[1][:-1]
    with pytest.raises(ValueError, match="same queries"):
        await run_climb(fake)


# ---------------- the result of a run


def config_for(tmp_path: Path, largest: int, **changes: object) -> DistractorLadderConfig:
    values: dict[str, object] = {
        "dataset_directory": tmp_path / "set",
        "settings": QuerySettings(),
        "results_root": tmp_path / "results",
        "first_size": 500,
        "largest_size": largest,
        "epsilon": EPSILON,
        "resamples": 200,
        "clock": lambda: NOW,
    }
    return DistractorLadderConfig(**{**values, **changes})  # type: ignore[arg-type]


async def test_a_run_writes_one_result_with_the_curve_every_step_and_the_plateau(
    tmp_path: Path,
) -> None:
    make_dataset(tmp_path / "set", distractors=40)
    fake = Fake("ddff")
    folder = await run_distractor_ladder(config_for(tmp_path, 64000), fake.measure)
    assert folder.parent.name == "g12-distractor-ladder-set"
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    assert result["config"]["plateau"] == 2000
    assert result["config"]["stopped"] == "flat twice"
    assert result["config"]["epsilon"] == EPSILON
    assert "max_mean_change" not in result["config"]
    metrics = result["metrics"]
    assert metrics["plateau_distractors"] == 2000
    assert metrics["d500_hybrid_ndcg_at_10"] > metrics["d8000_hybrid_ndcg_at_10"]
    # every step is in the result with its interval, not only the plateau label
    for first, second in ((500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)):
        for part in ("mean", "low", "high"):
            assert f"step_{first}_{second}_{part}" in metrics
    assert metrics["step_500_1000_mean"] == pytest.approx(-0.05)
    # on a flat step the interval has a width: low is below zero, high above, the mean between
    low, mean, high = (metrics[f"step_2000_4000_{part}"] for part in ("low", "mean", "high"))
    assert low < 0 < high
    assert low <= mean <= high
    assert set(result["raw_files"]) == {"curve.jsonl", "steps.jsonl", "per_query_hybrid.jsonl"}
    lines = (folder / "raw" / "curve.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["size"] for line in lines] == [500, 1000, 2000, 4000, 8000]
    assert b"\r" not in (folder / "raw" / "curve.jsonl").read_bytes()


async def test_the_per_query_table_has_a_row_per_size_with_every_query_for_hybrid_only(
    tmp_path: Path,
) -> None:
    make_dataset(tmp_path / "set", distractors=40)
    fake = Fake("ddff")
    folder = await run_distractor_ladder(config_for(tmp_path, 64000), fake.measure)
    rows = [
        json.loads(line)
        for line in (folder / "raw" / "per_query_hybrid.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["size"] for row in rows] == [500, 1000, 2000, 4000, 8000]
    assert all(len(row["hybrid_ndcg_at_10"]) == QUERIES for row in rows)
    assert set(rows[0]) == {"size", "hybrid_ndcg_at_10"}


async def test_every_step_with_its_interval_and_its_verdict_is_in_the_steps_file(
    tmp_path: Path,
) -> None:
    make_dataset(tmp_path / "set", distractors=40)
    folder = await run_distractor_ladder(config_for(tmp_path, 64000), Fake("ddff").measure)
    steps = [
        json.loads(line)
        for line in (folder / "raw" / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(s["from"], s["to"], s["flat"]) for s in steps] == [
        (500, 1000, False),
        (1000, 2000, False),
        (2000, 4000, True),
        (4000, 8000, True),
    ]
    assert all({"mean", "low", "high", "sd"} <= set(s) for s in steps)
    assert all(s["low"] >= -EPSILON and s["high"] <= EPSILON for s in steps if s["flat"])


async def test_a_run_that_never_flattens_has_no_plateau_metric_and_says_none_in_the_config(
    tmp_path: Path,
) -> None:
    make_dataset(tmp_path / "set", distractors=40)
    folder = await run_distractor_ladder(
        config_for(tmp_path, 4000, resamples=100), Fake("wwww").measure
    )
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    assert result["config"]["plateau"] is None
    assert result["config"]["stopped"] == "no more distractors"
    assert "plateau_distractors" not in result["metrics"]


def test_the_config_has_defaults_for_the_resamples_and_the_seed_but_not_for_epsilon() -> None:
    config = DistractorLadderConfig(
        dataset_directory=Path("x"),
        settings=QuerySettings(),
        first_size=500,
        largest_size=1000,
        epsilon=0.01,
    )
    assert (config.resamples, config.seed) == (1000, 7)
