"""Two gates. The C2 version gate: does the worker process an event or drop it as stale?
The G9 quality gate: does a rung of the ladder stay within 2 percent of the committed baseline?"""

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from eval.gate import (
    GATED_RUNGS,
    Baseline,
    BaselineMismatch,
    check,
    latest_result,
    load_result,
    main,
    propose_baseline,
)
from eval.ladder import run_ladder
from eval.results import RunResult, write_result
from services.embed.gate import GateState, VersionGateDecision, gate
from tests.unit.test_eval_loader import dataset, products
from tests.unit.test_ladder import (
    INDEXED,
    FakeEncoder,
    ladder_config,
    ladder_search,
    tiny_dataset,
)
from tests.unit.test_query_api import FakeSearch, hit

PROCEED = VersionGateDecision.VERSION_GATE_PROCEED
DROP = VersionGateDecision.VERSION_GATE_DROP


@pytest.mark.parametrize(
    ("version", "state", "expected"),
    [
        pytest.param(0, None, PROCEED, id="unknown document, version zero"),
        pytest.param(5, None, PROCEED, id="unknown document"),
        pytest.param(6, GateState(5, indexed=True), PROCEED, id="higher than an indexed version"),
        pytest.param(6, GateState(5, indexed=False), PROCEED, id="higher than a stuck version"),
        pytest.param(5, GateState(5, indexed=False), PROCEED, id="equal, not yet indexed: a retry"),
        pytest.param(5, GateState(5, indexed=True), DROP, id="equal and indexed: a duplicate"),
        pytest.param(0, GateState(0, indexed=True), DROP, id="duplicate at version zero"),
        pytest.param(4, GateState(5, indexed=True), DROP, id="lower than an indexed version"),
        pytest.param(4, GateState(5, indexed=False), DROP, id="lower than a stuck version"),
    ],
)
def test_gate_decision(
    version: int, state: GateState | None, expected: VersionGateDecision
) -> None:
    assert gate(version, state) is expected


def test_delete_is_gated_like_any_other_version() -> None:
    tombstone = GateState(current_version=5, indexed=True)  # a delete at version 5 is done
    assert gate(6, tombstone) is PROCEED  # a newer upsert may bring the product back
    assert gate(5, tombstone) is DROP  # the delete delivered again
    assert gate(3, tombstone) is DROP  # a delayed older upsert must not resurrect it


# ---------------- G9, the quality gate of the retrieval ladder

MANIFEST = "a" * 64
MODEL = "bge-small-en-v1.5_bm25-a256_1"


def baseline(**overrides: Any) -> Baseline:
    values: dict[str, Any] = {
        "dataset_manifest_sha256": MANIFEST,
        "model_version": MODEL,
        "ndcg_at_10": {"bm25": 0.5, "dense": 0.5, "hybrid": 0.5},
        "made_from": "benchmarks/results/g6-ladder-mini/20261025-0754",
    }
    return Baseline(**{**values, **overrides})


def result_with(tmp_path: Path, metrics: dict[str, float], minute: int = 0, **kw: Any) -> Path:
    """A real result folder written by write_result, with these metrics."""
    data = tmp_path / "data"
    if not (data / "manifest.json").exists():
        data.mkdir(parents=True, exist_ok=True)
        dataset(data, products(3))
    manifest_sha = kw.pop("manifest_sha", None)
    folder = write_result(
        tmp_path / "results",
        "g6-ladder-mini",
        command="make eval",
        environment_kind="local",
        dataset_directory=data,
        model_version=kw.pop("model_version", MODEL),
        config={},
        metrics=metrics,
        clock=lambda: datetime(2026, 10, 25, 9, minute, tzinfo=UTC),
    )
    if manifest_sha is not None:
        path = folder / "result.json"
        loaded = json.loads(path.read_text(encoding="utf-8"))
        loaded["dataset"]["manifest_sha256"] = manifest_sha
        path.write_text(json.dumps(loaded), encoding="utf-8")
    return folder


def metrics_of(bm25: float, dense: float, hybrid: float) -> dict[str, float]:
    return {
        "bm25_ndcg_at_10": bm25,
        "dense_ndcg_at_10": dense,
        "hybrid_ndcg_at_10": hybrid,
        "hybrid_chunk_ndcg_at_10": 0.1,
    }


def run_result(tmp_path: Path, metrics: dict[str, float], **kw: Any) -> RunResult:
    return load_result(result_with(tmp_path, metrics, **kw))


def matching(tmp_path: Path, metrics: dict[str, float]) -> RunResult:
    """A result of the same dataset as the baseline: the baseline takes its hash from it."""
    return run_result(tmp_path, metrics)


# ---------------- the rule


def test_the_gated_rungs_are_bm25_dense_and_hybrid_and_not_the_chunk_rung() -> None:
    assert GATED_RUNGS == ("bm25", "dense", "hybrid")


def test_a_drop_of_exactly_the_limit_passes_and_a_little_more_fails(tmp_path: Path) -> None:
    result = matching(tmp_path, metrics_of(0.4, 0.4, 0.4))
    base = baseline(dataset_manifest_sha256=result.dataset.manifest_sha256)
    # 0.5 down to 0.4 is exactly 20 percent
    assert check(base, result, max_relative_drop=0.2).passed
    worse = matching(tmp_path / "w", metrics_of(0.3999, 0.4, 0.4))
    assert not check(base, worse, max_relative_drop=0.2).passed


def test_an_improvement_passes_and_the_row_shows_the_change(tmp_path: Path) -> None:
    result = matching(tmp_path, metrics_of(0.55, 0.5, 0.6))
    base = baseline(dataset_manifest_sha256=result.dataset.manifest_sha256)
    outcome = check(base, result)
    assert outcome.passed
    rows = {row.rung: row for row in outcome.rows}
    assert rows["hybrid"].relative_change == pytest.approx(0.2)
    assert rows["dense"].relative_change == pytest.approx(0.0)


def test_each_gated_rung_is_judged_on_its_own(tmp_path: Path) -> None:
    # hybrid is fine, dense lost 10 percent: RRF must not hide a broken retriever
    result = matching(tmp_path, metrics_of(0.5, 0.45, 0.5))
    base = baseline(dataset_manifest_sha256=result.dataset.manifest_sha256)
    outcome = check(base, result)
    assert not outcome.passed
    assert {row.rung: row.passed for row in outcome.rows} == {
        "bm25": True,
        "dense": False,
        "hybrid": True,
    }


def test_the_chunk_rung_is_not_gated_even_when_it_collapses(tmp_path: Path) -> None:
    result = matching(tmp_path, {**metrics_of(0.5, 0.5, 0.5), "hybrid_chunk_ndcg_at_10": 0.0})
    base = baseline(dataset_manifest_sha256=result.dataset.manifest_sha256)
    outcome = check(base, result)
    assert outcome.passed
    assert [row.rung for row in outcome.rows] == list(GATED_RUNGS)


def test_a_rung_missing_from_the_result_fails(tmp_path: Path) -> None:
    result = matching(tmp_path, {"bm25_ndcg_at_10": 0.5, "dense_ndcg_at_10": 0.5})
    base = baseline(dataset_manifest_sha256=result.dataset.manifest_sha256)
    outcome = check(base, result)
    assert not outcome.passed
    hybrid = next(row for row in outcome.rows if row.rung == "hybrid")
    assert (hybrid.current, hybrid.passed) == (None, False)


def test_a_baseline_made_for_another_dataset_is_refused_not_failed(tmp_path: Path) -> None:
    result = matching(tmp_path, metrics_of(0.5, 0.5, 0.5))
    with pytest.raises(BaselineMismatch, match="new baseline"):
        check(baseline(dataset_manifest_sha256="b" * 64), result)


def test_a_baseline_made_for_another_model_version_is_refused(tmp_path: Path) -> None:
    result = matching(tmp_path, metrics_of(0.5, 0.5, 0.5))
    base = baseline(
        dataset_manifest_sha256=result.dataset.manifest_sha256, model_version="other_model_1"
    )
    with pytest.raises(BaselineMismatch, match="model version"):
        check(base, result)


# ---------------- the baseline file


def test_the_baseline_is_strict_about_its_fields() -> None:
    with pytest.raises(ValidationError):
        baseline(extra=1)
    with pytest.raises(ValidationError):
        baseline(ndcg_at_10={"bm25": 0.0, "dense": 0.5, "hybrid": 0.5})
    with pytest.raises(ValidationError):
        baseline(ndcg_at_10={"bm25": 0.5, "dense": 0.5})


def test_a_baseline_proposed_from_a_result_holds_the_three_values_and_where_they_came_from(
    tmp_path: Path,
) -> None:
    folder = result_with(tmp_path, metrics_of(0.61, 0.66, 0.67))
    proposed = propose_baseline(load_result(folder), made_from=str(folder))
    assert proposed.ndcg_at_10 == {"bm25": 0.61, "dense": 0.66, "hybrid": 0.67}
    assert proposed.model_version == MODEL
    assert proposed.made_from == str(folder)
    assert len(proposed.dataset_manifest_sha256) == 64


def test_a_result_without_a_gated_rung_cannot_make_a_baseline(tmp_path: Path) -> None:
    result = run_result(tmp_path, {"bm25_ndcg_at_10": 0.5})
    with pytest.raises(ValueError, match="dense"):
        propose_baseline(result, made_from="x")


def test_latest_result_is_the_newest_folder_by_name(tmp_path: Path) -> None:
    result_with(tmp_path, metrics_of(0.5, 0.5, 0.5), minute=1)
    newest = result_with(tmp_path, metrics_of(0.5, 0.5, 0.5), minute=7)
    result_with(tmp_path, metrics_of(0.5, 0.5, 0.5), minute=3)
    assert latest_result(tmp_path / "results", "g6-ladder-mini") == newest


def test_latest_result_says_what_to_run_when_there_is_none(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make eval"):
        latest_result(tmp_path / "results", "g6-ladder-mini")


# ---------------- the command


def test_the_command_exits_0_when_the_gate_passes_and_prints_a_row_per_rung(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = result_with(tmp_path, metrics_of(0.5, 0.5, 0.5))
    path = tmp_path / "baseline.json"
    path.write_text(
        propose_baseline(load_result(folder), made_from="x").model_dump_json(indent=2),
        encoding="utf-8",
    )
    assert main(["check", "--result", str(folder), "--baseline", str(path)]) == 0
    printed = capsys.readouterr().out
    assert all(rung in printed for rung in GATED_RUNGS)


def test_the_command_exits_1_when_a_rung_dropped_more_than_2_percent(tmp_path: Path) -> None:
    good = result_with(tmp_path, metrics_of(0.5, 0.5, 0.5), minute=1)
    bad = result_with(tmp_path, metrics_of(0.5, 0.5, 0.485), minute=2)
    path = tmp_path / "baseline.json"
    path.write_text(
        propose_baseline(load_result(good), made_from="x").model_dump_json(indent=2),
        encoding="utf-8",
    )
    # 0.485 against 0.5 is 3 percent
    assert main(["check", "--result", str(bad), "--baseline", str(path)]) == 1


def test_the_command_exits_2_and_names_make_baseline_when_there_is_no_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = result_with(tmp_path, metrics_of(0.5, 0.5, 0.5))
    assert main(["check", "--result", str(folder), "--baseline", str(tmp_path / "none.json")]) == 2
    assert "make baseline" in capsys.readouterr().err


def test_the_propose_command_writes_the_file_with_unix_line_endings_and_never_replaces(
    tmp_path: Path,
) -> None:
    folder = result_with(tmp_path, metrics_of(0.61, 0.66, 0.67))
    out = tmp_path / "base" / "ladder-mini.json"
    assert main(["propose", "--result", str(folder), "--out", str(out)]) == 0
    assert b"\r" not in out.read_bytes()
    assert json.loads(out.read_text(encoding="utf-8"))["ndcg_at_10"]["hybrid"] == 0.67
    # a baseline changes by a reviewed commit, not by a rerun
    assert main(["propose", "--result", str(folder), "--out", str(out)]) == 2


# ---------------- a seeded regression, through the real ladder


async def test_a_broken_ranking_made_by_the_ladder_fails_the_gate_against_a_good_baseline(
    tmp_path: Path,
) -> None:
    tiny_dataset(tmp_path / "mini")
    good_run = await run_ladder_in(tmp_path, ladder_search(), minute=1)
    base = propose_baseline(load_result(good_run), made_from=str(good_run))
    assert check(base, load_result(good_run)).passed

    # the same query answered with the best products at the bottom, for both retrievers
    broken = FakeSearch()
    broken.grouped["dense"] = [hit("P2"), hit("P1"), hit("P0")]
    broken.grouped["bm25"] = [hit("P2"), hit("P0"), hit("P3")]
    broken.chunks["dense"] = [hit("P2", 0), hit("P1", 0), hit("P0", 0)]
    broken.chunks["bm25"] = [hit("P2", 0), hit("P0", 0), hit("P3", 0)]
    bad_run = await run_ladder_in(tmp_path, broken, minute=2)
    outcome = check(base, load_result(bad_run))
    assert not outcome.passed
    assert not any(row.passed for row in outcome.rows if row.rung == "dense")


async def run_ladder_in(tmp_path: Path, search: FakeSearch, minute: int) -> Path:
    config = ladder_config(tmp_path)
    config = replace(config, clock=lambda: datetime(2026, 10, 25, 9, minute, tzinfo=UTC))
    return await run_ladder(config, FakeEncoder(), search, INDEXED)  # type: ignore[arg-type]
