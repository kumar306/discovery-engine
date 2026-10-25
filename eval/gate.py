"""G9, the quality gate: fail when a rung of the ladder gets worse than the committed baseline.

The baseline (benchmarks/baseline/ladder-mini.json) holds the nDCG@10 of the three rungs that are
gated, bm25, dense and hybrid, on the 50 test queries of ESCI-mini. The gate reads the newest
result of the ladder and fails when a rung lost more than 2 percent of its baseline value.
Each rung is judged alone: the fusion of the hybrid rung could hide a retriever that broke.
The chunk rung (the assistant's path) is measured but not gated yet.

The baseline changes only by a reviewed commit: `make baseline` writes a proposal and refuses to
replace a file that exists, a person reads the numbers and commits.

    python -m eval.gate check   [--result FOLDER] [--baseline FILE]   exit 0 pass, 1 fail, 2 refused
    python -m eval.gate propose [--result FOLDER] [--out FILE]
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from eval.results import RunResult

# the rungs the gate watches, in the order they are printed
GATED_RUNGS: tuple[str, ...] = ("bm25", "dense", "hybrid")
RESULTS_ROOT = Path("benchmarks/results")
BENCHMARK = "g6-ladder-mini"
BASELINE_PATH = Path("benchmarks/baseline/ladder-mini.json")


class BaselineMismatch(Exception):
    """The baseline was made for another dataset or another model, comparing would mean nothing."""


class Baseline(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    # which dataset (the sha256 of its manifest) and which model the numbers belong to
    dataset_manifest_sha256: str
    model_version: str
    # rung -> nDCG@10, exactly the gated rungs, all above zero
    ndcg_at_10: dict[str, float]
    # the result folder the numbers were taken from, so a reader can look at the rankings
    made_from: str

    @field_validator("ndcg_at_10")
    @classmethod
    def _exactly_the_gated_rungs(cls, values: dict[str, float]) -> dict[str, float]:
        if set(values) != set(GATED_RUNGS):
            raise ValueError(f"the baseline needs exactly the rungs {list(GATED_RUNGS)}")
        if any(value <= 0.0 for value in values.values()):
            raise ValueError("a baseline value must be above zero")
        return values


@dataclass(frozen=True)
class GateRow:
    rung: str
    baseline: float
    # None when the result has no number for this rung
    current: float | None
    relative_change: float | None
    passed: bool


@dataclass(frozen=True)
class GateResult:
    passed: bool
    rows: tuple[GateRow, ...]


def load_result(folder: Path) -> RunResult:
    return RunResult.model_validate_json((folder / "result.json").read_text(encoding="utf-8"))


def latest_result(results_root: Path, benchmark_id: str) -> Path:
    """The newest result folder of a benchmark, folder names are UTC times so names sort by age."""
    folders = sorted(
        path
        for path in (results_root / benchmark_id).glob("*")
        if path.is_dir() and (path / "result.json").exists()
    )
    if not folders:
        raise FileNotFoundError(f"no result under {results_root / benchmark_id}, run make eval")
    return folders[-1]


def check(baseline: Baseline, result: RunResult, *, max_relative_drop: float = 0.02) -> GateResult:
    """Compare the gated rungs of a ladder result with the baseline."""
    if baseline.dataset_manifest_sha256 != result.dataset.manifest_sha256:
        raise BaselineMismatch(
            "the baseline was made for another dataset, a changed dataset needs a new baseline "
            "(make baseline, read it, commit it)"
        )
    if baseline.model_version != result.dataset.model_version:
        raise BaselineMismatch(
            f"the baseline is for model version {baseline.model_version} and the result for "
            f"{result.dataset.model_version}, a changed model version needs a new baseline "
            "(make baseline, read it, commit it)"
        )
    rows: list[GateRow] = []
    for rung in GATED_RUNGS:
        expected = baseline.ndcg_at_10[rung]
        current = result.metrics.get(f"{rung}_ndcg_at_10")
        if current is None:
            rows.append(GateRow(rung, expected, None, None, False))
            continue
        # a drop of exactly the limit passes: the rung may keep (1 - limit) of its baseline value
        rows.append(
            GateRow(
                rung,
                expected,
                current,
                (current - expected) / expected,
                current >= expected * (1.0 - max_relative_drop),
            )
        )
    return GateResult(all(row.passed for row in rows), tuple(rows))


def propose_baseline(result: RunResult, *, made_from: str) -> Baseline:
    """The baseline that a result would make, to be read and committed by a person."""
    values: dict[str, float] = {}
    for rung in GATED_RUNGS:
        value = result.metrics.get(f"{rung}_ndcg_at_10")
        if value is None:
            raise ValueError(f"the result has no number for the {rung} rung")
        values[rung] = value
    return Baseline(
        dataset_manifest_sha256=result.dataset.manifest_sha256,
        model_version=result.dataset.model_version,
        ndcg_at_10=values,
        made_from=made_from,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="the quality gate of the retrieval ladder")
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check", help="compare a ladder result with the baseline")
    checking.add_argument("--result", type=Path, default=None, help="default: the newest result")
    checking.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    proposing = commands.add_parser("propose", help="write a proposed baseline from a result")
    proposing.add_argument("--result", type=Path, default=None, help="default: the newest result")
    proposing.add_argument("--out", type=Path, default=BASELINE_PATH)
    arguments = parser.parse_args(argv)

    folder = arguments.result or latest_result(RESULTS_ROOT, BENCHMARK)
    result = load_result(folder)

    if arguments.command == "propose":
        if arguments.out.exists():
            print(
                f"{arguments.out} exists; a baseline is replaced by a reviewed commit, delete the "
                "file on purpose first",
                file=sys.stderr,
            )
            return 2
        proposed = propose_baseline(result, made_from=str(folder).replace("\\", "/"))
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        # unix line endings on every machine, like every committed file
        arguments.out.write_text(
            proposed.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"proposed baseline written to {arguments.out}, read it, then commit it")
        return 0

    if not arguments.baseline.exists():
        print(
            f"no baseline at {arguments.baseline}, run make baseline, read it and commit it",
            file=sys.stderr,
        )
        return 2
    baseline = Baseline.model_validate_json(arguments.baseline.read_text(encoding="utf-8"))
    try:
        outcome = check(baseline, result)
    except BaselineMismatch as mismatch:
        print(str(mismatch), file=sys.stderr)
        return 2
    print(f"gate on {folder}")
    for row in outcome.rows:
        shown = "missing" if row.current is None else f"{row.current:.4f}"
        change = "" if row.relative_change is None else f" ({row.relative_change:+.2%})"
        print(
            f"  {row.rung:<7} baseline {row.baseline:.4f}  now {shown}{change}  "
            f"{'ok' if row.passed else 'FAIL'}"
        )
    print("gate " + ("passed" if outcome.passed else "FAILED"))
    return 0 if outcome.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
