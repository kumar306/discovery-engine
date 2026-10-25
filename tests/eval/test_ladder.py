"""G6: the ladder on the real index of ESCI-mini.

needs the dev stack with ESCI-mini loaded (make load-mini), marker eval, so make test skips it
    make eval
There is no quality threshold here: the numbers are for the result file and the gate (G9).
This test only proves that the whole path runs on the real stack and holds all four rungs.
"""

import json
from pathlib import Path

import pytest

from eval.ladder import RUNGS, LadderConfig, run_on_dev_stack
from services.query.settings import QuerySettings

pytestmark = pytest.mark.eval

DATASET = Path("data/processed/mini")


async def test_the_ladder_runs_on_the_real_index_and_scores_every_rung(tmp_path: Path) -> None:
    assert (DATASET / "loaded.json").exists(), "no loaded.json, run make load-mini first"
    folder = await run_on_dev_stack(
        LadderConfig(DATASET, QuerySettings(), results_root=tmp_path, resamples=200)
    )
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    metrics = result["metrics"]
    assert metrics["queries_scored"] == 50
    for rung in RUNGS:
        # a ranking that finds nothing relevant would score 0, so a positive score proves the rung
        # returned products that the judges know, no particular quality is asserted
        assert 0.0 < metrics[f"{rung}_ndcg_at_10"] <= 1.0
        assert 0.0 < metrics[f"{rung}_recall_at_100"] <= 1.0
        assert (folder / "raw" / f"{rung}.txt").stat().st_size > 0
