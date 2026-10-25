"""G9: the quality job on the real dev stack and the real index of ESCI-mini.

needs the dev stack with ESCI-mini loaded (make load-mini), marker eval, so make test skips it
    pytest -m eval tests/eval/test_quality_job.py -o addopts=""
The collection of the dev stack holds exactly the mini set, so the job reuses it, runs the ladder
and checks the gate against the committed baseline.
"""

from pathlib import Path

import pytest

from eval.quality_job import run_quality_job

pytestmark = pytest.mark.eval

DATASET = Path("data/processed/mini")


async def test_the_job_reuses_the_loaded_index_and_the_gate_passes(tmp_path: Path) -> None:
    assert (DATASET / "manifest.json").exists(), "no mini dataset, run make data first"
    assert await run_quality_job(DATASET, threads=2, results_root=tmp_path) == 0
