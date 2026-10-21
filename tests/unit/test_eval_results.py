"""The result writer: what a run folder contains, and that it is never half written."""

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from eval.results import ResultExists, RunResult, write_result
from tests.unit.test_eval_loader import dataset, products

NOW = datetime(2026, 10, 19, 9, 30, 15, tzinfo=UTC)
METRICS = {"ndcg@10": 0.5, "recall@100": 0.75, "mrr@10": 0.25}
FIELDS = {
    "benchmark_id",
    "date_utc",
    "command",
    "environment",
    "dataset",
    "config",
    "metrics",
    "raw_files",
    "notes",
}


def write(tmp_path: Path, root: str = "results", **overrides: Any) -> Path:
    data = tmp_path / "data"
    if not (data / "manifest.json").exists():
        data.mkdir(exist_ok=True)
        dataset(data, products(3))
    arguments: dict[str, Any] = {
        "command": "make eval",
        "environment_kind": "local",
        "dataset_directory": data,
        "model_version": "bge-small-en-v1.5_bm25-a256_1",
        "config": {"k": 10},
        "metrics": METRICS,
        "clock": lambda: NOW,
    }
    arguments.update(overrides)
    return write_result(tmp_path / root, "quality-ladder", **arguments)


def result_of(directory: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    return loaded


def test_the_folder_is_named_by_minute_and_holds_the_result_and_the_raw_files(
    tmp_path: Path,
) -> None:
    directory = write(tmp_path)
    assert directory == tmp_path / "results" / "quality-ladder" / "20261019-0930"
    assert sorted(path.name for path in directory.iterdir()) == ["raw", "result.json"]


def test_result_json_has_every_required_field(tmp_path: Path) -> None:
    directory = write(tmp_path)
    loaded = result_of(directory)
    assert set(loaded) == FIELDS
    RunResult.model_validate_json((directory / "result.json").read_text(encoding="utf-8"))
    assert (loaded["command"], loaded["metrics"]) == ("make eval", METRICS)
    assert loaded["environment"] == {"kind": "local"}
    assert loaded["date_utc"] == "2026-10-19T09:30:15+00:00"


def test_the_same_inputs_give_the_same_bytes(tmp_path: Path) -> None:
    first = write(tmp_path, "one")
    second = write(tmp_path, "two")
    assert (first / "result.json").read_bytes() == (second / "result.json").read_bytes()


def test_the_dataset_is_recorded_from_its_manifest(tmp_path: Path) -> None:
    directory = write(tmp_path)
    recorded = result_of(directory)["dataset"]
    manifest_bytes = (tmp_path / "data" / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert recorded["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert recorded["dataset_revision"] == manifest["dataset_revision"]
    assert recorded["counts"] == manifest["counts"]
    assert recorded["model_version"] == "bge-small-en-v1.5_bm25-a256_1"


@pytest.mark.parametrize("marker", ["none", "same", "other"])
def test_the_loaded_marker_counts_only_when_it_names_this_manifest(
    tmp_path: Path, marker: str
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    dataset(data, products(3))
    manifest_hash = hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest()
    if marker != "none":
        recorded = manifest_hash if marker == "same" else "f" * 64
        (data / "loaded.json").write_text(
            json.dumps({"manifest_sha256": recorded}), encoding="utf-8"
        )
    loaded_from = result_of(write(tmp_path))["dataset"]["index_loaded_from"]
    assert loaded_from == (manifest_hash if marker == "same" else None)


def test_raw_files_are_copied_and_listed_with_their_checksums(tmp_path: Path) -> None:
    source = tmp_path / "run.trec"
    source.write_text("1 Q0 P1 1 3.5 bm25\n", encoding="utf-8")
    directory = write(tmp_path, raw_files=[source])
    assert (directory / "raw" / "run.trec").read_bytes() == source.read_bytes()
    assert result_of(directory)["raw_files"] == {
        "run.trec": hashlib.sha256(source.read_bytes()).hexdigest()
    }


def test_two_raw_files_with_one_name_are_refused(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first, second = tmp_path / "a" / "run.trec", tmp_path / "b" / "run.trec"
    first.write_text("x", encoding="utf-8")
    second.write_text("y", encoding="utf-8")
    with pytest.raises(ValueError, match="same name"):
        write(tmp_path, raw_files=[first, second])


def test_an_existing_result_is_never_overwritten(tmp_path: Path) -> None:
    directory = write(tmp_path)
    before = (directory / "result.json").read_bytes()
    with pytest.raises(ResultExists):
        write(tmp_path, metrics={"ndcg@10": 0.9})
    assert (directory / "result.json").read_bytes() == before


def test_a_failed_write_leaves_no_folder_behind(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        write(tmp_path, raw_files=[tmp_path / "missing.trec"])
    assert list((tmp_path / "results" / "quality-ladder").iterdir()) == []


def test_a_metric_that_is_not_a_number_is_refused_and_nothing_is_written(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        write(tmp_path, metrics={"ndcg@10": math.nan})
    assert list((tmp_path / "results" / "quality-ladder").iterdir()) == []


def test_a_benchmark_id_that_could_leave_the_results_folder_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="plain name"):
        write_result(
            tmp_path / "results",
            "../x",
            command="c",
            environment_kind="local",
            dataset_directory=tmp_path,
            model_version="m",
            config={},
            metrics={},
            clock=lambda: NOW,
        )


def test_result_json_uses_unix_line_endings_so_a_checksum_survives_a_commit_on_windows(
    tmp_path: Path,
) -> None:
    # git turns windows line endings into unix ones when a file is committed, a hash taken
    # before that would no longer match the committed file
    directory = write(tmp_path)
    assert b"\r" not in (directory / "result.json").read_bytes()
