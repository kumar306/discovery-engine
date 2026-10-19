"""Write one folder per evaluation run: the numbers and what they were measured on.

result.json says which dataset (the hash of its manifest), which model and which command produced
the metrics, and lists the sha256 of every raw file that was copied next to it. The folder is
built under a .partial name and renamed when complete, so a crash never leaves a half result,
and an existing result is never replaced.
"""

import hashlib
import json
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

# a benchmark id becomes a folder name, so it may not contain a slash or start a path
_BENCHMARK_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


class ResultExists(Exception):
    """A result with this name is already on disk, it is never overwritten."""


class _Strict(BaseModel):
    # extra fields are refused, ints are not accepted as floats, nan and infinity are refused
    # because json cannot hold them
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class Environment(_Strict):
    # where the numbers were produced, quality metrics do not depend on the machine
    kind: Literal["local", "ci", "aws"]


class DatasetProvenance(_Strict):
    corpus: Literal["esci"]
    hf_repo: str
    hf_revision: str
    dataset_revision: int
    manifest_sha256: str
    counts: dict[str, int]
    # the manifest hash that loaded.json recorded, None when the index was not loaded from this
    index_loaded_from: str | None
    model_version: str


class RunResult(_Strict):
    benchmark_id: str
    date_utc: str
    command: str
    environment: Environment
    dataset: DatasetProvenance
    config: dict[str, str | int | float | bool | None]
    metrics: dict[str, float]
    # file name inside raw/ -> sha256 of that file
    raw_files: dict[str, str]
    notes: str


def write_result(
    results_root: Path,
    benchmark_id: str,
    *,
    command: str,
    environment_kind: Literal["local", "ci", "aws"],
    dataset_directory: Path,
    model_version: str,
    config: Mapping[str, str | int | float | bool | None],
    metrics: Mapping[str, float],
    clock: Callable[[], datetime],
    raw_files: Sequence[Path] = (),
    notes: str = "",
) -> Path:
    """Write the folder of one run and return its path."""
    if not _BENCHMARK_ID.fullmatch(benchmark_id):
        raise ValueError(f"benchmark id {benchmark_id!r} is not a plain name")
    raw_names = [source.name for source in raw_files]
    if len(set(raw_names)) != len(raw_names):
        raise ValueError("two raw files have the same name")

    manifest_bytes = (dataset_directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    # a marker from another dataset must not vouch for this one
    index_loaded_from: str | None = None
    marker_path = dataset_directory / "loaded.json"
    if marker_path.exists():
        marker_hash = json.loads(marker_path.read_text(encoding="utf-8")).get("manifest_sha256")
        if marker_hash == manifest_sha256:
            index_loaded_from = marker_hash

    now = clock().astimezone(UTC)
    run_name = f"{now:%Y%m%d-%H%M}"
    final_directory = results_root / benchmark_id / run_name
    if final_directory.exists():
        raise ResultExists(str(final_directory))
    # nothing is visible under the final name until everything is written
    staging = final_directory.with_name(run_name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "raw").mkdir(parents=True)
    try:
        raw_checksums: dict[str, str] = {}
        for source in raw_files:
            target = staging / "raw" / source.name
            shutil.copyfile(source, target)
            with target.open("rb") as handle:
                raw_checksums[source.name] = hashlib.file_digest(handle, "sha256").hexdigest()
        result = RunResult(
            benchmark_id=benchmark_id,
            date_utc=now.isoformat(),
            command=command,
            environment=Environment(kind=environment_kind),
            dataset=DatasetProvenance(
                corpus="esci",
                hf_repo=manifest["dataset"]["repo"],
                hf_revision=manifest["dataset"]["revision"],
                dataset_revision=manifest["dataset_revision"],
                manifest_sha256=manifest_sha256,
                counts=manifest["counts"],
                index_loaded_from=index_loaded_from,
                model_version=model_version,
            ),
            config=dict(config),
            metrics=dict(metrics),
            raw_files=raw_checksums,
            notes=notes,
        )
        (staging / "result.json").write_text(
            result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(final_directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final_directory
