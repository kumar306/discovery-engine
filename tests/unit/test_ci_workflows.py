"""The workflow files: the guards that nobody can break by accident, since only GitHub runs them."""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def load(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    return loaded


def triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    # yaml 1.1 reads the key "on" as the boolean True
    found: dict[str, Any] = workflow.get("on") or workflow[True]
    return found


def steps_of(workflow: dict[str, Any], job: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = workflow["jobs"][job]["steps"]
    return found


# the files whose change can move the numbers of the ladder
QUALITY_PATHS = [
    "libs/embedding/**",
    "libs/schemas/**",
    "services/embed/compose.py",
    "services/embed/qdrant.py",
    "services/query/**",
    "eval/**",
    "scripts/download_data.py",
    "benchmarks/baseline/**",
    "pyproject.toml",
    "uv.lock",
    ".github/workflows/quality.yml",
]


@pytest.mark.parametrize("name", ["ci.yml", "quality.yml"])
def test_every_action_is_pinned_to_a_commit_not_to_a_tag(name: str) -> None:
    text = (WORKFLOWS / name).read_text(encoding="utf-8")
    uses = re.findall(r"uses:\s*(\S+)", text)
    assert uses
    for reference in uses:
        assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", reference), reference


def test_the_quality_workflow_runs_when_index_defining_code_changes() -> None:
    on = triggers(load("quality.yml"))
    assert on["push"]["branches"] == ["main"]
    for event in ("push", "pull_request"):
        for path in QUALITY_PATHS:
            assert path in on[event]["paths"], (event, path)
    assert "workflow_dispatch" in on


def test_the_quality_workflow_can_only_read_the_repository_and_cannot_run_for_ever() -> None:
    workflow = load("quality.yml")
    assert workflow["permissions"] == {"contents": "read"}
    assert 30 <= workflow["jobs"]["quality"]["timeout-minutes"] <= 90


def test_qdrant_in_the_quality_job_is_the_same_image_as_in_the_compose_stack() -> None:
    compose = yaml.safe_load(
        (ROOT / "deploy" / "compose" / "docker-compose.yml").read_text(encoding="utf-8")
    )
    image = compose["services"]["qdrant"]["image"]
    commands = " ".join(
        str(step.get("run", "")) for step in steps_of(load("quality.yml"), "quality")
    )
    assert image in commands


def test_the_stored_index_is_keyed_by_everything_that_defines_it() -> None:
    steps = steps_of(load("quality.yml"), "quality")
    keys = [
        step["with"]["key"]
        for step in steps
        if "qdrant" in str(step.get("with", {}).get("path", ""))
    ]
    assert len(keys) == 1
    for defining in (
        "libs/embedding/**",
        "services/embed/compose.py",
        "libs/schemas/**",
        "eval/index_builder.py",
        "data/processed/mini/manifest.json",
    ):
        assert f"'{defining}'" in keys[0], defining
    # the versions of the libraries that make the vectors, not the whole lock file: any bump of any
    # dependency would rebuild the index
    assert "steps.index_libs.outputs.versions" in keys[0]
    assert "uv.lock" not in keys[0]
    # code that only reads the index must not be part of the key, or every change rebuilds it
    assert "services/query" not in keys[0]
    assert "eval/ladder.py" not in keys[0]


def test_the_library_versions_of_the_key_come_from_the_module_that_names_the_libraries() -> None:
    steps = steps_of(load("quality.yml"), "quality")
    versions = next(step for step in steps if step.get("id") == "index_libs")
    assert "python -m eval.index_libs" in versions["run"]
    # the step output the key reads
    assert "GITHUB_OUTPUT" in versions["run"]
    # and it runs before the step that restores the index
    names = [step.get("id") or step.get("name") for step in steps]
    restore = next(
        index
        for index, step in enumerate(steps)
        if "qdrant" in str(step.get("with", {}).get("path", ""))
    )
    assert names.index("index_libs") < restore


def test_a_dependency_bump_still_runs_the_job_but_does_not_by_itself_rebuild_the_index() -> None:
    on = triggers(load("quality.yml"))
    assert "uv.lock" in on["push"]["paths"]
    assert "uv.lock" in on["pull_request"]["paths"]


def test_the_stored_index_is_keyed_by_the_qdrant_version_because_its_files_depend_on_it() -> None:
    compose = yaml.safe_load(
        (ROOT / "deploy" / "compose" / "docker-compose.yml").read_text(encoding="utf-8")
    )
    version = compose["services"]["qdrant"]["image"].split(":")[1]
    steps = steps_of(load("quality.yml"), "quality")
    key = next(
        step["with"]["key"]
        for step in steps
        if "qdrant" in str(step.get("with", {}).get("path", ""))
    )
    assert key.startswith(f"qdrant-{version}-")


def test_the_stored_dataset_is_keyed_by_the_code_that_builds_it() -> None:
    steps = steps_of(load("quality.yml"), "quality")
    keys = [
        step["with"]["key"]
        for step in steps
        if "processed" in str(step.get("with", {}).get("path", ""))
    ]
    assert len(keys) == 1
    for builder in (
        "eval/build.py",
        "eval/dataset.py",
        "eval/source.py",
        "scripts/download_data.py",
    ):
        assert f"'{builder}'" in keys[0], builder


def test_qdrant_is_stopped_and_the_result_is_kept_even_when_the_gate_fails() -> None:
    steps = steps_of(load("quality.yml"), "quality")
    stop = next(step for step in steps if "docker stop" in str(step.get("run", "")))
    keep = next(step for step in steps if "upload-artifact" in str(step.get("uses", "")))
    assert stop["if"] == "always()"
    # the container wrote the storage as root, the cache step runs as the runner user
    assert "chown" in stop["run"]
    assert keep["if"] == "always()"
    assert "benchmarks/results" in keep["with"]["path"]


def test_the_quality_command_is_the_one_in_the_makefile() -> None:
    steps = steps_of(load("quality.yml"), "quality")
    assert any(step.get("run") == "make quality" for step in steps)
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert re.search(r"^quality:.*\n\t\$\(UV\) python -m eval\.quality_job", makefile, re.MULTILINE)


def test_qdrant_that_never_becomes_ready_fails_the_step_and_shows_its_log() -> None:
    steps = steps_of(load("quality.yml"), "quality")
    start = next(step for step in steps if "docker run" in str(step.get("run", "")))
    # a loop that just ends when Qdrant is not ready hides the cause behind a later connection error
    assert "exit 1" in start["run"]
    assert "docker logs qdrant" in start["run"]


def test_qdrant_gets_time_to_flush_before_the_storage_is_saved_and_its_log_is_kept_on_failure() -> (
    None
):
    steps = steps_of(load("quality.yml"), "quality")
    stop = next(step for step in steps if "docker stop" in str(step.get("run", "")))
    # the default grace of 10 seconds may kill it in the middle of writing the files the cache saves
    assert re.search(r"docker stop -t [3-9]\d", stop["run"])
    logs = [step for step in steps if "docker logs qdrant" in str(step.get("run", ""))]
    assert any(step.get("if") == "failure()" for step in logs)


def test_the_qdrant_image_is_not_the_release_that_cannot_restart_after_a_filtered_write() -> None:
    # Qdrant v1.17.0 panics at startup when its write-ahead log holds a set_payload or delete with a
    # filter on point ids ("Can't deserialize entry"), fixed in v1.17.1; the worker writes such
    # filters, see tests/integration/test_qdrant_restart.py for the real proof
    compose = yaml.safe_load(
        (ROOT / "deploy" / "compose" / "docker-compose.yml").read_text(encoding="utf-8")
    )
    tag = compose["services"]["qdrant"]["image"].split(":")[1]
    assert tag != "v1.17.0"
    assert "qdrant/qdrant:v1.17.0" not in (WORKFLOWS / "quality.yml").read_text(encoding="utf-8")
