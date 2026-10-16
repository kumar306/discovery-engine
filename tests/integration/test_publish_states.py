"""Stop a publish after every payload update and check what a reader would keep.

The publish flips a product from version N to N+1 one chunk at a time, and Qdrant cannot make
that atomic for readers, so a search can meet a product with some chunks at N and some at N+1.
Nothing in Qdrant picks between them: it returns every live chunk that matches. Our query
service does, by keeping only the chunks with the highest version of the product (see
docs/learning/reader_consistency.md). This test is the worker's half of that deal: at every half
done state the payloads must be such that the rule shows one version, never a changed chunk next
to its replacement, and the whole new version once the publish is finished. The reader's half,
through the real query code, is tested in S6.
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import pytest

from tests.support.crash import CallTrace, SimulatedCrash

from .conftest import Rig, words
from .observe import Observed, observe
from .test_crash_matrix import CrashPlan, Step, crash_plans, pipeline_for, send

pytestmark = pytest.mark.integration

# every bullet is longer than half the token limit of the test embedder, so each is its own chunk
FIVE: dict[str, Any] = {"title": "Acme mouse", "bullets": [words(40, t) for t in "abcde"]}


def five_with(**changed: str) -> dict[str, Any]:
    # replace some bullets by number, for example {"1": "new text"} changes the second bullet
    bullets = [changed.get(str(i), text) for i, text in enumerate(FIVE["bullets"])]
    return {**FIVE, "bullets": bullets}


# what was applied before, and the update whose publish is stopped half way
SCENARIOS: dict[str, tuple[list[Step], Step]] = {
    "five_chunks_two_change": (
        [Step("upsert", FIVE)],
        Step("upsert", five_with(**{"1": words(40, "x"), "3": words(40, "y")})),
    ),
    "five_chunks_one_removed": (
        [Step("upsert", FIVE)],
        Step("upsert", {**FIVE, "bullets": FIVE["bullets"][:4]}),
    ),
    "five_chunks_one_inserted_in_front": (
        [Step("upsert", FIVE)],
        Step("upsert", {**FIVE, "bullets": [words(40, "z"), *FIVE["bullets"]]}),
    ),
    "five_chunks_all_change": (
        [Step("upsert", FIVE)],
        Step("upsert", {**FIVE, "bullets": [words(40, t) for t in "vwxyz"]}),
    ),
    "five_chunks_nothing_changes": ([Step("upsert", FIVE)], Step("upsert", FIVE)),
    "first_insert": ([], Step("upsert", FIVE)),
}


def keep_highest_version(points: dict[UUID, dict[str, Any]]) -> set[UUID]:
    """The reader rule: of the live chunks of one product, keep those with the highest version."""
    live = {chunk_id: point for chunk_id, point in points.items() if point["state"] == "live"}
    if not live:
        return set()
    highest = max(point["version"] for point in live.values())
    return {chunk_id for chunk_id, point in live.items() if point["version"] == highest}


@dataclass
class Look:
    """What a reader would keep from one half done state."""

    version: int | None
    kept: set[UUID]


def what_a_reader_keeps(
    state: Observed, old: set[UUID], new: set[UUID], old_version: int, new_version: int, where: str
) -> Look:
    kept = keep_highest_version(state.points)

    # the version written on a live point must be true: a point of the new version belongs to the
    # new chunk set, a point of the old version to the old one, or the rule keeps the wrong chunks
    for chunk_id in state.live:
        version = state.points[chunk_id]["version"]
        if version == new_version:
            assert chunk_id in new, f"{where}: a chunk says the new version but is not in it"
        else:
            assert version == old_version, f"{where}: a chunk says version {version}"
            assert chunk_id in old, f"{where}: a chunk says the old version but is not in it"

    # a changed chunk and its replacement are never both kept
    assert not (kept & (old - new) and kept & (new - old)), f"{where}: two versions at once"
    # a product that was searchable never has nothing to show
    if old:
        assert kept, f"{where}: a searchable product showed nothing"
    versions = {state.points[chunk_id]["version"] for chunk_id in kept}
    assert len(versions) <= 1, where
    # while the old version is what is shown it is shown whole: no old chunk goes before a new
    # one is live
    if versions == {old_version}:
        assert kept == old, f"{where}: the old version was shown in part"
    return Look(versions.pop() if versions else None, kept)


async def apply_and_stop(
    rig: Rig, prior: list[Step], target: Step, doc_id: str, stop_after: int | None
) -> tuple[Observed, Observed, CallTrace]:
    version = 0
    for step in prior:  # the history is applied cleanly
        version += 1
        await rig.pipeline.handle(await send(rig, doc_id, step, version))
    before = await observe(rig, doc_id)
    event = await send(rig, doc_id, target, version + 1)
    trace = CallTrace()
    plan = None if stop_after is None else CrashPlan("publish_operations", stop_after)
    pipeline, _ = pipeline_for(rig, trace, plan)
    try:
        await pipeline.handle(event)
    except SimulatedCrash:
        assert stop_after is not None, "the publish stopped when nothing should stop it"
    return before, await observe(rig, doc_id), trace


@pytest.mark.parametrize("scenario", list(SCENARIOS))
async def test_a_reader_sees_one_version_at_every_half_done_state_of_a_publish(
    rig: Rig, scenario: str
) -> None:
    prior, target = SCENARIOS[scenario]
    old_version = len(prior)
    new_version = old_version + 1

    # the clean run is the reference and says how many payload updates the publish makes
    before, finished, trace = await apply_and_stop(rig, prior, target, f"{scenario}-clean", None)
    stops = [plan.at for plan in crash_plans(trace) if plan.kind == "publish_operations"]
    assert len(stops) >= 2, scenario  # one update leaves no half done state to look at

    # chunk ids depend on the document, so every run is judged against its own old and new sets
    old = set(before.chunks)
    looks = [
        what_a_reader_keeps(before, old, set(finished.chunks), old_version, new_version, "before")
    ]
    assert looks[0].kept == old
    for k in stops:
        start, half, _ = await apply_and_stop(rig, prior, target, f"{scenario}-after-{k}", k)
        # the commit happened and the version is not yet marked indexed
        assert (half.current, half.indexed) == (new_version, old_version or None), scenario
        where = f"{scenario}, after {k}"
        looks.append(
            what_a_reader_keeps(
                half, set(start.chunks), set(half.chunks), old_version, new_version, where
            )
        )
    last = what_a_reader_keeps(
        finished, old, set(finished.chunks), old_version, new_version, "done"
    )
    looks.append(last)

    # the finished publish shows the whole new version, not a part of it
    assert last.kept == set(finished.chunks), scenario
    assert last.version == new_version, scenario
    # a reader moves from the old version to the new one once, and never back
    seen = [look.version for look in looks if look.version is not None]
    assert seen == sorted(seen), f"{scenario}: the reader went back to an older version"
    # once the new version is what is shown, it only grows until it is whole
    grown = [len(look.kept) for look in looks if look.version == new_version]
    assert grown == sorted(grown), f"{scenario}: chunks of the new version disappeared again"
