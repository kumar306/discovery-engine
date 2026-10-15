"""Kill the worker at every point of every flow, bring it back, and check it finishes the job.

For each scenario the same event is first applied cleanly (the reference), then once more for every
crash point: before each call the pipeline makes, inside the staging, inside the publish. After
each crash the same message is delivered again. The recovered state must be indistinguishable from
the clean one, and what the crash left behind must never have shown readers a half done document.

The crash points are not written down here. They are discovered from the clean run, so a new call
in the pipeline adds crash points to this matrix on its own.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from libs.schemas import DocEvent
from services.embed.pipeline import KafkaEventResult, Pipeline
from tests.support.crash import (
    CallTrace,
    PartialPublishIndex,
    PartialStageIndex,
    SimulatedCrash,
)
from tests.support.hash_embedder import HashEmbedder

from .conftest import BASE, TEST_DIM, Rig, words
from .observe import Observed, observe

pytestmark = pytest.mark.crash

SKIPPED = KafkaEventResult.KAFKA_EVENT_SKIPPED
APPLIED = KafkaEventResult.KAFKA_EVENT_APPLIED


@dataclass(frozen=True)
class Step:
    kind: str  # "upsert" or "delete"
    fields: dict[str, Any] = field(default_factory=dict)


# what happened before the event under test, and the event under test, for each situation
SCENARIOS: dict[str, tuple[list[Step], Step]] = {
    "first_insert": ([], Step("upsert", BASE)),
    "update_some_hits": (
        [Step("upsert", BASE)],
        Step("upsert", {**BASE, "description": words(30, "x")}),
    ),
    "update_all_hits": ([Step("upsert", BASE)], Step("upsert", BASE)),
    "update_removes_a_chunk": (
        [Step("upsert", BASE)],
        Step("upsert", {"title": BASE["title"], "bullets": BASE["bullets"]}),
    ),
    "update_moves_a_chunk": (
        [Step("upsert", BASE)],
        Step("upsert", {**BASE, "bullets": [*BASE["bullets"], words(20, "c")]}),
    ),
    "delete": ([Step("upsert", BASE)], Step("delete")),
    "resurrect": ([Step("upsert", BASE), Step("delete")], Step("upsert", BASE)),
}


@dataclass(frozen=True)
class CrashPlan:
    kind: str  # before_call, stage_points or publish_operations
    at: int

    @property
    def label(self) -> str:
        return f"{self.kind.replace('_', '-')}-{self.at}"


@dataclass
class Run:
    before: Observed
    trace: CallTrace
    crashed: bool
    mid: Observed
    replay_embeds: int
    replay_result: KafkaEventResult
    final: Observed
    again_result: KafkaEventResult
    after_again: Observed


def pipeline_for(
    rig: Rig, trace: CallTrace, plan: CrashPlan | None
) -> tuple[Pipeline, HashEmbedder]:
    embedder = HashEmbedder(dense_dim=TEST_DIM, max_tokens=64)
    index = rig.qdrant_index
    if plan is not None and plan.kind == "stage_points":
        index = PartialStageIndex(rig.qdrant_index, plan.at)
    if plan is not None and plan.kind == "publish_operations":
        index = PartialPublishIndex(rig.qdrant_index, plan.at)
    pipeline = Pipeline(
        trace.wrap(rig.pg_chunk_repo),
        trace.wrap(index),
        trace.wrap(rig.api.s3_store),
        trace.wrap(embedder),
        rig.embed_settings,
    )
    return pipeline, embedder


async def send(rig: Rig, doc_id: str, step: Step, version: int) -> DocEvent:
    if step.kind == "delete":
        return await rig.remove(doc_id, version)
    return await rig.ingest(doc_id, version, **step.fields)


async def run_scenario(rig: Rig, scenario: str, plan: CrashPlan | None, doc_id: str) -> Run:
    prior, target = SCENARIOS[scenario]
    version = 0
    for step in prior:  # the history is applied cleanly, nothing crashes here
        version += 1
        await rig.pipeline.handle(await send(rig, doc_id, step, version))
    before = await observe(rig, doc_id)
    event = await send(rig, doc_id, target, version + 1)

    trace = CallTrace(crash_before=plan.at if plan and plan.kind == "before_call" else None)
    pipeline, _ = pipeline_for(rig, trace, plan)
    crashed = False
    try:
        await pipeline.handle(event)
    except SimulatedCrash:
        crashed = True
    mid = await observe(rig, doc_id)

    # the same message comes back and the worker, started again, handles it
    replay, embedder = pipeline_for(rig, CallTrace(), None)
    replay_result = await replay.handle(event)
    final = await observe(rig, doc_id)

    # and once more, as Kafka may deliver it any number of times
    again, _ = pipeline_for(rig, CallTrace(), None)
    again_result = await again.handle(event)
    return Run(
        before,
        trace,
        crashed,
        mid,
        len(embedder.calls),
        replay_result,
        final,
        again_result,
        await observe(rig, doc_id),
    )


def crash_plans(clean: CallTrace) -> list[CrashPlan]:
    plans = [CrashPlan("before_call", n) for n in range(1, len(clean.calls) + 1)]
    for name, args in clean.calls:
        if name == "stage":
            plans += [CrashPlan("stage_points", k) for k in range(1, len(args[0]))]
        if name == "publish":
            total = len(args[1]) + (1 if args[2] else 0)
            plans += [CrashPlan("publish_operations", k) for k in range(1, total)]
    return plans


def check_the_clean_run(run: Run, scenario: str, version: int) -> None:
    """The reference itself must be right, or equal to it would prove nothing."""
    final = run.final
    assert not run.crashed
    assert (final.current, final.indexed) == (version, version), scenario
    if scenario == "delete":
        assert final.deleted, scenario
        assert not final.chunks, scenario
        assert not final.live, scenario
        assert {p["state"] for p in final.points.values()} == {"deleted"}, scenario
        return
    assert not final.deleted, scenario
    assert final.live == set(final.chunks), scenario  # exactly the committed chunks are searchable
    assert all(final.points[cid]["version"] == version for cid in final.live), scenario
    positions = {cid: pos[0] for cid, (_, pos) in final.chunks.items()}
    assert {cid: final.points[cid]["chunk_idx"] for cid in final.live} == positions, scenario
    assert not set(final.chunks) & final.stale, scenario


def check_the_crash(
    run: Run, clean: Run, clean_names: list[str], scenario: str, plan: CrashPlan
) -> None:
    where = f"{scenario}, crash {plan.label}"
    assert run.crashed, f"{where}: the crash point was never reached"

    # 1. recovery is indistinguishable from a worker that never crashed
    assert run.replay_result is APPLIED, where
    assert run.final.shape == clean.final.shape, where
    # 2. delivered again, it is a normal skip and changes nothing at all
    assert run.again_result is SKIPPED, where
    assert run.after_again.shape == run.final.shape, where

    # 3. what the crash left behind never showed readers a half done document
    old, new, mid = set(run.before.chunks), set(run.final.chunks), run.mid
    committed = mid.current != run.before.current
    if not committed:
        # nothing visible changes before the commit: every point that existed is untouched,
        # and every point that was added is only staged
        for cid, point in run.before.points.items():
            assert mid.points.get(cid) == point, f"{where}: a point changed before the commit"
        for cid in set(mid.points) - set(run.before.points):
            assert mid.points[cid]["state"] == "staging", f"{where}: a new point was visible"
    if not committed and scenario != "delete":
        # staged points are invisible, so nothing new is live before the commit
        assert not mid.live & (new - old), f"{where}: a point of the new version was live"
    if scenario == "delete":
        assert old <= mid.live or not mid.live, f"{where}: a deleted product was half visible"
    elif old and new:
        assert old <= mid.live or new <= mid.live, f"{where}: the document was half visible"
    if mid.indexed is not None and mid.indexed == run.final.indexed:
        # indexed is what readers trust, so by then the points must really be in that state
        if scenario == "delete":
            assert not mid.live, f"{where}: marked indexed while points were still visible"
        else:
            assert new <= mid.live, f"{where}: marked indexed while chunks were not live"
    assert not set(mid.chunks) & mid.stale, f"{where}: a chunk was current and queued"

    # 4. the version numbers only move forward
    assert (mid.indexed if mid.indexed is not None else -1) >= (
        run.before.indexed if run.before.indexed is not None else -1
    ), where
    assert (run.final.indexed or 0) >= (mid.indexed or 0), where

    # 5. once the commit happened, the replay has nothing left to embed
    commit_call = 1 + clean_names.index(
        "commit_delete" if scenario == "delete" else "commit_version"
    )
    after_commit = plan.kind == "publish_operations" or (
        plan.kind == "before_call" and plan.at > commit_call
    )
    if after_commit:
        assert run.replay_embeds == 0, f"{where}: the replay embedded again after the commit"
        assert committed, f"{where}: expected the commit to have happened"
    else:
        assert not committed, f"{where}: committed too early"


@pytest.mark.parametrize("scenario", list(SCENARIOS))
async def test_a_crash_at_any_point_is_recovered_by_delivering_the_message_again(
    rig: Rig, scenario: str
) -> None:

    clean = await run_scenario(rig, scenario, None, f"{scenario}-clean")
    prior, _ = SCENARIOS[scenario]
    check_the_clean_run(clean, scenario, len(prior) + 1)
    assert clean.replay_result is SKIPPED  # nothing crashed, so the replay finds it all done

    plans = crash_plans(clean.trace)
    assert len(plans) >= 4, scenario  # a flow with almost no crash points would be suspicious
    for plan in plans:
        run = await run_scenario(rig, scenario, plan, f"{scenario}-{plan.label}")
        check_the_crash(run, clean, clean.trace.names, scenario, plan)
