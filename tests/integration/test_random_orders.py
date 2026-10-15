"""Random event orders, duplicates and crashes: the visible result is always the newest write.

The crash matrix kills one event at every point. This file mixes everything: several writes to one
document, delivered in any order, some more than once, some killed part way, and in the end every
message is delivered again. Whatever the history, a searcher must see exactly what applying only
the newest write on its own would show, and Postgres must agree.
"""

import asyncio
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from libs.schemas import DocEvent
from services.embed.pipeline import Pipeline
from tests.support.crash import CallTrace, SimulatedCrash
from tests.support.hash_embedder import HashEmbedder

from .conftest import BASE, TEST_DIM, Rig, words
from .observe import Observed, observe

pytestmark = pytest.mark.crash

VARIANTS: dict[str, dict[str, Any]] = {
    "A": BASE,
    "B": {**BASE, "description": words(30, "x")},
    "C": {**BASE, "bullets": [*BASE["bullets"], words(20, "c")]},
    "D": {"title": BASE["title"], "bullets": BASE["bullets"]},
}

# the pipeline makes at most nine calls for one event, a crash can land before any of them
CALLS = 9


# a plan is plain data that hypothesis draws before the test runs, and run_plan then carries it out
#   writes: the changes made to one document, in version order, "A" to "D" are uploads of that
#     content and "-" is a delete, so ("A", "B") means version 1 uploads A and version 2 uploads B
#   deliveries: the worker handling messages in the order they happen, each one is (which write,
#     crash), which write is the position in the list of events, 0 is the event of version 1, and
#     crash is the call number the process dies before, or None for a delivery that goes through
#     the call number counts the calls of that one delivery, so a delivery that makes fewer calls
#     (a stale event stops after the first one) never reaches it and is just a normal delivery
#   final_order: after the random part every message is delivered once more with nothing going
#     wrong, in this order, which models kafka redelivering whatever was not finished
# the three are drawn on their own and need no syncing, the check at the end does not depend on them
#
# why the end state is the same in any order
# the version gate and the commit under the row lock make the newest version always win
# an older message that arrives after a newer one is dropped, an older one that arrives first is
# applied and then replaced by the newer one, a delete hides every point and a later upload brings
# the product back, so for ("A", "-") the delete ends the document whether the upload lands before
# it (and is hidden) or after it (and is dropped as stale)
# only what a searcher and postgres can see is compared, the leftovers nobody can see (superseded
# points, the stale queue) differ with the history and that is fine, the reference is the newest
# write applied alone on a fresh document
@dataclass(frozen=True)
class Plan:
    writes: tuple[str, ...]  # "A".."D" is an upload of that content, "-" is a delete
    deliveries: tuple[tuple[int, int | None], ...]  # (which write, crash before call n or None)
    final_order: tuple[int, ...]  # the order the leftovers are finally delivered in


@st.composite
def plans(draw: st.DrawFn) -> Plan:
    writes = draw(st.lists(st.sampled_from(["A", "B", "C", "D", "-"]), min_size=1, max_size=4))
    index = st.integers(min_value=0, max_value=len(writes) - 1)
    crash = st.one_of(st.none(), st.integers(min_value=1, max_value=CALLS))
    deliveries = draw(st.lists(st.tuples(index, crash), max_size=6))
    final_order = draw(st.permutations(range(len(writes))))
    return Plan(tuple(writes), tuple(deliveries), tuple(final_order))


async def write(rig: Rig, doc_id: str, version: int, kind: str) -> DocEvent:
    if kind == "-":
        return await rig.remove(doc_id, version)
    return await rig.ingest(doc_id, version, **VARIANTS[kind])


def visible(seen: Observed) -> tuple[Any, ...]:
    """What a searcher and Postgres would show, leaving out points nobody can see."""
    return (
        seen.current,
        seen.indexed,
        seen.deleted,
        tuple(sorted(seen.chunks.values())),
        tuple(
            sorted(
                (p["content_hash"], p["version"], p["chunk_idx"])
                for p in seen.points.values()
                if p["state"] == "live"
            )
        ),
    )


async def run_plan(rig: Rig, plan: Plan, doc_id: str) -> None:
    events = [await write(rig, doc_id, v, kind) for v, kind in enumerate(plan.writes, start=1)]
    for which, crash_at in plan.deliveries:
        trace = CallTrace(crash_before=crash_at)
        pipeline = Pipeline(
            trace.wrap(rig.pg_chunk_repo),
            trace.wrap(rig.qdrant_index),
            trace.wrap(rig.api.s3_store),
            trace.wrap(HashEmbedder(dense_dim=TEST_DIM, max_tokens=64)),
            rig.embed_settings,
        )
        with suppress(SimulatedCrash):
            await pipeline.handle(events[which])
    # whatever happened, every message is delivered again, so each one is finished or dropped
    for which in plan.final_order:
        await rig.pipeline.handle(events[which])
    after = await observe(rig, doc_id)

    # the reference: only the newest write, applied to a fresh document, with nothing going wrong
    newest = len(plan.writes)
    reference_id = f"{doc_id}-ref"
    await rig.pipeline.handle(await write(rig, reference_id, newest, plan.writes[-1]))
    assert visible(after) == visible(await observe(rig, reference_id)), plan

    # and one more delivery of everything changes nothing
    for event in events:
        await rig.pipeline.handle(event)
    assert visible(await observe(rig, doc_id)) == visible(after), plan


@settings(
    max_examples=25,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
# plans that always run, the tricky histories found while building the matrix
@example(plan=Plan(("A", "-"), (), (0, 1)))  # upload, then delete
@example(plan=Plan(("-", "A"), (), (1, 0)))  # delete, then upload: the product comes back
@example(plan=Plan(("A", "B", "-"), ((2, 5), (0, 8), (1, 3)), (2, 0, 1)))  # newest first
@example(plan=Plan(("A", "B"), ((0, 7), (1, 6), (0, 1)), (0, 1)))  # old one after a crash
@given(plan=plans())
def test_any_order_of_writes_deliveries_and_crashes_ends_at_the_newest_write(
    rig: Rig, plan: Plan
) -> None:

    # hypothesis calls a plain function, the stores belong to the event loop of the fixtures
    loop = asyncio.get_event_loop_policy().get_event_loop()
    doc_id = f"random-{uuid.uuid4().hex[:12]}"
    loop.run_until_complete(run_plan(rig, plan, doc_id))


@pytest.mark.parametrize("plan", [Plan(("A", "B"), ((1, 5), (0, 7)), (1, 0))])
async def test_a_fixed_plan_runs_as_a_plain_async_test(rig: Rig, plan: Plan) -> None:
    await run_plan(rig, plan, "random-fixed-plan")
