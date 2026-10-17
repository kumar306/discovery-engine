"""The compactor and the ingester sharing the stale queue, against real Postgres and Qdrant.

A stale chunk is a point no version needs any more. The compactor claims its queue row, deletes
the point, then deletes the row, without holding a Postgres lock while it talks to Qdrant. The
ingester can bring a stale chunk back at any moment, so every interleaving of the two is tested.
"""

import asyncio
from collections.abc import Callable, Sequence
from typing import Any
from uuid import UUID, uuid4

import psycopg
import pytest
from prometheus_client import CollectorRegistry

from libs.observability.metrics import EmbedMetrics, MaintenanceMetrics
from services.embed.maintenance import Compactor
from services.embed.pipeline import ChunkBeingDeleted
from services.embed.repository import ClaimedChunk
from tests.support.crash import SimulatedCrash

from .conftest import Rig, words

# every bullet is longer than half the token limit of the test embedder, so each is its own chunk
FIVE: dict[str, Any] = {"title": "Acme mouse", "bullets": [words(40, t) for t in "abcde"]}
FOUR = {**FIVE, "bullets": FIVE["bullets"][:4]}  # the same product without its last bullet


def compactor_for(
    rig: Rig,
    grace_s: float = 0,
    claim_timeout_s: float = 600,
    index: Any = None,
    repo: Any = None,
    registry: CollectorRegistry | None = None,
) -> Compactor:
    return Compactor(
        repo or rig.pg_chunk_repo,
        index or rig.qdrant_index,
        grace_s=grace_s,
        claim_timeout_s=claim_timeout_s,
        batch_size=100,
        metrics=MaintenanceMetrics(registry or CollectorRegistry()),
    )


def run_sql(rig: Rig, statement: str, *params: Any) -> list[tuple[Any, ...]]:
    with psycopg.connect(rig.api.database.direct_dsn, autocommit=True) as conn:
        cursor = conn.execute(statement, params)
        return cursor.fetchall() if cursor.description else []


def stale_rows(rig: Rig, doc_id: str) -> dict[UUID, bool]:
    """chunk id -> is it claimed by a compactor"""
    rows = run_sql(
        rig,
        "select chunk_id, deletion_started_at is not null from stale_chunks where doc_id = %s",
        doc_id,
    )
    return {chunk_id: claimed for chunk_id, claimed in rows}


async def point_ids(rig: Rig, doc_id: str) -> set[UUID]:
    return set(await rig.points(doc_id))


async def one_chunk_goes_stale(rig: Rig, doc_id: str) -> UUID:
    """Version 1 has five chunks, version 2 drops one: its point is superseded and queued."""
    await rig.pipeline.handle(await rig.ingest(doc_id, 1, **FIVE))
    await rig.pipeline.handle(await rig.ingest(doc_id, 2, **FOUR))
    (stale,) = stale_rows(rig, doc_id)
    assert stale in await point_ids(rig, doc_id)
    return stale


def claimed_rows_of(chunks: Sequence[ClaimedChunk]) -> set[UUID]:
    return {chunk.chunk_id for chunk in chunks}


# the grace period and the plain job


async def test_nothing_is_claimed_before_the_grace_period_is_over(rig: Rig) -> None:
    stale = await one_chunk_goes_stale(rig, "G1")
    assert await compactor_for(rig, grace_s=3600).compact_once() == 0
    assert stale_rows(rig, "G1") == {stale: False}
    assert stale in await point_ids(rig, "G1")


async def test_after_the_grace_period_the_point_and_the_queue_row_are_both_gone(
    rig: Rig,
) -> None:
    stale = await one_chunk_goes_stale(rig, "G2")
    live_before = {c for c, p in (await rig.points("G2")).items() if p["state"] == "live"}
    run_sql(rig, "update stale_chunks set stale_at = now() - interval '2 hours'")
    registry = CollectorRegistry()
    assert await compactor_for(rig, grace_s=3600, registry=registry).compact_once() == 1
    assert stale_rows(rig, "G2") == {}
    assert stale not in await point_ids(rig, "G2")
    live_after = {c for c, p in (await rig.points("G2")).items() if p["state"] == "live"}
    assert live_after == live_before  # the live chunks were not touched
    assert registry.get_sample_value("embed_stale_chunks_deleted_total") == 1


async def test_a_current_chunk_is_never_claimed_even_if_a_queue_row_names_it(rig: Rig) -> None:
    await rig.pipeline.handle(await rig.ingest("C1", 1, **FIVE))
    (current_chunk,) = run_sql(
        rig, "select chunk_id from chunks where doc_id = 'C1' order by positions limit 1"
    )[0][:1]
    run_sql(
        rig,
        "insert into stale_chunks (corpus, doc_id, chunk_id) values ('esci', 'C1', %s)",
        current_chunk,
    )
    assert await compactor_for(rig).compact_once() == 0
    assert current_chunk in await point_ids(rig, "C1")


# the claim


async def test_a_claimed_row_is_not_claimed_again_until_the_claim_timeout_is_over(
    rig: Rig,
) -> None:
    stale = await one_chunk_goes_stale(rig, "T1")
    first = await rig.pg_chunk_repo.claim_stale(grace_s=0, claim_timeout_s=600, limit=10)
    assert claimed_rows_of(first) == {stale}
    assert stale_rows(rig, "T1") == {stale: True}
    assert await rig.pg_chunk_repo.claim_stale(grace_s=0, claim_timeout_s=600, limit=10) == []
    # the compactor that claimed it died long ago
    run_sql(rig, "update stale_chunks set deletion_started_at = now() - interval '1 hour'")
    again = await rig.pg_chunk_repo.claim_stale(grace_s=0, claim_timeout_s=600, limit=10)
    assert claimed_rows_of(again) == {stale}


async def test_a_row_an_ingester_holds_is_skipped_and_taken_on_the_next_pass(rig: Rig) -> None:
    stale = await one_chunk_goes_stale(rig, "L1")
    with psycopg.connect(rig.api.database.direct_dsn) as ingester:
        ingester.execute("select 1 from stale_chunks where doc_id = 'L1' for update")
        # the compactor does not wait for the lock, it leaves the row for the next pass
        # bounded, so a compactor that waited for the lock would fail this test and not hang it
        assert await asyncio.wait_for(compactor_for(rig).compact_once(), timeout=5) == 0
        assert stale in await point_ids(rig, "L1")
    assert await compactor_for(rig).compact_once() == 1
    assert stale not in await point_ids(rig, "L1")


async def test_two_compactors_at_once_never_claim_the_same_row(rig: Rig) -> None:
    await rig.pipeline.handle(await rig.ingest("M1", 1, **FIVE))
    await rig.pipeline.handle(
        await rig.ingest("M1", 2, title="Acme mouse", bullets=[words(5, "z")])
    )
    assert len(stale_rows(rig, "M1")) >= 4
    claims = await asyncio.gather(
        *(rig.pg_chunk_repo.claim_stale(grace_s=0, claim_timeout_s=600, limit=2) for _ in range(3))
    )
    seen = [chunk.chunk_id for claim in claims for chunk in claim]
    assert len(seen) == len(set(seen))  # nobody got a row twice
    assert len(seen) >= 4  # and between them they took the lot


# a compactor that dies or fails at each step of a pass


class Dies:
    """Wraps the index or the repository so that one call dies the way a killed process does."""

    def __init__(self, inner: Any, method: str, error: BaseException) -> None:
        self._inner, self._method, self._error = inner, method, error

    def __getattr__(self, name: str) -> Any:
        if name == self._method:

            async def fails(*args: Any, **kwargs: Any) -> None:
                raise self._error

            return fails
        return getattr(self._inner, name)


async def test_a_compactor_killed_after_the_claim_leaves_a_claim_the_next_pass_finishes(
    rig: Rig,
) -> None:
    stale = await one_chunk_goes_stale(rig, "K1")
    dying = compactor_for(rig, index=Dies(rig.qdrant_index, "delete", SimulatedCrash("killed")))
    with pytest.raises(SimulatedCrash):
        await dying.compact_once()
    assert stale_rows(rig, "K1") == {stale: True}
    assert stale in await point_ids(rig, "K1")  # nothing was deleted yet
    # before the claim timeout the row is left alone, in case that compactor is only slow
    assert await compactor_for(rig, claim_timeout_s=600).compact_once() == 0
    # after it, the next pass takes the row over and finishes
    assert await compactor_for(rig, claim_timeout_s=0).compact_once() == 1
    assert stale_rows(rig, "K1") == {}
    assert stale not in await point_ids(rig, "K1")


async def test_a_compactor_killed_after_the_delete_leaves_a_claim_and_a_repeat_is_harmless(
    rig: Rig,
) -> None:
    stale = await one_chunk_goes_stale(rig, "K2")
    dying = compactor_for(
        rig, repo=Dies(rig.pg_chunk_repo, "forget_claimed", SimulatedCrash("killed"))
    )
    with pytest.raises(SimulatedCrash):
        await dying.compact_once()
    assert stale not in await point_ids(rig, "K2")  # the point is already gone
    assert stale_rows(rig, "K2") == {stale: True}
    # deleting an already missing point succeeds, so the repeat just removes the row
    assert await compactor_for(rig, claim_timeout_s=0).compact_once() == 1
    assert stale_rows(rig, "K2") == {}


async def test_a_failing_qdrant_delete_keeps_the_claim_and_loses_nothing(rig: Rig) -> None:
    stale = await one_chunk_goes_stale(rig, "F1")
    failing = compactor_for(rig, index=Dies(rig.qdrant_index, "delete", ConnectionError("down")))
    with pytest.raises(ConnectionError):
        await failing.compact_once()
    assert stale_rows(rig, "F1") == {stale: True}
    assert stale in await point_ids(rig, "F1")
    assert await compactor_for(rig, claim_timeout_s=0).compact_once() == 1
    assert stale not in await point_ids(rig, "F1")


async def test_forgetting_leaves_a_row_that_was_queued_again_after_the_claim(rig: Rig) -> None:
    """A slow compactor must not remove the row of a chunk that went stale again in the meantime."""
    stale = await one_chunk_goes_stale(rig, "N1")
    (old_claim,) = await rig.pg_chunk_repo.claim_stale(grace_s=0, claim_timeout_s=600, limit=10)
    # the claim was taken over, finished, and the chunk came back and went stale a second time
    run_sql(rig, "delete from stale_chunks where doc_id = 'N1'")
    run_sql(
        rig,
        "insert into stale_chunks (corpus, doc_id, chunk_id) values ('esci', 'N1', %s)",
        stale,
    )
    await rig.pg_chunk_repo.forget_claimed([old_claim])  # the slow one finally finishes
    assert stale_rows(rig, "N1") == {stale: False}  # the new, unclaimed row is still queued


async def test_deleting_a_point_that_does_not_exist_is_fine(rig: Rig) -> None:
    await rig.qdrant_index.delete([uuid4(), uuid4()])


async def test_the_age_of_the_oldest_stale_chunk_is_reported(rig: Rig) -> None:
    assert await rig.pg_chunk_repo.oldest_stale_age_s() == 0  # nothing waiting
    await one_chunk_goes_stale(rig, "A1")
    run_sql(rig, "update stale_chunks set stale_at = now() - interval '2 hours'")
    assert 7100 < await rig.pg_chunk_repo.oldest_stale_age_s() < 7300


# the ingester and the compactor, in every order they can meet


async def test_a_chunk_brought_back_before_the_claim_is_never_claimed(rig: Rig) -> None:
    stale = await one_chunk_goes_stale(rig, "R1")
    await rig.pipeline.handle(await rig.ingest("R1", 3, **FIVE))  # the dropped bullet returns
    assert stale_rows(rig, "R1") == {}
    assert await compactor_for(rig).compact_once() == 0
    assert stale in await point_ids(rig, "R1")
    assert (await rig.points("R1"))[stale]["state"] == "live"


async def test_an_event_that_needs_a_claimed_chunk_waits_without_embedding_anything(
    rig: Rig,
) -> None:
    stale = await one_chunk_goes_stale(rig, "W1")
    await rig.pg_chunk_repo.claim_stale(
        grace_s=0, claim_timeout_s=600, limit=10
    )  # a compactor is mid pass
    event = await rig.ingest("W1", 3, **FIVE)
    embeds_before = len(rig.embedder.calls)
    registry = CollectorRegistry()
    waiting = rig.pipeline_with(metrics=EmbedMetrics(registry))
    with pytest.raises(ChunkBeingDeleted):
        await waiting.handle(event)
    assert len(rig.embedder.calls) == embeds_before  # nothing was embedded for it
    assert rig.row("W1")["current"] == 2  # and nothing was committed
    assert registry.get_sample_value("embed_chunk_being_deleted_total") == 1
    assert stale_rows(rig, "W1") == {stale: True}


async def test_once_the_compactor_has_finished_the_same_event_goes_through_and_the_point_exists(
    rig: Rig,
) -> None:
    stale = await one_chunk_goes_stale(rig, "W2")
    event = await rig.ingest("W2", 3, **FIVE)
    await rig.pg_chunk_repo.claim_stale(grace_s=0, claim_timeout_s=600, limit=10)
    with pytest.raises(ChunkBeingDeleted):
        await rig.pipeline.handle(event)
    await compactor_for(rig, claim_timeout_s=0).compact_once()  # the pass is finished
    assert stale not in await point_ids(rig, "W2")
    await rig.pipeline.handle(event)  # the retry
    assert (rig.row("W2")["current"], rig.row("W2")["indexed"]) == (3, 3)
    assert stale in await point_ids(rig, "W2")  # embedded again and staged, the delete was real
    assert (await rig.points("W2"))[stale]["state"] == "live"
    assert stale_rows(rig, "W2") == {}


class BeforeCommit:
    """Runs something between the diff and the commit, where the compactor can slip in."""

    def __init__(self, inner: Any, action: Callable[[], Any]) -> None:
        self._inner, self._action = inner, action

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def commit_version(self, *args: Any, **kwargs: Any) -> Any:
        await self._action()
        return await self._inner.commit_version(*args, **kwargs)


async def test_a_claim_that_lands_after_the_diff_stops_the_commit(rig: Rig) -> None:
    stale = await one_chunk_goes_stale(rig, "X1")
    event = await rig.ingest("X1", 3, **FIVE)

    async def the_compactor_claims_now() -> None:
        await rig.pg_chunk_repo.claim_stale(grace_s=0, claim_timeout_s=600, limit=10)

    racing = rig.pipeline_with(
        pg_chunk_repo=BeforeCommit(rig.pg_chunk_repo, the_compactor_claims_now)
    )
    with pytest.raises(ChunkBeingDeleted):
        await racing.handle(event)
    assert rig.row("X1")["current"] == 2  # not committed, so no chunk row without a point
    assert stale_rows(rig, "X1") == {stale: True}  # and the claim was not undone
    await compactor_for(rig, claim_timeout_s=0).compact_once()
    await rig.pipeline.handle(event)
    assert (rig.row("X1")["current"], rig.row("X1")["indexed"]) == (3, 3)
    assert stale in await point_ids(rig, "X1")


async def test_a_pass_that_finishes_after_the_diff_stops_the_commit_and_the_retry_embeds_again(
    rig: Rig,
) -> None:
    stale = await one_chunk_goes_stale(rig, "X2")
    event = await rig.ingest("X2", 3, **FIVE)

    async def the_whole_pass_runs_now() -> None:
        await compactor_for(rig).compact_once()

    racing = rig.pipeline_with(
        pg_chunk_repo=BeforeCommit(rig.pg_chunk_repo, the_whole_pass_runs_now)
    )
    with pytest.raises(ChunkBeingDeleted):
        await racing.handle(event)
    assert stale not in await point_ids(rig, "X2")
    assert rig.row("X2")["current"] == 2
    await rig.pipeline.handle(event)  # the diff now sees a plain miss, embeds it and stages it
    assert (rig.row("X2")["current"], rig.row("X2")["indexed"]) == (3, 3)
    assert (await rig.points("X2"))[stale]["state"] == "live"


async def test_a_deleted_product_loses_its_points_after_the_grace_period(rig: Rig) -> None:
    await rig.pipeline.handle(await rig.ingest("D1", 1, **FIVE))
    await rig.pipeline.handle(await rig.remove("D1", 2))
    assert len(stale_rows(rig, "D1")) == 5
    assert await compactor_for(rig, grace_s=3600).compact_once() == 0
    assert await compactor_for(rig).compact_once() == 5
    assert await point_ids(rig, "D1") == set()


async def test_a_product_brought_back_while_its_chunks_are_claimed_waits_and_then_succeeds(
    rig: Rig,
) -> None:
    await rig.pipeline.handle(await rig.ingest("D2", 1, **FIVE))
    await rig.pipeline.handle(await rig.remove("D2", 2))
    event = await rig.ingest("D2", 3, **FIVE)
    await rig.pg_chunk_repo.claim_stale(grace_s=0, claim_timeout_s=600, limit=10)
    with pytest.raises(ChunkBeingDeleted):
        await rig.pipeline.handle(event)
    await compactor_for(rig, claim_timeout_s=0).compact_once()
    await rig.pipeline.handle(event)
    live = {c for c, p in (await rig.points("D2")).items() if p["state"] == "live"}
    assert len(live) == 5
    assert stale_rows(rig, "D2") == {}


async def test_a_claim_not_yet_committed_when_the_commit_looks_is_seen_once_it_commits(
    rig: Rig,
) -> None:
    """The check and the revive are one step because the stale rows are locked while checking.

    Here a compactor has updated the row but not committed. A check that only read the row
    would see it as waiting, go on, and delete the claimed row when the claim commits, and the
    compactor would then delete a point the new version needs.
    """
    stale = await one_chunk_goes_stale(rig, "Y1")
    event = await rig.ingest("Y1", 3, **FIVE)
    with psycopg.connect(rig.api.database.direct_dsn) as compactor:
        compactor.execute("update stale_chunks set deletion_started_at = now() where doc_id = 'Y1'")
        handling = asyncio.create_task(rig.pipeline.handle(event))
        await asyncio.sleep(0.7)  # the event is at the commit, waiting for the row to be free
        assert not handling.done()
        compactor.commit()  # the claim is now real
        with pytest.raises(ChunkBeingDeleted):
            await asyncio.wait_for(handling, timeout=10)
    assert rig.row("Y1")["current"] == 2
    assert stale_rows(rig, "Y1") == {stale: True}
