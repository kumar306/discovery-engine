"""The pipeline against real Postgres, Qdrant and S3, with the hash stub for the model.

Documents come in through the real ingest api, so the events are exactly what the relay would
carry. The stub embedder keeps these tests about the pipeline: order, versions, crash safety.
"""

from typing import Any

import pytest
from qdrant_client import models

from libs.resilience.retry import PermanentError
from libs.schemas import UpsertEvent
from libs.schemas.hashing import content_sha
from services.embed.compose import ComposeError
from services.embed.embedder import EmbedTransient
from services.embed.pipeline import KafkaEventResult, Pipeline
from services.ingest.storage import MissingObject, StorageUnavailable
from tests.support.hash_embedder import HashEmbedder

from .conftest import BASE, TEST_DIM, Rig, first_words, words

DONE, DROPPED = KafkaEventResult.KAFKA_EVENT_APPLIED, KafkaEventResult.KAFKA_EVENT_SKIPPED


class Recorder:
    """Stands between the pipeline and the real thing and writes down every call, in order."""

    def __init__(self, inner: Any, log: list[str]) -> None:
        self._inner = inner
        self._log = log

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._inner, name)
        if not callable(target):
            return target

        async def call(*args: Any, **kwargs: Any) -> Any:
            self._log.append(name)
            return await target(*args, **kwargs)

        return call


class Breaker:
    """Fails the named call before it runs, which is the same as a crash after the step before."""

    def __init__(self, inner: Any, name: str, error: BaseException) -> None:
        self._inner = inner
        self._name = name
        self._error = error

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._inner, name)
        if not callable(target) or name != self._name:
            return target

        async def call(*args: Any, **kwargs: Any) -> Any:
            raise self._error

        return call


class RaceBefore:
    """Runs another action just before the named call, the way a second worker would."""

    def __init__(self, inner: Any, name: str, action: Any) -> None:
        self._inner = inner
        self._name = name
        self._action = action

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._inner, name)
        if not callable(target) or name != self._name:
            return target

        async def call(*args: Any, **kwargs: Any) -> Any:
            await self._action()
            return await target(*args, **kwargs)

        return call


# a document becomes searchable


async def test_a_new_document_is_embedded_staged_committed_and_published(rig: Rig) -> None:
    event = await rig.ingest("P1", 1)
    assert await rig.pipeline.handle(event) is DONE
    row = rig.row("P1")
    assert (row["current"], row["indexed"], row["deleted"]) == (1, 1, False)
    snapshot = await rig.pg_chunk_repo.snapshot("esci", "P1")
    assert sorted(snapshot.current.values()) == [(0,), (1,)]
    points = await rig.points("P1")
    assert set(points) == set(snapshot.current)
    for chunk_id, positions in snapshot.current.items():
        payload = points[chunk_id]
        assert (payload["state"], payload["version"], payload["chunk_idx"]) == (
            "live",
            1,
            positions[0],
        )
        assert payload["model_version"] == rig.embed_settings.model_version
        assert payload["doc_id"] == "P1"
    # one call, with exactly the two chunk texts
    assert len(rig.embedder.calls) == 1
    assert first_words(rig.embedder.calls[0]) == ["a0", "d0"]


async def test_the_payload_carries_the_product_fields_a_search_filters_on(rig: Rig) -> None:
    fields = {**BASE, "brand": "Acme", "color": "black"}
    await rig.pipeline.handle(await rig.ingest("P1", 1, **fields))
    payloads = (await rig.points("P1")).values()
    assert {(p["locale"], p["brand"], p["color"]) for p in payloads} == {("us", "Acme", "black")}


async def test_an_update_embeds_only_the_chunk_that_changed(rig: Rig) -> None:
    await rig.pipeline.handle(await rig.ingest("P1", 1))
    before = await rig.pg_chunk_repo.snapshot("esci", "P1")
    # only the description changes
    event = await rig.ingest("P1", 2, **{**BASE, "description": words(30, "x")})
    assert await rig.pipeline.handle(event) is DONE
    assert len(rig.embedder.calls) == 2
    assert first_words(rig.embedder.calls[1]) == ["x0"]  # the bullets chunk was not embedded again
    after = await rig.pg_chunk_repo.snapshot("esci", "P1")
    kept = set(before.current) & set(after.current)
    dropped = set(before.current) - set(after.current)
    assert len(kept) == 1  # the bullets chunk is the same point
    assert len(dropped) == 1
    assert after.stale == dropped
    points = await rig.points("P1")
    (kept_id,) = kept
    (dropped_id,) = dropped
    assert (points[kept_id]["state"], points[kept_id]["version"]) == ("live", 2)
    assert (points[dropped_id]["state"], points[dropped_id]["version"]) == ("superseded", 1)
    live = {cid for cid, p in points.items() if p["state"] == "live"}
    assert live == set(after.current)
    row = rig.row("P1")
    assert (row["current"], row["indexed"]) == (2, 2)


async def test_a_chunk_that_only_moves_gets_its_new_position_without_being_embedded(
    rig: Rig,
) -> None:
    await rig.pipeline.handle(await rig.ingest("P1", 1))
    # a third bullet pushes the unchanged description chunk from position 1 to position 2
    more = {**BASE, "bullets": [*BASE["bullets"], words(20, "c")]}
    await rig.pipeline.handle(await rig.ingest("P1", 2, **more))
    assert [first_words(call) for call in rig.embedder.calls] == [["a0", "d0"], ["c0"]]
    points = await rig.points("P1")
    by_text = {p["text"].split("\n")[1].split()[0]: p for p in points.values()}
    assert (by_text["d0"]["chunk_idx"], by_text["d0"]["version"]) == (2, 2)
    assert (by_text["c0"]["chunk_idx"], by_text["a0"]["chunk_idx"]) == (1, 0)
    assert len(points) == 3  # the moved chunk is the same point, nothing was added for it


# events that must change nothing


async def test_an_older_event_is_dropped_before_anything_is_read_or_computed(rig: Rig) -> None:
    first = await rig.ingest("P1", 1)
    second = await rig.ingest("P1", 2, **{**BASE, "description": words(30, "x")})
    await rig.pipeline.handle(second)
    log: list[str] = []
    guarded = rig.pipeline_with(
        s3_documents=Recorder(rig.api.s3_store, log), embedder=Recorder(rig.embedder, log)
    )
    before = await rig.points("P1")
    assert await guarded.handle(first) is DROPPED
    assert log == []  # no read from s3 and no embedding for a stale event
    assert await rig.points("P1") == before
    assert rig.row("P1")["current"] == 2


async def test_the_same_event_delivered_twice_changes_nothing_the_second_time(rig: Rig) -> None:
    event = await rig.ingest("P1", 1)
    await rig.pipeline.handle(event)
    before = (await rig.points("P1"), await rig.pg_chunk_repo.snapshot("esci", "P1"))
    calls = len(rig.embedder.calls)
    assert await rig.pipeline.handle(event) is DROPPED
    assert (await rig.points("P1"), await rig.pg_chunk_repo.snapshot("esci", "P1")) == before
    assert len(rig.embedder.calls) == calls


async def test_the_pipeline_calls_its_parts_in_the_order_that_keeps_readers_safe(rig: Rig) -> None:
    log: list[str] = []
    recorded = rig.pipeline_with(
        pg_chunk_repo=Recorder(rig.pg_chunk_repo, log),
        qdrant_index=Recorder(rig.qdrant_index, log),
        s3_documents=Recorder(rig.api.s3_store, log),
        embedder=Recorder(rig.embedder, log),
    )
    await recorded.handle(await rig.ingest("P1", 1))
    # nothing visible before the commit, the commit before the publish, the mark last
    assert log == [
        "gate_state",
        "get",
        "snapshot",
        "embed",
        "stage",
        "commit_version",
        "snapshot",
        "publish",
        "mark_indexed",
    ]


# crashes: the same message comes back and finishes the job


async def test_a_crash_after_staging_leaves_the_document_unchanged_and_the_retry_finishes(
    rig: Rig,
) -> None:
    event = await rig.ingest("P1", 1)
    crashing = rig.pipeline_with(
        pg_chunk_repo=Breaker(rig.pg_chunk_repo, "commit_version", RuntimeError("died"))
    )
    with pytest.raises(RuntimeError, match="died"):
        await crashing.handle(event)
    await rig.untouched("P1")
    staged = await rig.points("P1")
    assert len(staged) == 2
    assert {p["state"] for p in staged.values()} == {"staging"}
    assert await rig.pipeline.handle(event) is DONE
    points = await rig.points("P1")
    assert set(points) == set(staged)  # the same two points, not four
    assert {p["state"] for p in points.values()} == {"live"}


async def test_a_crash_between_the_commit_and_the_publish_is_finished_without_embedding(
    rig: Rig,
) -> None:
    event = await rig.ingest("P1", 1)
    crashing = rig.pipeline_with(
        qdrant_index=Breaker(rig.qdrant_index, "publish", RuntimeError("died"))
    )
    with pytest.raises(RuntimeError, match="died"):
        await crashing.handle(event)
    row = rig.row("P1")
    assert (row["current"], row["indexed"]) == (1, None)  # committed, not live
    assert {p["state"] for p in (await rig.points("P1")).values()} == {"staging"}
    calls = len(rig.embedder.calls)
    assert await rig.pipeline.handle(event) is DONE
    assert len(rig.embedder.calls) == calls  # every chunk was already committed, nothing to embed
    assert {p["state"] for p in (await rig.points("P1")).values()} == {"live"}
    row = rig.row("P1")
    assert (row["current"], row["indexed"]) == (1, 1)


# errors: nothing is committed, and the error says whose fault it is


async def test_a_document_that_does_not_match_its_sha_is_a_permanent_error(rig: Rig) -> None:
    event = await rig.ingest("P1", 1)
    assert isinstance(event, UpsertEvent)
    tampered = event.model_copy(update={"content_sha": "0" * 64})
    with pytest.raises(PermanentError, match="sha"):
        await rig.pipeline.handle(tampered)
    await rig.untouched("P1")


async def test_an_event_that_points_at_nothing_is_a_permanent_error(rig: Rig) -> None:
    event = await rig.ingest("P1", 1)
    assert isinstance(event, UpsertEvent)
    lost = event.model_copy(update={"s3_key": "esci/NOPE/v1.json"})
    with pytest.raises(MissingObject):
        await rig.pipeline.handle(lost)
    await rig.untouched("P1")


async def test_a_stored_document_that_is_not_valid_is_a_permanent_error_not_a_value_error(
    rig: Rig,
) -> None:
    event = await rig.ingest("P1", 1)
    assert isinstance(event, UpsertEvent)
    garbage = b'{"corpus": "esci", "title": 7'
    await rig.api.s3_store.put_if_absent("esci/P1/garbage.json", garbage, content_sha(garbage))
    broken = event.model_copy(
        update={"s3_key": "esci/P1/garbage.json", "content_sha": content_sha(garbage)}
    )
    with pytest.raises(PermanentError, match="not a valid document") as caught:
        await rig.pipeline.handle(broken)
    assert not isinstance(caught.value, ValueError)  # decide() would not trust that as bad input
    await rig.untouched("P1")


async def test_a_document_that_belongs_to_another_event_is_a_permanent_error(rig: Rig) -> None:
    one = await rig.ingest("P1", 1)
    other = await rig.ingest("P2", 1)
    assert isinstance(one, UpsertEvent)
    assert isinstance(other, UpsertEvent)
    mixed = one.model_copy(update={"s3_key": other.s3_key, "content_sha": other.content_sha})
    with pytest.raises(PermanentError, match="another document"):
        await rig.pipeline.handle(mixed)
    await rig.untouched("P1")


async def test_a_product_that_cannot_be_composed_is_a_permanent_error(rig: Rig) -> None:
    # the stub takes 20 tokens here and 16 are kept as a margin, so a long header does not fit
    tiny = rig.pipeline_with(embedder=HashEmbedder(dense_dim=TEST_DIM, max_tokens=20))
    event = await rig.ingest("P1", 1, title=words(10, "t"))
    with pytest.raises(ComposeError):
        await tiny.handle(event)
    await rig.untouched("P1")


async def test_a_failing_embedder_commits_and_stages_nothing(rig: Rig) -> None:
    event = await rig.ingest("P1", 1)
    broken = rig.pipeline_with(embedder=Breaker(rig.embedder, "embed", EmbedTransient("slow")))
    with pytest.raises(EmbedTransient):
        await broken.handle(event)
    await rig.untouched("P1")
    assert await rig.points("P1") == {}
    assert await rig.pipeline.handle(event) is DONE  # and the same message works when it clears


async def test_an_s3_outage_commits_nothing_and_is_not_a_missing_document(rig: Rig) -> None:
    event = await rig.ingest("P1", 1)
    down = Breaker(rig.api.s3_store, "get", StorageUnavailable("s3 is down"))
    with pytest.raises(StorageUnavailable):
        await rig.pipeline_with(s3_documents=down).handle(event)
    await rig.untouched("P1")


async def test_a_qdrant_outage_while_staging_commits_nothing(rig: Rig) -> None:
    event = await rig.ingest("P1", 1)
    down = Breaker(rig.qdrant_index, "stage", ConnectionError("qdrant is down"))
    with pytest.raises(ConnectionError):
        await rig.pipeline_with(qdrant_index=down).handle(event)
    await rig.untouched("P1")
    assert await rig.pipeline.handle(event) is DONE


# deletes


async def test_a_delete_hides_the_product_queues_its_chunks_and_leaves_a_tombstone(
    rig: Rig,
) -> None:
    await rig.pipeline.handle(await rig.ingest("P1", 1))
    before = await rig.pg_chunk_repo.snapshot("esci", "P1")
    event = await rig.remove("P1", 2)
    assert await rig.pipeline.handle(event) is DONE
    assert {p["state"] for p in (await rig.points("P1")).values()} == {"deleted"}
    snapshot = await rig.pg_chunk_repo.snapshot("esci", "P1")
    assert (snapshot.current, snapshot.stale) == ({}, frozenset(before.current))
    row = rig.row("P1")
    assert (row["current"], row["indexed"], row["deleted"]) == (2, 2, True)
    assert await rig.pipeline.handle(event) is DROPPED  # delivered again, nothing left to do


async def test_a_crash_between_the_tombstone_and_the_hiding_is_finished_on_the_retry(
    rig: Rig,
) -> None:
    await rig.pipeline.handle(await rig.ingest("P1", 1))
    event = await rig.remove("P1", 2)
    crashing = rig.pipeline_with(
        qdrant_index=Breaker(rig.qdrant_index, "mark_deleted", RuntimeError("died"))
    )
    with pytest.raises(RuntimeError, match="died"):
        await crashing.handle(event)
    assert {p["state"] for p in (await rig.points("P1")).values()} == {"live"}  # still visible
    assert rig.row("P1")["indexed"] == 1
    assert await rig.pipeline.handle(event) is DONE
    assert {p["state"] for p in (await rig.points("P1")).values()} == {"deleted"}
    assert rig.row("P1")["indexed"] == 2


async def test_a_delete_of_a_document_that_was_never_indexed_still_leaves_the_tombstone(
    rig: Rig,
) -> None:
    first = await rig.ingest("P1", 1)
    event = await rig.remove("P1", 2)
    assert await rig.pipeline.handle(event) is DONE
    row = rig.row("P1")
    assert (row["current"], row["indexed"], row["deleted"]) == (2, 2, True)
    # and the upload that was waiting behind it can no longer bring the product back
    assert await rig.pipeline.handle(first) is DROPPED
    assert await rig.points("P1") == {}


async def test_a_newer_upload_after_a_delete_brings_the_product_back(rig: Rig) -> None:
    await rig.pipeline.handle(await rig.ingest("P1", 1))
    await rig.pipeline.handle(await rig.remove("P1", 2))
    assert await rig.pipeline.handle(await rig.ingest("P1", 3)) is DONE
    points = await rig.points("P1")
    assert {p["state"] for p in points.values()} == {"live"}
    assert len(points) == 2  # the same chunks, so the same two points came back
    snapshot = await rig.pg_chunk_repo.snapshot("esci", "P1")
    assert snapshot.stale == frozenset()  # revived, so no longer waiting to be deleted
    row = rig.row("P1")
    assert (row["current"], row["indexed"], row["deleted"]) == (3, 3, False)


# races and leftovers that only show up in unlucky timing


async def test_losing_the_race_at_the_commit_drops_the_event_and_publishes_nothing(
    rig: Rig,
) -> None:
    first = await rig.ingest("P1", 1)
    second = await rig.ingest("P1", 2, **{**BASE, "description": words(30, "x")})
    log: list[str] = []
    # between the gate and the commit another worker applies version 2 completely
    racing = rig.pipeline_with(
        pg_chunk_repo=RaceBefore(
            rig.pg_chunk_repo, "commit_version", lambda: rig.pipeline.handle(second)
        ),
        qdrant_index=Recorder(rig.qdrant_index, log),
    )
    assert await racing.handle(first) is DROPPED
    assert "publish" not in log  # the loser never makes anything visible
    row = rig.row("P1")
    assert (row["current"], row["indexed"]) == (2, 2)
    snapshot = await rig.pg_chunk_repo.snapshot("esci", "P1")
    live = {cid: p for cid, p in (await rig.points("P1")).items() if p["state"] == "live"}
    assert set(live) == set(snapshot.current)
    assert {p["version"] for p in live.values()} == {2}


async def test_a_stale_delete_never_takes_the_document_lock(rig: Rig) -> None:
    await rig.pipeline.handle(await rig.ingest("P1", 1))
    old_delete = await rig.remove("P1", 2)
    await rig.pipeline.handle(await rig.ingest("P1", 3))  # a newer upload was applied first
    log: list[str] = []
    guarded = rig.pipeline_with(pg_chunk_repo=Recorder(rig.pg_chunk_repo, log))
    assert await guarded.handle(old_delete) is DROPPED
    assert log == ["gate_state"]  # judged without a lock, commit_delete was never reached


async def test_a_chunk_that_returns_after_its_point_was_deleted_is_embedded_and_staged_again(
    rig: Rig,
) -> None:
    await rig.pipeline.handle(await rig.ingest("P1", 1))
    await rig.pipeline.handle(await rig.remove("P1", 2))
    queued = (await rig.pg_chunk_repo.snapshot("esci", "P1")).stale
    # the compactor deleted the points and stopped before it cleared the queue rows
    await rig.qdrant_client.delete(
        rig.embed_settings.qdrant_collection,
        points_selector=models.PointIdsList(points=[str(cid) for cid in queued]),
        wait=True,
    )
    assert await rig.points("P1") == {}
    calls = len(rig.embedder.calls)
    assert await rig.pipeline.handle(await rig.ingest("P1", 3)) is DONE
    assert len(rig.embedder.calls) == calls + 1  # not trusted to exist, so embedded again
    points = await rig.points("P1")
    assert len(points) == 2
    assert {p["state"] for p in points.values()} == {"live"}


async def test_a_new_model_version_makes_every_chunk_new(rig: Rig) -> None:
    await rig.pipeline.handle(await rig.ingest("P1", 1))
    before = await rig.pg_chunk_repo.snapshot("esci", "P1")
    newer = rig.embed_settings.model_copy(update={"model_revision": 2})
    assert newer.model_version != rig.embed_settings.model_version
    other_model = Pipeline(
        rig.pg_chunk_repo, rig.qdrant_index, rig.api.s3_store, rig.embedder, newer
    )
    # the same content in a newer version, so only the model version differs
    assert await other_model.handle(await rig.ingest("P1", 2)) is DONE
    assert [len(call) for call in rig.embedder.calls] == [2, 2]  # everything embedded again
    after = await rig.pg_chunk_repo.snapshot("esci", "P1")
    assert set(after.current).isdisjoint(before.current)
    assert after.stale == frozenset(before.current)
    live = [p for p in (await rig.points("P1")).values() if p["state"] == "live"]
    assert {p["model_version"] for p in live} == {newer.model_version}
