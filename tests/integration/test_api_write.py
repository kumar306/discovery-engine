"""The ingest api end to end: real Postgres through PgBouncer, real S3 (Moto), no fakes."""

import json

from libs.schemas import UpsertEvent, doc_event_adapter
from libs.schemas.hashing import canonical_bytes, content_sha
from services.ingest.events import EVENT_SCHEMA, prepare_upsert
from services.ingest.storage import PutResult

from .conftest import NOW, Api, document, outbox, request_for


async def test_a_new_product_is_stored_in_s3_recorded_in_postgres_and_queued_for_kafka(
    api: Api,
) -> None:
    response = await api.post("D1", 1)
    assert response.status_code == 202
    assert response.json() == {
        "request_id": response.headers["x-request-id"],
        "doc_id": "D1",
        "version": 1,
        "accepted_at": NOW.isoformat(),
    }
    expected = request_for("D1", 1)
    assert api.object_bytes("esci/D1/v1.json") == canonical_bytes(expected)
    row = document(api.database.direct_dsn, "D1")
    assert row is not None
    assert (row["latest"], row["sha"], row["current"], row["indexed"]) == (
        1,
        content_sha(canonical_bytes(expected)),
        None,
        None,
    )
    ((_, doc_id, request_id, schema, payload, published_at),) = outbox(api.database.direct_dsn)
    assert (doc_id, request_id, schema, published_at) == (
        "D1",
        response.json()["request_id"],
        EVENT_SCHEMA,
        None,
    )
    event = doc_event_adapter.validate_json(json.dumps(payload))
    assert isinstance(event, UpsertEvent)
    prepared = prepare_upsert(expected, NOW)
    assert (event.s3_key, event.content_sha, event.size_bytes) == (
        "esci/D1/v1.json",
        prepared.content_sha,
        len(prepared.data),
    )


async def test_a_new_version_is_a_new_object_and_a_new_event(api: Api) -> None:
    assert (await api.post("D1", 1)).status_code == 202
    assert (await api.post("D1", 2, title="Better Mouse")).status_code == 202
    assert api.object_exists("esci/D1/v1.json")
    assert api.object_exists("esci/D1/v2.json")
    assert [event.version for event in api.events("D1")] == [1, 2]
    row = document(api.database.direct_dsn, "D1")
    assert row is not None
    assert row["latest"] == 2


async def test_repeating_a_completed_request_is_accepted_but_writes_nothing_more(api: Api) -> None:
    first = await api.post("D1", 1)
    before = api.object_bytes("esci/D1/v1.json")
    again = await api.post("D1", 1)
    assert (first.status_code, again.status_code) == (202, 202)
    assert again.json()["request_id"] != first.json()["request_id"]
    assert api.object_bytes("esci/D1/v1.json") == before
    assert len(outbox(api.database.direct_dsn)) == 1


async def test_a_retry_after_a_crash_between_s3_and_postgres_completes_the_write(api: Api) -> None:
    # the first attempt got as far as s3 and then died: the object exists, postgres knows nothing
    prepared = prepare_upsert(request_for("D1", 1), NOW)
    assert (
        await api.store.put_if_absent(prepared.key, prepared.data, prepared.content_sha)
        is PutResult.CREATED
    )
    assert document(api.database.direct_dsn, "D1") is None
    response = await api.post("D1", 1)
    assert response.status_code == 202
    row = document(api.database.direct_dsn, "D1")
    assert row is not None
    assert row["latest"] == 1
    assert len(api.events("D1")) == 1


async def test_an_older_version_is_refused_and_nothing_is_written_for_it(api: Api) -> None:
    await api.post("D1", 5)
    response = await api.post("D1", 4, title="Older")
    assert response.status_code == 409
    body = response.json()
    assert (body["error"], body["current_version"]) == ("stale_version", 5)
    assert body["request_id"] == response.headers["x-request-id"]
    assert not api.object_exists("esci/D1/v4.json")
    assert len(outbox(api.database.direct_dsn)) == 1


async def test_the_same_version_with_other_content_is_refused_and_the_stored_object_stays(
    api: Api,
) -> None:
    await api.post("D1", 5, title="First Title")
    stored = api.object_bytes("esci/D1/v5.json")
    response = await api.post("D1", 5, title="Second Title")
    assert response.status_code == 409
    assert (response.json()["error"], response.json()["current_version"]) == ("content_conflict", 5)
    assert api.object_bytes("esci/D1/v5.json") == stored
    assert len(outbox(api.database.direct_dsn)) == 1


async def test_a_leftover_object_with_other_content_blocks_the_version_without_a_row(
    api: Api,
) -> None:
    # a crashed attempt left other content at this version, and postgres has no row for it
    leftover = prepare_upsert(request_for("D1", 1, title="Leftover Title"), NOW)
    await api.store.put_if_absent(leftover.key, leftover.data, leftover.content_sha)
    response = await api.post("D1", 1, title="New Title")
    assert response.status_code == 409
    assert (response.json()["error"], response.json()["current_version"]) == (
        "content_conflict",
        None,
    )
    assert api.object_bytes("esci/D1/v1.json") == leftover.data
    assert document(api.database.direct_dsn, "D1") is None
    assert outbox(api.database.direct_dsn) == []


# DELETE


async def test_a_delete_writes_no_object_leaves_a_tombstone_row_and_queues_the_event(
    api: Api,
) -> None:
    response = await api.delete("D1", 4)
    assert response.status_code == 202
    assert response.json()["version"] == 4
    assert api.events("D1")[0].op == "delete"
    row = document(api.database.direct_dsn, "D1")
    assert row is not None
    # ingest records the newest accepted version, marking the document deleted is for the worker
    assert (row["latest"], row["sha"], row["deleted"]) == (4, None, False)
    assert not api.object_exists("esci/D1/v4.json")
    assert len(outbox(api.database.direct_dsn)) == 1


async def test_a_delete_after_an_upsert_then_a_retry_then_a_resurrecting_upsert(api: Api) -> None:
    assert (await api.post("D1", 1)).status_code == 202
    assert (await api.delete("D1", 2)).status_code == 202
    assert (await api.delete("D1", 2)).status_code == 202  # a retry changes nothing
    assert [event.op for event in api.events("D1")] == ["upsert", "delete"]
    assert (await api.post("D1", 3, title="Back Again")).status_code == 202
    assert [event.op for event in api.events("D1")] == ["upsert", "delete", "upsert"]


async def test_deletes_that_arrive_too_late_or_collide_with_an_upsert_are_refused(
    api: Api,
) -> None:
    await api.post("D1", 5)
    stale = await api.delete("D1", 4)
    assert stale.status_code == 409
    assert (stale.json()["error"], stale.json()["current_version"]) == ("stale_version", 5)
    collision = await api.delete("D1", 5)  # an upsert already holds version 5
    assert collision.status_code == 409
    assert collision.json()["error"] == "content_conflict"
    assert len(outbox(api.database.direct_dsn)) == 1
