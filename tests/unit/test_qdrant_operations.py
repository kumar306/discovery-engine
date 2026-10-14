"""What a publish sends to qdrant, and in which order. No server needed."""

import uuid
from typing import Any

import pytest
from qdrant_client import models

from services.embed.qdrant import ChunkIndex, LiveChunk, publish_operations


def ident(number: int) -> uuid.UUID:
    return uuid.UUID(int=number)


def payload_of(operation: models.UpdateOperation) -> dict[str, Any]:
    assert isinstance(operation, models.SetPayloadOperation)
    return operation.set_payload.payload


def ids_of(operation: models.UpdateOperation) -> list[str]:
    assert isinstance(operation, models.SetPayloadOperation)
    where = operation.set_payload.filter
    assert where is not None
    assert where.must is not None
    condition = where.must[0]
    assert isinstance(condition, models.HasIdCondition)
    return [str(i) for i in condition.has_id]


def test_every_live_chunk_gets_state_version_and_position() -> None:
    operations = publish_operations(7, [LiveChunk(ident(1), 0), LiveChunk(ident(2), 3)], [])
    assert [payload_of(op) for op in operations] == [
        {"state": "live", "version": 7, "chunk_idx": 0},
        {"state": "live", "version": 7, "chunk_idx": 3},
    ]
    assert [ids_of(op) for op in operations] == [[str(ident(1))], [str(ident(2))]]


def test_the_new_version_goes_live_before_the_old_one_is_superseded() -> None:
    operations = publish_operations(
        7, [LiveChunk(ident(1), 0), LiveChunk(ident(2), 1)], [ident(8), ident(9)]
    )
    assert [payload_of(op)["state"] for op in operations] == ["live", "live", "superseded"]
    assert ids_of(operations[-1]) == [str(ident(8)), str(ident(9))]
    # superseding does not touch the version or the position of what it hides
    assert payload_of(operations[-1]) == {"state": "superseded"}


def test_nothing_is_superseded_when_there_is_nothing_to_supersede() -> None:
    operations = publish_operations(1, [LiveChunk(ident(1), 0)], [])
    assert len(operations) == 1


async def test_publishing_no_live_chunk_is_refused_before_anything_is_sent() -> None:
    # no client at all: any attempt to send would fail with something other than a ValueError
    index = ChunkIndex(None, "unused")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no live chunks"):
        await index.publish(2, [], [ident(1)])
