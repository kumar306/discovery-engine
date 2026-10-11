from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from uuid import UUID

Positions = tuple[int, ...]


@dataclass(frozen=True)
class Hit:
    chunk_id: UUID
    positions: Positions
    moved: bool


@dataclass(frozen=True)
class ChunkDiff:
    hits: tuple[Hit, ...]
    misses: tuple[UUID, ...]
    removed: tuple[UUID, ...]
    revived: tuple[UUID, ...]


def diff_chunks(
    new: Mapping[UUID, Positions],
    current: Mapping[UUID, Positions],
    stale: AbstractSet[UUID],
) -> ChunkDiff:
    # sorting by first position keeps every result in document order, batches stay the same each run
    new_order = sorted(new, key=lambda chunk_id: new[chunk_id][0])
    current_order = sorted(current, key=lambda chunk_id: current[chunk_id][0])
    # moved flags a chunk whose position changed, qdrant payload then needs the new chunk idx
    # unmoved hits only need the version patch, no embedding for any hit
    hits = tuple(
        Hit(chunk_id, new[chunk_id], moved=new[chunk_id] != current[chunk_id])
        for chunk_id in new_order
        if chunk_id in current
    )
    # misses are the only chunks that get embedded and staged
    misses = tuple(chunk_id for chunk_id in new_order if chunk_id not in current)
    # removed chunks go to the stale queue, qdrant deletes them later
    removed = tuple(chunk_id for chunk_id in current_order if chunk_id not in new)
    # tracking revived so those chunks are removed from the stale chunk queue
    revived = tuple(chunk_id for chunk_id in misses if chunk_id in stale)
    return ChunkDiff(hits, misses, removed, revived)
