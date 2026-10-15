"""One look at everything the stores hold about a document, for the crash tests."""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import psycopg

from .conftest import Rig, document


@dataclass(frozen=True)
class Observed:
    """Everything the stores hold about one document at one moment."""

    current: int | None
    indexed: int | None
    deleted: bool
    chunks: dict[UUID, tuple[str, tuple[int, ...]]]  # chunk id -> (content hash, positions)
    stale: frozenset[UUID]
    points: dict[UUID, dict[str, Any]]

    @property
    def live(self) -> set[UUID]:
        return {cid for cid, point in self.points.items() if point["state"] == "live"}

    @property
    def shape(self) -> tuple[Any, ...]:
        # without chunk ids, which depend on the document, so two documents can be compared
        return (
            self.current,
            self.indexed,
            self.deleted,
            tuple(sorted(self.chunks.values())),
            tuple(
                sorted(
                    (p["content_hash"], p["state"], p["version"], p["chunk_idx"])
                    for p in self.points.values()
                )
            ),
        )


async def observe(rig: Rig, doc_id: str) -> Observed:
    row = document(rig.api.database.direct_dsn, doc_id)
    if row is None:  # the document has never been written, there is nothing to see
        return Observed(None, None, False, {}, frozenset(), {})
    with psycopg.connect(rig.api.database.direct_dsn) as conn:
        rows = conn.execute(
            "select chunk_id, content_hash, positions from chunks "
            "where corpus = 'esci' and doc_id = %s",
            (doc_id,),
        ).fetchall()
    snapshot = await rig.pg_chunk_repo.snapshot("esci", doc_id)
    return Observed(
        current=row["current"],
        indexed=row["indexed"],
        deleted=row["deleted"],
        chunks={cid: (content_hash, tuple(positions)) for cid, content_hash, positions in rows},
        stale=snapshot.stale,
        points=await rig.points(doc_id),
    )
