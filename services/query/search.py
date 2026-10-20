"""The only code of the query service that talks to Qdrant.

Every request built here carries the condition state == "live", whatever else is asked for, so a
point that is staging, superseded or deleted can never be returned. The two methods differ in what
a result is: grouped_search returns one best chunk per product (Flow S), chunk_search returns the
chunks as they rank (Flow C).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from qdrant_client import AsyncQdrantClient, models

from libs.embedding.collection import QdrantPointState
from services.query.primitives import Hit

type Vector = Sequence[float] | models.SparseVector


@dataclass(frozen=True)
class Filters:
    """What a request may narrow the search to. Keys are already folded with filter_key."""

    brand_key: str | None = None
    color_key: str | None = None
    locale: str | None = None
    # only these products (Flow C, a follow up question about products already shown)
    doc_ids: tuple[str, ...] = ()


def _condition(filters: Filters) -> models.Filter:
    # live first and always, then exactly the filters that were given, nothing else
    must: list[models.Condition] = [
        models.FieldCondition(
            key="state", match=models.MatchValue(value=QdrantPointState.QDRANT_POINT_LIVE.value)
        )
    ]
    for key, value in (
        ("brand_key", filters.brand_key),
        ("color_key", filters.color_key),
        ("locale", filters.locale),
    ):
        if value is not None:
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    if filters.doc_ids:
        must.append(
            models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(filters.doc_ids)))
        )
    return models.Filter(must=must)


def _hit(point: models.ScoredPoint) -> Hit:
    payload = point.payload or {}
    return Hit(
        point_id=str(point.id),
        doc_id=str(payload["doc_id"]),
        chunk_idx=int(payload["chunk_idx"]),
        score=float(point.score),
        text=str(payload["text"]),
        # a point written before titles were stored has none
        title=str(payload.get("title", "")),
        brand=None if payload.get("brand") is None else str(payload["brand"]),
        color=None if payload.get("color") is None else str(payload["color"]),
    )


class QdrantSearch:
    def __init__(self, qdrant_client: AsyncQdrantClient, qdrant_collection: str) -> None:
        self._qdrant_client = qdrant_client
        self._qdrant_collection = qdrant_collection

    async def grouped_search(
        self, using: Literal["dense", "bm25"], vector: Vector, filters: Filters, limit: int
    ) -> list[Hit]:
        """The best chunk of each of the top `limit` products, best product first."""
        answer = await self._qdrant_client.query_points_groups(
            self._qdrant_collection,
            query=_as_query(vector),
            using=using,
            group_by="doc_id",
            group_size=1,
            limit=limit,
            query_filter=_condition(filters),
            with_payload=True,
        )
        return [_hit(group.hits[0]) for group in answer.groups if group.hits]

    async def chunk_search(
        self, using: Literal["dense", "bm25"], vector: Vector, filters: Filters, limit: int
    ) -> list[Hit]:
        """The top `limit` chunks as they rank, several may belong to one product."""
        answer = await self._qdrant_client.query_points(
            self._qdrant_collection,
            query=_as_query(vector),
            using=using,
            limit=limit,
            query_filter=_condition(filters),
            with_payload=True,
        )
        return [_hit(point) for point in answer.points]


def _as_query(vector: Vector) -> list[float] | models.SparseVector:
    return vector if isinstance(vector, models.SparseVector) else list(vector)
