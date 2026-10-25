"""The quality job: reuse the index when it holds the dataset, build it when it is missing."""

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eval.ladder import live_product_ids
from eval.quality_job import IndexMismatch, ensure_index
from tests.support.hash_embedder import HashEmbedder
from tests.unit.test_index_builder import DIM, FakeQdrant, dataset, products, settings


class ExistingQdrant(FakeQdrant):
    """A Qdrant whose collection may already exist and holds these product ids, one per chunk."""

    def __init__(self, doc_ids: Sequence[str] | None = None, page: int = 2) -> None:
        super().__init__()
        self._exists = doc_ids is not None
        self._doc_ids = list(doc_ids or [])
        self._page = page
        self.scroll_filters: list[Any] = []

    async def collection_exists(self, name: str) -> bool:
        return self._exists

    async def scroll(
        self,
        name: str,
        *,
        scroll_filter: Any,
        limit: int,
        with_payload: Any,
        with_vectors: bool,
        offset: int | None,
    ) -> tuple[list[Any], int | None]:
        self.scroll_filters.append(scroll_filter)
        start = offset or 0
        records = [
            SimpleNamespace(payload={"doc_id": doc_id})
            for doc_id in self._doc_ids[start : start + self._page]
        ]
        following = start + self._page
        return records, (following if following < len(self._doc_ids) else None)


def embedder() -> HashEmbedder:
    return HashEmbedder(dense_dim=DIM, max_tokens=64)


# ---------------- the product ids of a collection


async def test_every_page_of_the_scroll_is_read_and_only_live_points_are_asked_for() -> None:
    fake = ExistingQdrant(["A", "B", "C", "D", "E"], page=2)
    assert await live_product_ids(fake, "any") == {"A", "B", "C", "D", "E"}  # type: ignore[arg-type]
    assert len(fake.scroll_filters) == 3
    assert fake.scroll_filters[0].must[0].match.value == "live"


async def test_several_chunks_of_one_product_count_once() -> None:
    fake = ExistingQdrant(["A", "A", "B"], page=10)
    assert await live_product_ids(fake, "any") == {"A", "B"}  # type: ignore[arg-type]


# ---------------- ensure_index


async def test_a_missing_collection_is_built(tmp_path: Path) -> None:
    dataset(tmp_path / "set", products())
    fake = ExistingQdrant(None)
    how = await ensure_index(tmp_path / "set", settings(), fake, embedder())  # type: ignore[arg-type]
    assert how == "built"
    assert {point.payload["doc_id"] for point in fake.points.values()} == {"P0", "P1", "P4"}


async def test_a_collection_that_holds_exactly_the_dataset_is_reused_and_nothing_is_embedded(
    tmp_path: Path,
) -> None:
    dataset(tmp_path / "set", products())
    fake = ExistingQdrant(["P0", "P0", "P1", "P4"])
    model = embedder()
    how = await ensure_index(tmp_path / "set", settings(), fake, model)  # type: ignore[arg-type]
    assert how == "reused"
    assert model.calls == []
    assert fake.calls == []
    assert fake.created == []


async def test_a_collection_with_a_product_the_dataset_does_not_have_is_refused_not_rebuilt(
    tmp_path: Path,
) -> None:
    dataset(tmp_path / "set", products())
    fake = ExistingQdrant(["P0", "P1", "P4", "X9"])
    with pytest.raises(IndexMismatch, match=r"1 in the collection are not in the dataset"):
        await ensure_index(tmp_path / "set", settings(), fake, embedder())  # type: ignore[arg-type]
    # a collection of someone else is never deleted or written to
    assert fake.calls == []
    assert fake.created == []


async def test_an_incomplete_collection_is_refused_and_says_how_many_are_missing(
    tmp_path: Path,
) -> None:
    dataset(tmp_path / "set", products())
    fake = ExistingQdrant(["P0", "P1"])
    with pytest.raises(IndexMismatch, match=r"1 of the dataset are missing"):
        await ensure_index(tmp_path / "set", settings(), fake, embedder())  # type: ignore[arg-type]
    assert fake.calls == []
