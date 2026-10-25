"""The index builder: a dataset into a Qdrant collection with the production parts."""

import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from eval.dataset import JudgementRow, Product, QueryRow, Selection, SelectionConfig, write_dataset
from eval.index_builder import BuildStopped, build_index
from libs.embedding.model import Embedding, EmbedPermanent
from libs.schemas.hashing import chunk_id
from services.embed.settings import EmbedSettings
from tests.support.hash_embedder import HashEmbedder

REVISION = 3
DIM = 4


def words(count: int, tag: str) -> str:
    return " ".join(f"{tag}{n}" for n in range(count))


class FakeQdrant:
    """Records what the builder sends and applies the payload updates the way Qdrant would."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.created: list[str] = []
        self.points: dict[str, Any] = {}

    async def collection_exists(self, name: str) -> bool:
        return False

    async def create_collection(self, name: str, **kwargs: Any) -> None:
        self.created.append(name)

    async def create_payload_index(self, name: str, field: str, schema: Any) -> None:
        return None

    async def upsert(
        self, name: str, points: Sequence[Any], *, wait: bool, update_mode: Any
    ) -> None:
        self.calls.append(("upsert", len(points)))
        for point in points:
            self.points.setdefault(str(point.id), point)

    async def batch_update_points(
        self, name: str, *, update_operations: Sequence[Any], wait: bool
    ) -> None:
        self.calls.append(("publish", len(update_operations)))
        for operation in update_operations:
            ids = operation.set_payload.filter.must[0].has_id
            for point_id in ids:
                self.points[point_id].payload.update(operation.set_payload.payload)


def products() -> list[Product]:
    return [
        # two chunks under the stub's 48 token budget: the bullets, then the description
        Product(
            "P0",
            "us",
            "Acme mouse",
            description=words(30, "d"),
            bullet_point=words(20, "a") + "\n" + words(20, "b"),
            brand="Acme",
            color="Black",
        ),
        Product("P1", "us", "Plain pen", None, None, None, None),
        # a text with U+2028 inside, which splitlines would cut in two
        Product("P4", "us", "wire" + chr(0x2028) + "less pad", None, None, None, None),
    ]


# the three ways a product cannot be indexed, each stops the build
# breaks the api rules (an empty title)
BREAKS_THE_API_RULES = Product("P2", "us", "", None, None, None, None)
# a header that alone exceeds the token budget, cannot be composed
HEADER_TOO_LONG = Product("P3", "us", words(100, "t"), None, None, None, None)
# a text the model refuses to embed
REFUSED_BY_THE_MODEL = Product("P5", "us", "POISON item", None, None, None, None)


def dataset(directory: Path, items: list[Product]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    selection = Selection(
        (QueryRow(1, "mouse", "validation"),),
        tuple(JudgementRow(1, item.product_id, "E") for item in items),
        tuple(sorted(item.product_id for item in items)),
    )
    write_dataset(directory, selection, items, SelectionConfig(seed=1), REVISION)


def settings() -> EmbedSettings:
    return EmbedSettings(qdrant_collection="built", dense_dim=DIM)


class PoisonedEmbedder(HashEmbedder):
    """Refuses one text, the way the model does for a text it cannot tokenize."""

    async def embed(self, texts: Sequence[str], timeout_s: float) -> list[Embedding]:
        for index, text in enumerate(texts):
            if "POISON" in text:
                raise EmbedPermanent(index, ValueError("cannot tokenize"))
        return await super().embed(texts, timeout_s)


@pytest.fixture
def built(tmp_path: Path) -> Path:
    dataset(tmp_path / "set", products())
    return tmp_path / "set"


async def test_every_composable_product_is_staged_and_published_with_the_production_ids(
    built: Path,
) -> None:
    fake = FakeQdrant()
    report = await build_index(built, settings(), fake, HashEmbedder(dense_dim=DIM, max_tokens=64))  # type: ignore[arg-type]
    assert (report.products, report.chunks) == (3, 4)
    # P0 has two chunks, P1 and P4 one each
    by_doc: dict[str, list[Any]] = {}
    for point in fake.points.values():
        by_doc.setdefault(point.payload["doc_id"], []).append(point)
    assert {doc: len(found) for doc, found in by_doc.items()} == {"P0": 2, "P1": 1, "P4": 1}
    for point in fake.points.values():
        payload = point.payload
        assert uuid.UUID(str(point.id)) == chunk_id(
            payload["doc_id"], settings().model_version, payload["content_hash"]
        )
        assert (payload["state"], payload["version"]) == ("live", REVISION)
        assert payload["model_version"] == settings().model_version


async def test_the_payload_has_the_fields_the_search_filters_and_shows(built: Path) -> None:
    fake = FakeQdrant()
    await build_index(built, settings(), fake, HashEmbedder(dense_dim=DIM, max_tokens=64))  # type: ignore[arg-type]
    first = next(p.payload for p in fake.points.values() if p.payload["doc_id"] == "P0")
    assert (first["locale"], first["title"], first["brand"], first["color"]) == (
        "us",
        "Acme mouse",
        "Acme",
        "Black",
    )
    assert (first["brand_key"], first["color_key"]) == ("acme", "black")
    chunk_indexes = sorted(
        p.payload["chunk_idx"] for p in fake.points.values() if p.payload["doc_id"] == "P0"
    )
    assert chunk_indexes == [0, 1]
    plain = next(p.payload for p in fake.points.values() if p.payload["doc_id"] == "P1")
    assert "brand" not in plain
    assert "brand_key" not in plain


async def test_the_version_is_the_dataset_revision_of_the_manifest(built: Path) -> None:
    fake = FakeQdrant()
    await build_index(built, settings(), fake, HashEmbedder(dense_dim=DIM, max_tokens=64))  # type: ignore[arg-type]
    assert {p.payload["version"] for p in fake.points.values()} == {REVISION}


async def test_a_product_that_breaks_the_api_rules_stops_the_build_and_is_named(
    tmp_path: Path,
) -> None:
    # two models must index exactly the same products, so a product one model cannot take is an
    # error, never a quiet skip that would make the comparison unfair
    dataset(tmp_path / "set", [*products(), BREAKS_THE_API_RULES])
    with pytest.raises(BuildStopped, match=r"P2.*title") as stopped:
        await build_index(
            tmp_path / "set",
            settings(),
            FakeQdrant(),  # type: ignore[arg-type]
            HashEmbedder(dense_dim=DIM, max_tokens=64),
        )
    assert stopped.value.product_id == "P2"


async def test_a_product_that_cannot_be_composed_stops_the_build_and_is_named(
    tmp_path: Path,
) -> None:
    dataset(tmp_path / "set", [*products(), HEADER_TOO_LONG])
    with pytest.raises(BuildStopped, match=r"P3.*budget") as stopped:
        await build_index(
            tmp_path / "set",
            settings(),
            FakeQdrant(),  # type: ignore[arg-type]
            HashEmbedder(dense_dim=DIM, max_tokens=64),
        )
    assert stopped.value.product_id == "P3"


async def test_after_a_stop_the_products_before_it_were_written_and_the_ones_after_were_not(
    tmp_path: Path,
) -> None:
    # sorted by id the file reads P0, P1, P2 (broken), P4; with one product per batch the first two
    # are already live when P2 stops the build, P4 is never read, the caller must discard the
    # half built collection
    dataset(tmp_path / "set", [*products(), BREAKS_THE_API_RULES])
    fake = FakeQdrant()
    with pytest.raises(BuildStopped):
        await build_index(
            tmp_path / "set",
            settings(),
            fake,  # type: ignore[arg-type]
            HashEmbedder(dense_dim=DIM, max_tokens=64),
            batch_chunks=1,
        )
    assert {p.payload["doc_id"] for p in fake.points.values()} == {"P0", "P1"}


async def test_a_unicode_line_separator_inside_a_product_does_not_split_a_record(
    built: Path,
) -> None:
    fake = FakeQdrant()
    await build_index(built, settings(), fake, HashEmbedder(dense_dim=DIM, max_tokens=64))  # type: ignore[arg-type]
    titles = {p.payload["title"] for p in fake.points.values() if p.payload["doc_id"] == "P4"}
    # the stored title is the normalised one: the separator is a space, the record stayed whole
    assert titles == {"wire less pad"}


async def test_chunks_of_one_product_are_embedded_and_published_together_in_small_batches(
    tmp_path: Path,
) -> None:
    items = [
        Product(f"Q{n}", "us", f"item {n}", words(30, "d"), words(20, "a"), None, None)
        for n in range(5)
    ]
    dataset(tmp_path / "set", items)
    embedder = HashEmbedder(dense_dim=DIM, max_tokens=64)
    fake = FakeQdrant()
    # each product has two chunks, a batch is closed once it holds at least 3 chunks, but only at a
    # product boundary, so the batches hold 4, 4 and 2 chunks
    report = await build_index(tmp_path / "set", settings(), fake, embedder, batch_chunks=3)  # type: ignore[arg-type]
    assert report.chunks == 10
    assert [len(call) for call in embedder.calls] == [4, 4, 2]


async def test_a_batch_is_staged_before_it_is_published_and_never_the_other_way_round(
    built: Path,
) -> None:
    fake = FakeQdrant()
    await build_index(
        built, settings(), fake, HashEmbedder(dense_dim=DIM, max_tokens=64), batch_chunks=1
    )  # type: ignore[arg-type]
    kinds = [kind for kind, _ in fake.calls]
    # one stage and one publish per batch, in that order
    assert kinds == ["upsert", "publish"] * (len(kinds) // 2)
    assert len(kinds) > 2


async def test_the_collection_is_created_for_the_model_of_the_settings(built: Path) -> None:
    fake = FakeQdrant()
    await build_index(built, settings(), fake, HashEmbedder(dense_dim=DIM, max_tokens=64))  # type: ignore[arg-type]
    assert fake.created == ["built"]


async def test_a_text_the_model_refuses_stops_the_build_and_names_its_product(
    tmp_path: Path,
) -> None:
    dataset(tmp_path / "set", [*products()[:2], REFUSED_BY_THE_MODEL])
    fake = FakeQdrant()
    with pytest.raises(BuildStopped, match=r"P5.*refused") as stopped:
        await build_index(
            tmp_path / "set",
            settings(),
            fake,  # type: ignore[arg-type]
            PoisonedEmbedder(dense_dim=DIM, max_tokens=64),
            batch_chunks=100,
        )
    assert stopped.value.product_id == "P5"
    # the batch that held the refused text is not written at all
    assert fake.points == {}


async def test_progress_is_reported_after_every_batch_with_running_totals(built: Path) -> None:
    seen: list[tuple[int, int]] = []
    await build_index(
        built,
        settings(),
        FakeQdrant(),
        HashEmbedder(dense_dim=DIM, max_tokens=64),  # type: ignore[arg-type]
        batch_chunks=1,
        progress=lambda products, chunks: seen.append((products, chunks)),
    )
    assert seen == sorted(seen)
    assert seen[-1] == (3, 4)


async def test_an_empty_dataset_builds_an_empty_collection(tmp_path: Path) -> None:
    dataset(tmp_path / "set", [])
    fake = FakeQdrant()
    report = await build_index(
        tmp_path / "set", settings(), fake, HashEmbedder(dense_dim=DIM, max_tokens=64)
    )  # type: ignore[arg-type]
    assert (report.products, report.chunks) == (0, 0)
    assert fake.created == ["built"]
    assert fake.calls == []
