"""What the payload of a point holds, and the filter key that brand and color filters compare."""

import uuid
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from libs.embedding.collection import filter_key
from services.embed.qdrant import ChunkIndex, ChunkPayload, IndexedChunk


class RecordingClient:
    """Stands in for the Qdrant client and keeps the points it was asked to write."""

    def __init__(self) -> None:
        self.points: list[Any] = []

    async def upsert(self, collection: str, points: list[Any], **kwargs: Any) -> None:
        self.points.extend(points)


async def written_payload(**fields: Any) -> dict[str, Any]:
    payload = ChunkPayload(
        doc_id="P1",
        version=1,
        chunk_idx=0,
        content_hash="a" * 64,
        model_version="m",
        locale="us",
        text="Red Shoe | Nike | Red\nbody",
        title="Red Shoe",
        **fields,
    )
    chunk = IndexedChunk(uuid.UUID(int=1), [0.0], [1], [1.0], payload)
    client = RecordingClient()
    await ChunkIndex(client, "chunks").stage([chunk])  # type: ignore[arg-type]
    written: dict[str, Any] = client.points[0].payload
    return written


# the filter key


@pytest.mark.parametrize(
    ("raw", "key"),
    [
        ("Nike", "nike"),
        ("  NIKE   Air ", "nike air"),
        ("Café", "café"),
        ("Café", "café"),
        ("Straße", "strasse"),
        ("", ""),
    ],
)
def test_filter_key_makes_one_spelling_of_a_brand_or_color(raw: str, key: str) -> None:
    assert filter_key(raw) == key


@given(st.text())
def test_filter_key_is_stable(text: str) -> None:
    assert filter_key(filter_key(text)) == filter_key(text)


# the payload


async def test_the_payload_carries_the_title_and_the_filter_keys() -> None:
    payload = await written_payload(brand="Nike", color="Red")
    assert payload["title"] == "Red Shoe"
    assert (payload["brand"], payload["brand_key"]) == ("Nike", "nike")
    assert (payload["color"], payload["color_key"]) == ("Red", "red")


async def test_the_original_is_kept_for_display_and_only_the_key_is_folded() -> None:
    payload = await written_payload(brand="  NIKE   Air ", color="Dark  Blue")
    assert payload["brand"] == "  NIKE   Air "
    assert (payload["brand_key"], payload["color_key"]) == ("nike air", "dark blue")


async def test_a_missing_brand_or_color_leaves_out_the_field_and_its_key() -> None:
    payload = await written_payload()
    assert payload["title"] == "Red Shoe"
    assert not {"brand", "brand_key", "color", "color_key"} & set(payload)
