"""The embedder with the real model: what the machinery cannot show, shapes and meaning.

A handful of tests on purpose: the pipeline tests use the hash stub, this file proves the real model
and its tokenizer behave the way the rest of the code assumes. Needs the model files (cached after
the first download), no other service.
"""

import math
from collections.abc import Iterator
from itertools import pairwise

import pytest

from libs.schemas import UpsertRequest
from services.embed.compose import compose
from services.embed.embedder import ModelMismatch, OnnxEmbedder, build_onnx_embedder
from services.embed.settings import EmbedSettings

TIMEOUT = 120.0


@pytest.fixture(scope="module")
def embedder() -> Iterator[OnnxEmbedder]:
    real = build_onnx_embedder(EmbedSettings())
    try:
        yield real
    finally:
        real.close()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


async def test_vectors_have_the_collection_size_and_unit_length(embedder: OnnxEmbedder) -> None:
    (result,) = await embedder.embed(["wireless mouse"], TIMEOUT)
    assert len(result.dense) == EmbedSettings().dense_dim
    assert math.sqrt(sum(x * x for x in result.dense)) == pytest.approx(1.0, abs=1e-4)


async def test_the_same_text_twice_gives_the_same_vector(embedder: OnnxEmbedder) -> None:
    first, second = await embedder.embed(["wireless mouse", "wireless mouse"], TIMEOUT)
    assert first == second


async def test_a_batch_matches_the_same_texts_embedded_one_by_one(embedder: OnnxEmbedder) -> None:
    # the embedder reorders texts by length, so this also proves every vector went back to its text
    texts = [
        "mouse",
        "a very long description of an ergonomic wireless mouse with silent clicks " * 4,
        "usb cable",
        "ergonomic wireless mouse with usb receiver",
    ]
    together = await embedder.embed(texts, TIMEOUT)
    for text, from_batch in zip(texts, together, strict=True):
        (alone,) = await embedder.embed([text], TIMEOUT)
        assert from_batch.dense == pytest.approx(alone.dense, abs=1e-5)


async def test_similar_products_are_closer_than_unrelated_ones(embedder: OnnxEmbedder) -> None:
    mouse, cordless, lamp = await embedder.embed(
        ["wireless mouse", "cordless computer mouse", "dimmable desk lamp"], TIMEOUT
    )
    assert cosine(mouse.dense, cordless.dense) > cosine(mouse.dense, lamp.dense)


async def test_a_text_far_beyond_the_model_limit_is_embedded_without_an_error(
    embedder: OnnxEmbedder,
) -> None:
    long_text = "wireless mouse " * 2000
    (result,) = await embedder.embed([long_text], TIMEOUT)
    assert len(result.dense) == 384
    # the model cut the text silently, the tokenizer the composer uses must still see all of it
    assert len(embedder.tokenizer.spans(long_text)) == 4000
    assert embedder.limits.max_tokens == 512


def test_token_spans_cut_the_original_text_without_special_tokens(embedder: OnnxEmbedder) -> None:
    text = "Wireless  Mouse, café"
    spans = embedder.tokenizer.spans(text)
    pieces = [text[start:end] for start, end in spans]
    assert pieces[:2] == ["Wireless", "Mouse"]
    assert all(piece for piece in pieces)
    assert spans == sorted(spans)  # in reading order
    assert all(first[1] <= second[0] for first, second in pairwise(spans))


async def test_stop_words_alone_give_no_sparse_terms_and_repeats_saturate(
    embedder: OnnxEmbedder,
) -> None:
    stops, once, many, lots = await embedder.embed(
        ["the and of to a", "mouse", "mouse " * 200, "mouse " * 2000], TIMEOUT
    )
    assert (stops.sparse_indices, stops.sparse_values) == ([], [])
    assert many.sparse_values[0] > once.sparse_values[0]
    # the weight stops growing, so a long repetitive text cannot drown a short one
    assert many.sparse_values[0] == pytest.approx(lots.sparse_values[0], rel=0.05)


async def test_a_long_product_is_split_into_chunks_that_all_fit_the_real_model(
    embedder: OnnxEmbedder,
) -> None:
    product = UpsertRequest.model_validate(
        {
            "corpus": "esci",
            "doc_id": "P1",
            "version": 1,
            "title": "Ergonomic wireless mouse",
            "locale": "us",
            "bullets": [f"Feature {n} of the mouse is silent and light." for n in range(20)],
            "description": " ".join(
                f"Sentence number {n} describes the mouse." for n in range(120)
            )[:5000],
        }
    )
    chunks = compose(product, embedder.tokenizer, embedder.limits)
    assert len(chunks) > 1  # the split path is real for the long tail
    for chunk in chunks:
        assert len(embedder.tokenizer.spans(chunk.text)) <= embedder.limits.budget
    results = await embedder.embed([chunk.text for chunk in chunks], TIMEOUT)
    assert len(results) == len(chunks)


async def test_verify_passes_for_the_right_size_and_stops_a_wrong_one(
    embedder: OnnxEmbedder,
) -> None:
    await embedder.verify()
    wrong = build_onnx_embedder(EmbedSettings(dense_dim=768))
    try:
        with pytest.raises(ModelMismatch):
            await wrong.verify()
    finally:
        wrong.close()
