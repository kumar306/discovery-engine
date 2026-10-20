"""The query encoder folds the text for both vectors, and the query side of the embedder."""

import time
from typing import Any

import pytest

from libs.embedding.model import EmbedTransient, Limits, ModelMismatch, OnnxEmbedder
from services.query.encoder import OnnxQueryEncoder, SparseQuery


class RecordingEmbedder:
    """Stands in for the embedder and keeps what the encoder asked it."""

    def __init__(self) -> None:
        self.dense_calls: list[tuple[str, float, bool]] = []
        self.sparse_calls: list[str] = []

    async def embed_query_dense(
        self, text: str, timeout_s: float, instruction: bool
    ) -> list[float]:
        self.dense_calls.append((text, timeout_s, instruction))
        return [1.0, 2.0]

    async def embed_query_sparse(self, text: str) -> tuple[list[int], list[float]]:
        self.sparse_calls.append(text)
        return [3, 7], [0.5, 1.5]


def encoder(embedder: RecordingEmbedder, instruction: bool = False) -> OnnxQueryEncoder:
    return OnnxQueryEncoder(embedder, timeout_s=2.5, dense_instruction=instruction)  # type: ignore[arg-type]


async def test_both_vectors_are_made_from_the_same_cleaned_and_folded_text() -> None:
    embedder = RecordingEmbedder()
    # "NIKE", extra spaces and an accented e written as e plus a combining accent
    messy = "  NIKE   Café  Shoes "
    await encoder(embedder).encode_dense(messy)
    await encoder(embedder).encode_sparse(messy)
    assert embedder.dense_calls[0][0] == "nike café shoes"
    assert embedder.sparse_calls == ["nike café shoes"]


async def test_nike_and_NIKE_reach_the_model_as_the_same_text() -> None:
    embedder = RecordingEmbedder()
    await encoder(embedder).encode_dense("NIKE")
    await encoder(embedder).encode_dense("nike")
    assert embedder.dense_calls[0][0] == embedder.dense_calls[1][0]


async def test_the_timeout_and_the_instruction_setting_reach_the_embedder() -> None:
    embedder = RecordingEmbedder()
    await encoder(embedder, instruction=True).encode_dense("shoes")
    assert embedder.dense_calls == [("shoes", 2.5, True)]


async def test_the_sparse_terms_come_back_as_indices_and_values() -> None:
    assert await encoder(RecordingEmbedder()).encode_sparse("shoes") == SparseQuery(
        [3, 7], [0.5, 1.5]
    )


# the query side of the embedder itself, with fake models


class FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


class FakeDense:
    def __init__(self, size: int = 4, delay_s: float = 0.0) -> None:
        self.size = size
        self.delay_s = delay_s
        self.document_calls: list[list[str]] = []
        self.query_calls: list[Any] = []

    def embed(self, documents: list[str], **kwargs: Any) -> Any:
        time.sleep(self.delay_s)
        self.document_calls.append(documents)
        return iter([FakeVector([1.0] * self.size) for _ in documents])

    def query_embed(self, query: Any, **kwargs: Any) -> Any:
        self.query_calls.append(query)
        return iter([FakeVector([2.0] * self.size)])


class FakeSparse:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def query_embed(self, query: str, **kwargs: Any) -> Any:
        self.queries.append(query)

        class Item:
            indices = FakeVector([1, 5])
            values = FakeVector([0.5, 1.0])

        return iter([Item()])

    def embed(self, documents: list[str], **kwargs: Any) -> Any:
        raise AssertionError("a query must never take the document path of the sparse model")


def embedder_with(dense: FakeDense, sparse: FakeSparse) -> OnnxEmbedder:
    return OnnxEmbedder(
        dense,
        sparse,
        tokenizer=None,  # type: ignore[arg-type]
        limits=Limits(max_tokens=64),
        dense_dim=4,
        sub_batch=4,
    )


async def test_the_plain_dense_path_is_the_document_path_and_the_instruction_path_is_not() -> None:
    dense, embedder = FakeDense(), None
    embedder = embedder_with(dense, FakeSparse())
    try:
        assert await embedder.embed_query_dense("shoes", 5.0, False) == [1.0] * 4
        assert await embedder.embed_query_dense("shoes", 5.0, True) == [2.0] * 4
    finally:
        embedder.close()
    assert dense.document_calls == [["shoes"]]
    assert dense.query_calls == [["shoes"]]


async def test_a_dense_vector_of_the_wrong_size_is_a_model_mismatch() -> None:
    embedder = embedder_with(FakeDense(size=3), FakeSparse())
    try:
        with pytest.raises(ModelMismatch, match="expected 4"):
            await embedder.embed_query_dense("shoes", 5.0, False)
    finally:
        embedder.close()


async def test_a_slow_dense_model_is_a_transient_error_after_the_timeout() -> None:
    embedder = embedder_with(FakeDense(delay_s=0.4), FakeSparse())
    try:
        with pytest.raises(EmbedTransient, match="took more than"):
            await embedder.embed_query_dense("shoes", 0.05, False)
    finally:
        embedder.close()


async def test_the_sparse_query_uses_the_query_path_of_the_bm25_model() -> None:
    sparse = FakeSparse()
    embedder = embedder_with(FakeDense(), sparse)
    try:
        assert await embedder.embed_query_sparse("shoes") == ([1, 5], [0.5, 1.0])
    finally:
        embedder.close()
    assert sparse.queries == ["shoes"]
