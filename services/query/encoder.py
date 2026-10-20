"""Turn the text of a query into the two vectors the collection is searched with.

Both vectors are made from the same cleaned text: Unicode NFC, one space between words, and case
folded, so "NIKE", "Nike" and "nike" give the same vectors. The two encoders are separate calls on
purpose: each retriever is "encode, then search", and a failure of either step takes only that
retriever out of the answer.
"""

from dataclasses import dataclass
from typing import Protocol

from libs.embedding.collection import normalise
from libs.embedding.model import OnnxEmbedder


@dataclass(frozen=True)
class SparseQuery:
    """The bm25 terms of a query: term numbers and their weights."""

    indices: list[int]
    values: list[float]


class QueryEncoder(Protocol):
    """What the two flows need, the real model and the test stubs both fit."""

    async def encode_dense(self, query: str) -> list[float]: ...

    async def encode_sparse(self, query: str) -> SparseQuery: ...


class OnnxQueryEncoder:
    def __init__(self, embedder: OnnxEmbedder, timeout_s: float, dense_instruction: bool) -> None:
        self._embedder = embedder
        self._timeout_s = timeout_s
        # whether the dense model gets its query prefix, a setting until the ladder has measured it
        self._dense_instruction = dense_instruction

    async def encode_dense(self, query: str) -> list[float]:
        text = normalise(query).casefold()
        return await self._embedder.embed_query_dense(
            text, self._timeout_s, self._dense_instruction
        )

    async def encode_sparse(self, query: str) -> SparseQuery:
        text = normalise(query).casefold()
        indices, values = await self._embedder.embed_query_sparse(text)
        return SparseQuery(indices, values)
