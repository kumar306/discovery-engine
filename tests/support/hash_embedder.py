"""A stand in for the model in the pipeline tests: same shape, no tensors, always the same answer.

The worker's crash safety, ordering and versioning have nothing to do with what the vectors mean,
so most tests use this and only a handful load the real model (see test_embedder_model.py).
"""

import hashlib
import re
from collections.abc import Sequence

from libs.embedding.model import Embedding, Limits

_WORD = re.compile(r"\S+")


class WhitespaceTokenizer:
    def spans(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in _WORD.finditer(text)]


def _numbers(text: str, count: int) -> list[float]:
    # the same text always gives the same numbers, and different texts give different ones
    values: list[float] = []
    block = 0
    while len(values) < count:
        digest = hashlib.sha256(f"{block}:{text}".encode()).digest()
        values.extend(byte / 255 + 0.01 for byte in digest)
        block += 1
    return values[:count]


class HashEmbedder:
    def __init__(self, dense_dim: int = 4, max_tokens: int = 512) -> None:
        self.tokenizer = WhitespaceTokenizer()
        self.limits = Limits(max_tokens=max_tokens)
        self._dense_dim = dense_dim
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str], timeout_s: float) -> list[Embedding]:
        self.calls.append(list(texts))
        return [self._one(text) for text in texts]

    async def verify(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _one(self, text: str) -> Embedding:
        words = sorted(set(text.lower().split()))
        # one sparse term per distinct word, its id from the word itself, so equal words match
        indices = [int(hashlib.sha256(word.encode()).hexdigest()[:6], 16) for word in words]
        return Embedding(_numbers(text, self._dense_dim), indices, [1.0] * len(indices))
