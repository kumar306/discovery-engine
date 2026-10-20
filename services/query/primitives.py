"""Pure building blocks shared by the two flows: the hit type and reciprocal rank fusion."""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    """One chunk returned by Qdrant, from either retriever."""

    point_id: str
    doc_id: str
    chunk_idx: int
    # the retriever's own score: cosine for dense, a bm25 style number for sparse, not comparable
    score: float
    text: str
    title: str
    brand: str | None
    color: str | None


@dataclass(frozen=True)
class Fused[K: Hashable]:
    key: K
    score: float
    # retriever name -> 1-based rank, a retriever that did not return the key is absent
    ranks: dict[str, int]


def rrf[K: Hashable](rankings: Mapping[str, Sequence[K]], k: int = 60) -> list[Fused[K]]:
    """Reciprocal rank fusion: score(key) = sum over rankings of 1 / (k + rank).

    The rankings are ordered lists of keys, best first, named by their retriever. A key that
    several retrievers returned is boosted, a key that one returned still counts. The output is
    ordered best first, ties by the text of the key, so the same input always gives the same order.
    """
    scores: dict[K, float] = {}
    ranks: dict[K, dict[str, int]] = {}
    # names in sorted order, so the floating point sum for a key never depends on the order the
    # caller built the mapping in (a sum of three or more floats can differ in the last digit)
    for name in sorted(rankings):
        counted: set[K] = set()
        for position, key in enumerate(rankings[name], start=1):
            # a key repeated inside one ranking counts once, at its best rank
            if key in counted:
                continue
            counted.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position)
            ranks.setdefault(key, {})[name] = position
    fused = [Fused(key, score, ranks[key]) for key, score in scores.items()]
    fused.sort(key=lambda item: (-item.score, str(item.key)))
    return fused
