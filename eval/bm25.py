"""A reference BM25 written from the formula, to check the sparse retriever against.

the real sparse search is built by fastembed (at index time) and qdrant (at query time) and we
cannot see inside them, so the same ranking is computed here a second way, over the same chunk
texts, and the two rankings are compared in the bm25 oracle test
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Scored:
    """One chunk and its score for one query."""

    doc_id: str
    score: float


def tokenise(bm25_model: Any, text: str) -> list[str]:
    """The tokens of a text exactly as fastembed's BM25 makes them, repeats kept, in order.

    bm25_model is SparseTextEmbedding("Qdrant/bm25").model, the object the real pipeline uses
    steps: every character that is not a letter, a digit or a space becomes a space, the text
    is split into words, stopwords and words over 40 characters are dropped, the rest is
    lowercased and stemmed (Snowball, English)
    example: "Running Shoes for Men!" gives ["run", "shoe", "men"]
    _stem is a private method of fastembed and its version is pinned, the test in
    tests/integration/test_bm25_tokens.py compares our tokens with the term ids fastembed makes
    itself, so a library change that breaks this fails a test
    """
    # imported here, not at the top, so that using the corpus alone does not load fastembed
    from fastembed.common.utils import remove_non_alphanumeric

    cleaned = remove_non_alphanumeric(text)
    tokens: list[str] = bm25_model._stem(bm25_model.tokenizer.tokenize(cleaned))
    return tokens


class Bm25Corpus:
    """Chunks as lists of tokens, with the counts BM25 needs."""

    def __init__(self, tokens_by_doc: Mapping[str, Sequence[str]]) -> None:
        self._term_counts = {doc_id: Counter(tokens) for doc_id, tokens in tokens_by_doc.items()}
        self._length_of = {doc_id: len(tokens) for doc_id, tokens in tokens_by_doc.items()}
        # in how many chunks each token occurs, a chunk counts once however often it repeats
        self._docs_with: Counter[str] = Counter()
        for counts in self._term_counts.values():
            self._docs_with.update(counts.keys())
        lengths = list(self._length_of.values())
        # the true mean, the real pipeline assumes a fixed 256 instead
        self.average_length = sum(lengths) / len(lengths) if lengths else 0.0

    def idf(self, term: str) -> float:
        # ln(1 + (N - n + 0.5) / (n + 0.5)), N chunks in all, n chunks that contain the term
        # a rarer term gives a bigger number, example N = 2, n = 1 gives ln 2 = 0.693
        documents = len(self._term_counts)
        containing = self._docs_with.get(term, 0)
        return math.log(1.0 + (documents - containing + 0.5) / (containing + 0.5))

    def rank(
        self, query_tokens: Sequence[str], *, k1: float, b: float, average_length: float, limit: int
    ) -> list[Scored]:
        """The best `limit` chunks, best first, ties by chunk id.

        score = sum over the distinct query words of
                idf(word) * n (k1 + 1) / (n + k1 (1 - b + b len / average_length))
        n is how often the word occurs in the chunk, len is the chunk's length in tokens
        """
        if average_length <= 0:
            raise ValueError("the average length is a divisor and must be above zero")
        # distinct words, as in production: the query side gives every word the weight 1
        # sorted, so the floating point sum below adds the words in the same order in every
        # process, a set of strings is ordered by a hash that is different in every run and the
        # last digit of a sum depends on the order of its terms
        query_terms = sorted(set(query_tokens))
        # looked up once per word, not once per chunk
        idf_of_term = {term: self.idf(term) for term in query_terms}
        scored: list[Scored] = []
        for doc_id, counts in self._term_counts.items():
            score = 0.0
            for term in query_terms:
                occurrences = counts.get(term, 0)
                if occurrences == 0:
                    continue
                length_part = 1.0 - b + b * self._length_of[doc_id] / average_length
                weight = occurrences * (k1 + 1.0) / (occurrences + k1 * length_part)
                score += idf_of_term[term] * weight
            # a chunk that shares no word with the query is not a result, as in Qdrant
            if score > 0.0:
                scored.append(Scored(doc_id, score))
        scored.sort(key=lambda item: (-item.score, item.doc_id))
        return scored[:limit]
