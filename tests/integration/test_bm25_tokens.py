"""tokenise against the real fastembed BM25 object: the tokens, the term ids, the stored weight.

Needs the BM25 files of fastembed (cached after the first download), no other service. These are
the tests that pin the oracle to production: if fastembed changes how it cleans, stems or weighs,
one of them fails.
"""

import pytest
from fastembed import SparseTextEmbedding

from eval.bm25 import tokenise


@pytest.fixture(scope="module")
def sparse() -> SparseTextEmbedding:
    return SparseTextEmbedding("Qdrant/bm25", avg_len=256.0)


def test_tokens_are_cleaned_stemmed_and_stopwords_are_dropped(sparse: SparseTextEmbedding) -> None:
    assert tokenise(sparse.model, "Running Shoes for Men!") == ["run", "shoe", "men"]


def test_a_chunk_text_with_its_header_line_is_tokenised_in_order_with_repeats(
    sparse: SparseTextEmbedding,
) -> None:
    text = "Acme wireless mouse | Acme | black\nErgonomic design with silent clicks"
    assert tokenise(sparse.model, text) == [
        "acm",
        "wireless",
        "mous",
        "acm",
        "black",
        "ergonom",
        "design",
        "silent",
        "click",
    ]


def test_repeats_are_kept_because_the_length_of_a_chunk_counts_them(
    sparse: SparseTextEmbedding,
) -> None:
    assert tokenise(sparse.model, "red red running shoes") == ["red", "red", "run", "shoe"]


def test_a_word_over_forty_characters_is_dropped_and_empty_text_gives_nothing(
    sparse: SparseTextEmbedding,
) -> None:
    assert tokenise(sparse.model, "ok " + "x" * 41) == ["ok"]
    assert tokenise(sparse.model, "") == []
    assert tokenise(sparse.model, "!!! ... ???") == []


@pytest.mark.parametrize(
    "text",
    [
        "red running shoes for men",
        "Acme wireless mouse | Acme | black",
        "USB-C cable, 2 m, braided (nylon)",
        "Straße café naïve",
    ],
)
def test_our_tokens_make_the_same_term_ids_as_fastembeds_own_query_path(
    sparse: SparseTextEmbedding, text: str
) -> None:
    ours = sorted({sparse.model.compute_token_id(token) for token in tokenise(sparse.model, text)})
    theirs = sorted(next(iter(sparse.query_embed(text))).indices.tolist())
    assert ours == theirs


def test_the_stored_weight_of_a_word_is_the_bm25_formula_with_the_fixed_average_length(
    sparse: SparseTextEmbedding,
) -> None:
    # the document 'red red running shoe' has 4 tokens and 'red' occurs twice (n = 2)
    # weight = n (k + 1) / (n + k (1 - b + b len / avg_len)) with k 1.2, b 0.75, avg_len 256
    stored = next(iter(sparse.embed(["red red running shoe"])))
    weight_of = dict(zip(stored.indices.tolist(), stored.values.tolist(), strict=True))
    expected = 2 * 2.2 / (2 + 1.2 * (1 - 0.75 + 0.75 * 4 / 256))
    assert weight_of[sparse.model.compute_token_id("red")] == pytest.approx(expected, abs=1e-4)
    assert expected == pytest.approx(1.90142, abs=1e-4)
