"""The reference BM25: the formula worked by hand, and the properties the comparison relies on."""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from eval.bm25 import Bm25Corpus, Scored

# two chunks and a query, with the scores worked out by hand in docs/learning/g10_commit1_review.md
CHUNKS = {"d1": ["red", "shoe"], "d2": ["red", "red", "running", "shoe"]}


def rank_chunks(query: list[str], limit: int = 10) -> list[Scored]:
    corpus = Bm25Corpus(CHUNKS)
    return corpus.rank(query, k1=1.2, b=0.75, average_length=3.0, limit=limit)


def test_the_hand_worked_example_gives_the_scores_and_the_order_worked_out_by_hand() -> None:
    result = rank_chunks(["red", "running"])
    assert [item.doc_id for item in result] == ["d2", "d1"]
    assert result[0].score == pytest.approx(0.83918, abs=1e-4)
    assert result[1].score == pytest.approx(0.21111, abs=1e-4)


def test_idf_is_larger_for_a_rarer_word() -> None:
    corpus = Bm25Corpus(CHUNKS)
    assert corpus.idf("red") == pytest.approx(math.log(1.2))
    assert corpus.idf("running") == pytest.approx(math.log(2.0))
    # a word in no chunk: ln(1 + (2 - 0 + 0.5) / (0 + 0.5)) = ln 6
    assert corpus.idf("nothing") == pytest.approx(math.log(6.0))
    assert corpus.idf("running") > corpus.idf("red")


def test_the_average_length_is_the_true_mean_and_zero_for_nothing() -> None:
    assert Bm25Corpus(CHUNKS).average_length == pytest.approx(3.0)
    assert Bm25Corpus({}).average_length == 0.0


def test_a_chunk_with_no_query_word_is_not_a_result() -> None:
    assert rank_chunks(["zzz"]) == []
    assert rank_chunks([]) == []
    assert [item.doc_id for item in rank_chunks(["running"])] == ["d2"]


def test_limit_cuts_the_list_and_equal_scores_are_ordered_by_chunk_id() -> None:
    corpus = Bm25Corpus({"c": ["red"], "a": ["red"], "b": ["red"]})
    everything = corpus.rank(["red"], k1=1.2, b=0.75, average_length=1.0, limit=10)
    assert [item.doc_id for item in everything] == ["a", "b", "c"]
    two = corpus.rank(["red"], k1=1.2, b=0.75, average_length=1.0, limit=2)
    assert [item.doc_id for item in two] == ["a", "b"]


def test_a_repeated_query_word_counts_once() -> None:
    assert rank_chunks(["red", "red", "running"]) == rank_chunks(["red", "running"])


def test_the_length_of_a_chunk_matters_when_b_is_above_zero() -> None:
    corpus = Bm25Corpus({"short": ["red", "x"], "long": ["red", "x", "y", "z", "w", "v"]})
    with_length = corpus.rank(["red"], k1=1.2, b=0.75, average_length=3.0, limit=10)
    assert [item.doc_id for item in with_length] == ["short", "long"]
    assert with_length[0].score > with_length[1].score
    # with b = 0 the length is ignored, the scores are equal and the order falls back to the id
    without_length = corpus.rank(["red"], k1=1.2, b=0.0, average_length=3.0, limit=10)
    assert without_length[0].score == without_length[1].score
    assert [item.doc_id for item in without_length] == ["long", "short"]


def test_an_average_length_that_is_not_above_zero_is_refused() -> None:
    corpus = Bm25Corpus(CHUNKS)
    with pytest.raises(ValueError, match="average length"):
        corpus.rank(["red"], k1=1.2, b=0.75, average_length=0.0, limit=10)


# the order of things must not change a score, down to the last digit


def test_the_order_of_the_query_words_does_not_change_the_scores() -> None:
    corpus = Bm25Corpus(
        {"d1": ["red", "shoe", "run"], "d2": ["red", "red", "blue", "run"], "d3": ["blue", "shoe"]}
    )
    words = ["red", "shoe", "run", "blue", "green"]
    forward = corpus.rank(words, k1=1.2, b=0.75, average_length=3.0, limit=10)
    backward = corpus.rank(list(reversed(words)), k1=1.2, b=0.75, average_length=3.0, limit=10)
    assert forward == backward


SCORES_IN_A_NEW_PROCESS = """
from eval.bm25 import Bm25Corpus
corpus = Bm25Corpus({
    "d1": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split(),
    "d2": "alpha alpha beta gamma delta delta nu xi omicron pi rho sigma".split(),
    "d3": "tau upsilon phi chi psi omega alpha beta".split(),
})
query = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron".split()
for item in corpus.rank(query, k1=1.2, b=0.75, average_length=256.0, limit=10):
    print(item.doc_id, repr(item.score))
"""


def test_the_scores_are_identical_in_processes_with_different_string_hashes() -> None:
    # python orders a set of strings by a hash that is different in every process, so a sum over
    # the query words in set order could differ in its last digit between two runs
    outputs = set()
    for hash_seed in ("0", "1", "2", "3"):
        environment = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(Path.cwd())}
        done = subprocess.run(  # noqa: S603, the command is our own fixed snippet
            [sys.executable, "-c", SCORES_IN_A_NEW_PROCESS],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
            timeout=60,
        )
        outputs.add(done.stdout)
    assert len(outputs) == 1


tokens = st.lists(st.sampled_from(list("abcde")), max_size=6)
corpora = st.dictionaries(st.sampled_from(["c1", "c2", "c3", "c4", "c5"]), tokens, min_size=1)


@given(corpora, st.lists(st.sampled_from(list("abcde")), max_size=4), st.randoms())
def test_the_order_the_chunks_were_given_in_changes_nothing(
    corpus: dict[str, list[str]], query: list[str], random: object
) -> None:
    items = list(corpus.items())
    random.shuffle(items)  # type: ignore[attr-defined]
    rank = {"k1": 1.2, "b": 0.75, "average_length": 3.0, "limit": 10}
    assert Bm25Corpus(corpus).rank(query, **rank) == Bm25Corpus(dict(items)).rank(query, **rank)  # type: ignore[arg-type]
