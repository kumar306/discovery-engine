"""Reciprocal rank fusion and the shared hit type, pure code with no stack."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from services.query.primitives import rrf


def test_two_retrievers_agree_and_disagree() -> None:
    result = rrf({"dense": ["a", "b", "c"], "sparse": ["b", "a", "d"]}, k=60)
    assert [item.key for item in result] == ["a", "b", "c", "d"]
    assert result[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert result[0].score == result[1].score  # a tie, broken by the key
    assert result[0].ranks == {"dense": 1, "sparse": 2}
    assert result[3].ranks == {"sparse": 3}


def test_a_key_both_retrievers_returned_beats_a_key_only_one_ranked_first() -> None:
    result = rrf({"dense": ["x", "both"], "sparse": ["both", "y"]}, k=60)
    assert result[0].key == "both"
    assert result[0].ranks == {"dense": 2, "sparse": 1}


def test_a_key_only_one_retriever_returned_still_counts() -> None:
    result = rrf({"dense": ["a", "b"], "sparse": []})
    assert [(item.key, item.ranks) for item in result] == [("a", {"dense": 1}), ("b", {"dense": 2})]


def test_nothing_in_gives_nothing_out() -> None:
    assert rrf({}) == []
    assert rrf({"dense": [], "sparse": []}) == []


def test_the_constant_k_sets_the_weight_of_first_place() -> None:
    assert rrf({"dense": ["a"]}, k=1)[0].score == pytest.approx(1 / 2)
    assert rrf({"dense": ["a"]}, k=60)[0].score == pytest.approx(1 / 61)


def test_a_key_repeated_inside_one_ranking_counts_once_at_its_best_rank() -> None:
    result = rrf({"dense": ["a", "b", "a"]}, k=60)
    assert [item.key for item in result] == ["a", "b"]
    assert result[0].ranks == {"dense": 1}
    assert result[0].score == pytest.approx(1 / 61)


def test_two_chunks_of_one_product_stay_two_results_when_chunks_are_the_keys() -> None:
    # Flow C ranks chunk ids, the product id plays no part in fusion
    result = rrf({"dense": ["p1-c0", "p1-c1"], "sparse": ["p1-c1"]})
    assert [item.key for item in result] == ["p1-c1", "p1-c0"]


ranking = st.lists(st.sampled_from(list("abcdefgh")), unique=True, max_size=8)


@given(st.dictionaries(st.sampled_from(["dense", "sparse", "other"]), ranking))
def test_the_order_the_rankings_are_given_in_changes_nothing(
    rankings: dict[str, list[str]],
) -> None:
    reversed_order = dict(reversed(list(rankings.items())))
    assert rrf(rankings) == rrf(reversed_order)


@given(st.dictionaries(st.sampled_from(["dense", "sparse", "other"]), ranking))
def test_every_key_comes_out_once_and_scores_never_increase(
    rankings: dict[str, list[str]],
) -> None:
    result = rrf(rankings)
    expected = {key for keys in rankings.values() for key in keys}
    assert sorted(item.key for item in result) == sorted(expected)
    scores = [item.score for item in result]
    assert scores == sorted(scores, reverse=True)
