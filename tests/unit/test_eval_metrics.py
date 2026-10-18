"""The ranking metrics on small examples worked out by hand.

Every expected value is written as the arithmetic it comes from, so a reader can follow it with a
pencil. The gains are the standard ones of the ESCI task: E 1.0, S 0.1, C 0.01, I 0.
"""

from math import log2

import pytest

from eval.metrics import (
    GAINS,
    QueryScores,
    Scores,
    evaluate,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    score_query,
)

# one product of every kind, as the judges labelled them for some query
JUDGED = {"a": "E", "b": "S", "c": "C", "d": "I"}

# the best possible order puts the exact product first, then the substitute, then the complement
IDEAL_DCG = 1.0 / log2(2) + 0.1 / log2(3) + 0.01 / log2(4)


def test_the_standard_gains() -> None:
    assert GAINS == {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


# ndcg


def test_a_perfect_ranking_scores_one() -> None:
    assert ndcg_at_k(["a", "b", "c", "d"], JUDGED) == pytest.approx(1.0)


def test_the_reverse_ranking_scores_less_because_the_good_products_come_late() -> None:
    # rank 1 d (gain 0), rank 2 c, rank 3 b, rank 4 a, each divided by log2(rank + 1)
    got = 0.0 / log2(2) + 0.01 / log2(3) + 0.1 / log2(4) + 1.0 / log2(5)
    assert ndcg_at_k(["d", "c", "b", "a"], JUDGED) == pytest.approx(got / IDEAL_DCG)


def test_a_product_nobody_judged_counts_as_not_relevant() -> None:
    # "x" was never judged, so it takes rank 1 and pushes "a" down to rank 2
    assert ndcg_at_k(["x", "a"], {"a": "E"}) == pytest.approx(1.0 / log2(3))


def test_fewer_results_than_the_cutoff_are_fine() -> None:
    # only "a" was returned, the ideal list would also have had "b" at rank 2
    ideal = 1.0 / log2(2) + 0.1 / log2(3)
    assert ndcg_at_k(["a"], {"a": "E", "b": "S"}) == pytest.approx(1.0 / ideal)


def test_nothing_after_the_cutoff_counts() -> None:
    ranked = [f"x{n}" for n in range(10)] + ["a"]  # "a" is at rank 11
    assert ndcg_at_k(ranked, {"a": "E"}, k=10) == 0.0
    assert ndcg_at_k(ranked, {"a": "E"}, k=11) > 0.0


def test_the_ideal_list_is_cut_at_the_same_cutoff() -> None:
    # twelve exact products exist, but only ten can be in the top ten, so ten at the top is perfect
    judged = {f"p{n}": "E" for n in range(12)}
    assert ndcg_at_k([f"p{n}" for n in range(10)], judged, k=10) == pytest.approx(1.0)


def test_a_query_with_nothing_relevant_scores_zero_not_an_error() -> None:
    assert ndcg_at_k(["a", "b"], {"a": "I", "b": "I"}) == 0.0
    assert ndcg_at_k(["a"], {}) == 0.0


def test_an_empty_ranking_scores_zero() -> None:
    assert ndcg_at_k([], JUDGED) == 0.0


# recall


def test_recall_is_the_share_of_exact_products_that_were_found() -> None:
    judged = {"a": "E", "b": "E", "c": "S", "d": "I"}
    assert recall_at_k(["a", "x", "c"], judged, k=100) == pytest.approx(1 / 2)


def test_recall_looks_only_at_the_exact_products_by_default() -> None:
    # a substitute is not what the shopper asked for, so finding only that is a miss
    assert recall_at_k(["b"], {"a": "E", "b": "S"}, k=100) == 0.0


def test_recall_can_be_told_that_substitutes_count_too() -> None:
    assert recall_at_k(["b"], {"a": "E", "b": "S"}, k=100, relevant={"E", "S"}) == pytest.approx(
        0.5
    )


def test_recall_with_nothing_to_find_is_zero() -> None:
    assert recall_at_k(["a"], {"a": "I"}, k=100) == 0.0


def test_recall_ignores_what_is_beyond_the_cutoff() -> None:
    ranked = ["x0", "x1", "a"]
    assert recall_at_k(ranked, {"a": "E"}, k=2) == 0.0
    assert recall_at_k(ranked, {"a": "E"}, k=3) == 1.0


# mrr


def test_mrr_is_one_over_the_rank_of_the_first_exact_product() -> None:
    assert mrr_at_k(["x", "y", "a", "b"], {"a": "E", "b": "E"}, k=10) == pytest.approx(1 / 3)


def test_mrr_skips_products_that_are_not_exact() -> None:
    assert mrr_at_k(["b", "c", "a"], {"a": "E", "b": "S", "c": "C"}, k=10) == pytest.approx(1 / 3)


def test_mrr_is_zero_when_the_first_exact_product_is_past_the_cutoff() -> None:
    ranked = [f"x{n}" for n in range(10)] + ["a"]
    assert mrr_at_k(ranked, {"a": "E"}, k=10) == 0.0


def test_mrr_is_zero_when_there_is_no_exact_product_at_all() -> None:
    assert mrr_at_k(["a", "b"], {"a": "S", "b": "C"}, k=10) == 0.0


# the whole set of queries


def test_the_scores_of_one_query() -> None:
    scores = score_query(["b", "a", "z"], {"a": "E", "b": "S"})
    assert scores == QueryScores(
        ndcg_at_10=pytest.approx((0.1 / log2(2) + 1.0 / log2(3)) / (1.0 + 0.1 / log2(3))),
        recall_at_100=1.0,
        mrr_at_10=0.5,
    )


def test_the_score_of_a_set_is_the_mean_over_its_judged_queries() -> None:
    judgements = {"q1": {"a": "E"}, "q2": {"a": "E"}}
    run = {"q1": ["a"], "q2": ["x", "a"]}
    scores = evaluate(run, judgements)
    assert scores.queries == 2
    assert scores.ndcg_at_10 == pytest.approx((1.0 + 1.0 / log2(3)) / 2)
    assert scores.mrr_at_10 == pytest.approx((1.0 + 0.5) / 2)
    assert scores.recall_at_100 == 1.0


def test_a_query_the_system_returned_nothing_for_counts_as_zero_and_is_not_skipped() -> None:
    judgements = {"q1": {"a": "E"}, "q2": {"a": "E"}}
    scores = evaluate({"q1": ["a"]}, judgements)  # q2 is missing from the run
    assert scores.queries == 2
    assert scores.ndcg_at_10 == pytest.approx(0.5)
    assert scores.recall_at_100 == pytest.approx(0.5)
    assert scores.mrr_at_10 == pytest.approx(0.5)


def test_a_query_in_the_run_that_nobody_judged_is_ignored() -> None:
    scores = evaluate({"q1": ["a"], "stray": ["a"]}, {"q1": {"a": "E"}})
    assert scores == Scores(1.0, 1.0, 1.0, 1)


def test_no_judged_queries_is_an_error_and_not_a_silent_zero() -> None:
    with pytest.raises(ValueError, match="no judged queries"):
        evaluate({"q1": ["a"]}, {})


# input that would give a wrong number is refused loudly


def test_a_product_ranked_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="twice"):
        ndcg_at_k(["a", "b", "a"], JUDGED)


def test_an_unknown_label_is_refused() -> None:
    with pytest.raises(ValueError, match="label"):
        ndcg_at_k(["a"], {"a": "X"})


@pytest.mark.parametrize("k", [0, -1])
def test_a_cutoff_below_one_is_refused(k: int) -> None:
    with pytest.raises(ValueError, match="cutoff"):
        ndcg_at_k(["a"], JUDGED, k=k)
