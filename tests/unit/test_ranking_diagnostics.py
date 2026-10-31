"""Reading a ranking: how much of the score is finding products and how much is ordering them."""

import math
import statistics

import pytest

from eval.metrics import ndcg_at_k
from eval.ranking_diagnostics import (
    HARD_QUERY_MAX_JUDGED,
    QueryNumbers,
    RankedLists,
    diagnose,
    group_label,
    hard_query_ids,
    is_fusion_loss,
    is_hard_query,
    judged_groups,
    judged_in_top,
    random_order_ndcg,
    rerank_ceiling,
)

# a query with an exact, a substitute and an irrelevant product
JUDGED = {"a": "E", "b": "S", "c": "I"}


def lists(query_id: str, dense: list[str], bm25: list[str], hybrid: list[str]) -> RankedLists:
    return RankedLists(query_id, tuple(dense), tuple(bm25), tuple(hybrid))


# ---------------- how many judged products are in the first ten


def test_judged_in_top_counts_every_letter_and_only_the_first_ten_places() -> None:
    ranked = ["a", "u1", "b", "c"] + [f"u{n}" for n in range(2, 8)] + ["a2"]
    judged = {"a": "E", "b": "S", "c": "I", "a2": "E"}
    # a, b and c are in the first ten (c is irrelevant but judged), a2 sits at place 11
    assert judged_in_top(ranked, judged) == 3
    assert judged_in_top(ranked[:2], judged) == 1
    assert judged_in_top([], judged) == 0


def test_there_is_one_definition_of_a_hard_query_and_it_is_two_of_ten() -> None:
    assert HARD_QUERY_MAX_JUDGED == 2


def test_a_query_with_two_judged_products_in_the_first_ten_is_hard_and_three_is_not() -> None:
    judged = {"a": "E", "b": "S", "c": "I"}
    assert is_hard_query(["x", "a", "y", "b"], judged)
    assert not is_hard_query(["x", "a", "y", "b", "c"], judged)
    assert is_hard_query([], judged)


# ---------------- the groups of the table


@pytest.mark.parametrize(
    ("count", "label"),
    [
        (0, "0 to 2 judged"),
        (2, "0 to 2 judged"),
        (3, "3 to 5 judged"),
        (5, "3 to 5 judged"),
        (6, "6 to 8 judged"),
        (8, "6 to 8 judged"),
        (9, "9 or 10 judged"),
        (10, "9 or 10 judged"),
    ],
)
def test_each_count_of_judged_products_belongs_to_one_group(count: int, label: str) -> None:
    assert group_label(count) == label


def test_the_hard_group_is_exactly_the_hard_queries() -> None:
    for count in range(0, 11):
        assert (group_label(count) == "0 to 2 judged") == (count <= HARD_QUERY_MAX_JUDGED)


def test_groups_come_best_first_with_their_means_and_an_empty_group_is_left_out() -> None:
    rows = [
        QueryNumbers("1", 10, 0.75, 0.5, 0.25),
        QueryNumbers("2", 9, 0.25, 0.5, 0.75),
        QueryNumbers("3", 1, 0.0, 0.5, 0.0),
    ]
    groups = judged_groups(rows)
    assert [(g.label, g.queries) for g in groups] == [("9 or 10 judged", 2), ("0 to 2 judged", 1)]
    assert (groups[0].hybrid, groups[0].dense, groups[0].bm25) == (0.5, 0.5, 0.5)
    assert (groups[1].hybrid, groups[1].dense, groups[1].bm25) == (0.0, 0.5, 0.0)


# ---------------- the random order baseline


def test_two_products_have_two_orders_and_the_baseline_is_their_mean() -> None:
    judged = {"a": "E", "b": "S"}
    best = ndcg_at_k(["a", "b"], judged)
    worst = ndcg_at_k(["b", "a"], judged)
    assert best == 1.0
    assert worst == pytest.approx(0.6877, abs=1e-3)
    assert random_order_ndcg(judged, draws=2000, seed=1) == pytest.approx(
        (best + worst) / 2, abs=0.02
    )


def test_when_every_product_has_the_same_letter_every_order_is_perfect() -> None:
    assert random_order_ndcg({"a": "E", "b": "E", "c": "E"}, draws=50, seed=1) == 1.0


def test_the_baseline_is_the_same_for_the_same_seed_and_needs_a_draw() -> None:
    first = random_order_ndcg(JUDGED, draws=100, seed=5)
    assert first == random_order_ndcg(JUDGED, draws=100, seed=5)
    with pytest.raises(ValueError, match="draw"):
        random_order_ndcg(JUDGED, draws=0, seed=5)


def test_the_baseline_does_not_change_the_judgements_it_was_given() -> None:
    judged = {"a": "E", "b": "S"}
    random_order_ndcg(judged, draws=10, seed=1)
    assert judged == {"a": "E", "b": "S"}


# ---------------- the ceiling of a perfect reranker


def test_a_perfect_reranker_of_twenty_finds_an_exact_product_that_sits_at_place_fifteen() -> None:
    ranked = [f"u{n}" for n in range(14)] + ["a"]
    assert ndcg_at_k(ranked, {"a": "E"}) == 0.0
    assert rerank_ceiling(ranked, {"a": "E"}, depth=20) == 1.0


def test_a_depth_of_ten_cannot_reach_a_product_that_sits_below_it() -> None:
    ranked = [f"u{n}" for n in range(14)] + ["a"]
    assert rerank_ceiling(ranked, {"a": "E"}, depth=10) == 0.0


def test_the_ceiling_counts_only_what_was_found_so_a_missing_exact_product_stays_missing() -> None:
    assert rerank_ceiling(["u1", "u2", "b"], {"a": "E", "b": "S"}, depth=100) == pytest.approx(
        ndcg_at_k(["b"], {"a": "E", "b": "S"})
    )


def test_the_ceiling_sorts_by_gain_and_keeps_our_order_among_equal_gains() -> None:
    ranked = ["s1", "e1", "s2", "e2"]
    judged = {"s1": "S", "s2": "S", "e1": "E", "e2": "E"}
    assert rerank_ceiling(ranked, judged, depth=4) == pytest.approx(
        ndcg_at_k(["e1", "e2", "s1", "s2"], judged)
    )
    assert rerank_ceiling(ranked, judged, depth=4) == 1.0


def test_the_ceiling_is_never_below_the_score_of_our_own_order() -> None:
    ranked = ["c", "b", "x", "a"]
    assert rerank_ceiling(ranked, JUDGED, depth=100) >= ndcg_at_k(ranked, JUDGED)


def test_a_depth_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="depth"):
        rerank_ceiling(["a"], JUDGED, depth=0)


# ---------------- the fusion loss


def test_hybrid_loses_when_the_better_single_rung_is_ahead_by_more_than_the_margin() -> None:
    assert is_fusion_loss(hybrid=0.25, dense=0.5, bm25=0.0)
    assert is_fusion_loss(hybrid=0.25, dense=0.0, bm25=0.5)


def test_hybrid_does_not_lose_by_a_small_gap_or_when_it_is_the_best() -> None:
    assert not is_fusion_loss(hybrid=0.375, dense=0.5, bm25=0.0)
    assert not is_fusion_loss(hybrid=0.5, dense=0.25, bm25=0.25)


# ---------------- all of it together on three queries worked by hand
#
# q1: every rung is perfect. judged 3 of 10 in the hybrid top ten
# q2: dense is perfect, hybrid puts three irrelevant products above the exact one:
#     hybrid ndcg = 1 / log2(5) = 0.4307 (a fusion loss), judged 4 of 10
# q3: nothing judged is returned: ndcg 0 on every rung, judged 0 of 10, a hard query

J1 = {"a": "E", "b": "S", "c": "I"}
J2 = {"a": "E", "b": "I", "c": "I", "d": "I"}
J3 = {"a": "E"}
JUDGEMENTS = {"1": J1, "2": J2, "3": J3}
RANKINGS = {
    "1": lists("1", ["a", "b", "c"], ["a", "b"], ["a", "b", "c"]),
    "2": lists("2", ["a", "b", "c", "d"], ["b", "c"], ["b", "c", "d", "a"]),
    "3": lists("3", ["x", "y"], ["x", "y"], ["x", "y"]),
}
Q2_HYBRID = 1 / math.log2(5)
# the hard queries of the three, fixed by the caller: only q3 has 2 or fewer judged in its top ten
HARD = frozenset({"3"})


def test_the_means_with_and_without_the_hard_queries() -> None:
    result = diagnose(RANKINGS, JUDGEMENTS, hard_queries=HARD, draws=10, seed=1)
    assert result.queries == 3
    assert result.hybrid_ndcg == pytest.approx((1 + Q2_HYBRID + 0) / 3)
    assert result.dense_ndcg == pytest.approx((1 + 1 + 0) / 3)
    assert result.bm25_ndcg == pytest.approx((1 + 0 + 0) / 3)
    assert result.hard_queries == 1
    assert result.hybrid_ndcg_without_hard == pytest.approx((1 + Q2_HYBRID) / 2)
    assert result.dense_ndcg_without_hard == pytest.approx(1.0)
    assert result.bm25_ndcg_without_hard == pytest.approx(0.5)


def test_the_means_without_hard_queries_do_not_exist_when_every_query_is_hard() -> None:
    result = diagnose({"3": RANKINGS["3"]}, {"3": J3}, hard_queries=HARD, draws=10, seed=1)
    assert result.hard_queries == 1
    assert result.hybrid_ndcg_without_hard is None
    assert result.dense_ndcg_without_hard is None
    assert result.bm25_ndcg_without_hard is None


def test_the_judged_share_the_recall_and_the_fusion_losses() -> None:
    result = diagnose(RANKINGS, JUDGEMENTS, hard_queries=HARD, draws=10, seed=1)
    # judged products in the first ten of the hybrid list: 3, 4 and 0, out of ten places each
    assert result.judged_share_top10 == pytest.approx((0.3 + 0.4 + 0.0) / 3)
    # the exact product a is in the first hundred of the hybrid list for q1 and q2, not for q3
    assert result.hybrid_recall_at_100 == pytest.approx(2 / 3)
    # dense finds a for q1 and q2 and nothing for q3
    assert result.dense_recall_at_100 == pytest.approx(2 / 3)
    assert result.fusion_losses == 1
    # best of the three per query: 1, 1 and 0
    assert result.best_of_three == pytest.approx(2 / 3)


def test_the_ceilings_are_what_a_perfect_reranker_makes_of_the_hybrid_list() -> None:
    result = diagnose(RANKINGS, JUDGEMENTS, hard_queries=HARD, draws=10, seed=1)
    # q1 is already perfect, q2 becomes perfect (a is in its list), q3 has nothing to reorder
    assert result.ceiling_top20 == pytest.approx(2 / 3)
    assert result.ceiling_top100 == pytest.approx(2 / 3)


def test_the_baseline_is_the_mean_of_the_baseline_of_each_query() -> None:
    result = diagnose(RANKINGS, JUDGEMENTS, hard_queries=HARD, draws=200, seed=3)
    expected = statistics.fmean(
        random_order_ndcg(JUDGEMENTS[q], draws=200, seed=3) for q in ("1", "2", "3")
    )
    assert result.random_order_ndcg == pytest.approx(expected)


def test_the_per_query_numbers_come_in_the_order_of_the_rankings() -> None:
    result = diagnose(RANKINGS, JUDGEMENTS, hard_queries=HARD, draws=10, seed=1)
    assert [row.query_id for row in result.per_query] == ["1", "2", "3"]
    assert [row.judged_top10 for row in result.per_query] == [3, 4, 0]
    assert result.per_query[1].hybrid == pytest.approx(Q2_HYBRID)
    assert result.per_query[1].dense == 1.0
    assert result.per_query[1].bm25 == 0.0


def test_the_groups_of_the_table_cover_every_query_once() -> None:
    result = diagnose(RANKINGS, JUDGEMENTS, hard_queries=HARD, draws=10, seed=1)
    assert sum(group.queries for group in result.groups) == 3
    assert [g.label for g in result.groups] == ["3 to 5 judged", "0 to 2 judged"]


def test_a_run_with_no_queries_is_refused() -> None:
    with pytest.raises(ValueError, match="no queries"):
        diagnose({}, {}, hard_queries=frozenset(), draws=10, seed=1)


def test_a_ranked_query_nobody_judged_is_refused() -> None:
    with pytest.raises(ValueError, match="judg"):
        diagnose(
            {"9": lists("9", ["a"], ["a"], ["a"])},
            JUDGEMENTS,
            hard_queries=frozenset(),
            draws=10,
            seed=1,
        )


# ---------------- each rung is read for what it is, not for what another rung did

J5 = {"a": "E", "b": "S", "c": "I", "d": "I"}


def test_hardness_is_read_from_the_hybrid_list_not_from_the_dense_one() -> None:
    # hybrid has 4 judged products in its first ten, dense has none
    one = {"1": lists("1", ["x"], ["x"], ["a", "b", "c", "d"])}
    assert hard_query_ids(one, {"1": J5}) == frozenset()
    # and the other way round: dense has 4 judged, hybrid none
    other = {"1": lists("1", ["a", "b", "c", "d"], ["x"], ["x"])}
    assert hard_query_ids(other, {"1": J5}) == frozenset({"1"})


def test_the_best_of_three_counts_bm25_when_bm25_is_the_best() -> None:
    only_bm25 = {"1": lists("1", ["x"], ["a", "b", "c", "d"], ["x"])}
    assert (
        diagnose(only_bm25, {"1": J5}, hard_queries=frozenset(), draws=5, seed=1).best_of_three
        == 1.0
    )


def test_the_best_of_three_counts_dense_when_dense_is_the_best() -> None:
    only_dense = {"1": lists("1", ["a", "b", "c", "d"], ["x"], ["x"])}
    assert (
        diagnose(only_dense, {"1": J5}, hard_queries=frozenset(), draws=5, seed=1).best_of_three
        == 1.0
    )


def test_the_hybrid_and_the_dense_recall_are_each_read_from_their_own_list() -> None:
    found_by_hybrid = {"1": lists("1", ["x"], ["x"], ["a"])}
    result = diagnose(found_by_hybrid, {"1": J5}, hard_queries=frozenset(), draws=5, seed=1)
    assert (result.hybrid_recall_at_100, result.dense_recall_at_100) == (1.0, 0.0)
    found_by_dense = {"1": lists("1", ["a"], ["x"], ["x"])}
    result = diagnose(found_by_dense, {"1": J5}, hard_queries=frozenset(), draws=5, seed=1)
    assert (result.hybrid_recall_at_100, result.dense_recall_at_100) == (0.0, 1.0)


# ---------------- the hard queries are fixed ONCE and handed to every model


def test_the_hard_queries_of_rankings_are_the_ones_with_two_or_fewer_judged_in_the_top() -> None:
    assert hard_query_ids(RANKINGS, JUDGEMENTS) == frozenset({"3"})


def test_no_rankings_have_no_hard_queries_and_an_unjudged_query_is_refused() -> None:
    assert hard_query_ids({}, JUDGEMENTS) == frozenset()
    with pytest.raises(ValueError, match="judg"):
        hard_query_ids({"9": lists("9", ["a"], ["a"], ["a"])}, JUDGEMENTS)


def test_diagnose_uses_the_hard_set_it_is_given_and_not_its_own() -> None:
    # q1 is perfect on every rung and has 3 judged products in its top ten: by the rule it is not
    # hard, but another model made it hard, and this model is scored on that same set
    result = diagnose(RANKINGS, JUDGEMENTS, hard_queries=frozenset({"1"}), draws=10, seed=1)
    assert result.hard_queries == 1
    assert result.hybrid_ndcg_without_hard == pytest.approx((Q2_HYBRID + 0) / 2)
    assert result.dense_ndcg_without_hard == pytest.approx(1 / 2)
    assert result.bm25_ndcg_without_hard == pytest.approx(0.0)
    # the groups still describe this model's own top ten, q3 is in the group of 0 to 2 judged
    assert [(g.label, g.queries) for g in result.groups] == [
        ("3 to 5 judged", 2),
        ("0 to 2 judged", 1),
    ]


def test_an_empty_hard_set_leaves_the_means_without_hard_equal_to_the_means() -> None:
    result = diagnose(RANKINGS, JUDGEMENTS, hard_queries=frozenset(), draws=10, seed=1)
    assert result.hard_queries == 0
    assert result.hybrid_ndcg_without_hard == pytest.approx(result.hybrid_ndcg)


def test_a_hard_query_that_was_not_ranked_is_refused() -> None:
    with pytest.raises(ValueError, match="hard"):
        diagnose(RANKINGS, JUDGEMENTS, hard_queries=frozenset({"1", "77"}), draws=10, seed=1)
