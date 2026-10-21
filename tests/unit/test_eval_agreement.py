"""How much two rankings of the same chunks agree: overlap at k, same first result, order."""

import pytest

from eval.metrics import overlap_at, rank_correlation, score_gap_by_rank, top1_equal

# overlap at k


def test_overlap_of_identical_lists_is_one_and_of_disjoint_lists_is_zero() -> None:
    assert overlap_at(3, ["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert overlap_at(3, ["a", "b", "c"], ["x", "y", "z"]) == 0.0


def test_overlap_is_the_share_of_the_top_k_that_both_lists_have() -> None:
    assert overlap_at(4, ["a", "b", "c", "d"], ["a", "b", "x", "y"]) == 0.5
    assert overlap_at(3, ["a", "b", "c"], ["c", "a", "d"]) == pytest.approx(2 / 3)


def test_the_order_inside_the_top_k_does_not_matter_and_only_the_top_k_counts() -> None:
    assert overlap_at(3, ["a", "b", "c"], ["c", "b", "a"]) == 1.0
    assert overlap_at(2, ["a", "b", "x"], ["a", "b", "c"]) == 1.0


def test_two_identical_short_lists_are_a_perfect_match_not_a_partial_one() -> None:
    # a search with only 6 matching chunks has 6 results on both sides, k = 10 must not make that
    # look like a disagreement
    six = ["a", "b", "c", "d", "e", "f"]
    assert overlap_at(10, six, six) == 1.0
    assert overlap_at(10, ["a", "b"], ["b", "a"]) == 1.0


def test_a_short_list_against_a_longer_one_is_divided_by_the_longer() -> None:
    assert overlap_at(10, ["a"], ["a", "b", "c"]) == pytest.approx(1 / 3)
    assert overlap_at(10, ["a", "b", "c"], ["a"]) == pytest.approx(1 / 3)
    # only the top k counts, so the longer list is cut to k before it is measured
    assert overlap_at(2, ["a", "b", "c"], ["a", "b", "d"]) == 1.0


def test_when_both_lists_are_empty_there_is_nothing_to_compare_so_it_is_skipped() -> None:
    assert overlap_at(10, [], []) is None


def test_when_only_one_list_is_empty_the_lists_are_completely_disjoint() -> None:
    assert overlap_at(10, [], ["a", "b"]) == 0.0
    assert overlap_at(10, ["a", "b"], []) == 0.0


def test_k_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        overlap_at(0, ["a"], ["a"])


# the first result


def test_the_first_results_are_equal_when_they_are_the_same_chunk() -> None:
    assert top1_equal(["a", "b"], ["a", "c"]) is True
    assert top1_equal(["a", "b"], ["b", "a"]) is False


def test_when_both_lists_are_empty_the_first_result_is_skipped_not_counted_as_equal() -> None:
    assert top1_equal([], []) is None


def test_when_only_one_list_is_empty_the_first_results_are_not_equal() -> None:
    assert top1_equal([], ["a"]) is False
    assert top1_equal(["a"], []) is False


# the order


def test_the_same_order_is_one_and_the_reverse_order_is_minus_one() -> None:
    assert rank_correlation(["a", "b", "c", "d"], ["a", "b", "c", "d"]) == 1.0
    assert rank_correlation(["a", "b", "c", "d"], ["d", "c", "b", "a"]) == -1.0


def test_one_swap_of_neighbours_among_four_is_point_eight() -> None:
    assert rank_correlation(["a", "b", "c", "d"], ["a", "c", "b", "d"]) == pytest.approx(0.8)


def test_ids_that_are_in_only_one_list_are_ignored() -> None:
    assert rank_correlation(["a", "x", "b", "c"], ["a", "b", "y", "c"]) == 1.0


def test_fewer_than_two_shared_ids_has_no_correlation() -> None:
    assert rank_correlation(["a", "b"], ["x", "y"]) is None
    assert rank_correlation(["a", "b"], ["a", "y"]) is None
    assert rank_correlation([], []) is None


def test_an_id_that_repeats_inside_a_list_is_refused() -> None:
    with pytest.raises(ValueError, match="distinct"):
        rank_correlation(["a", "b", "a"], ["a", "b", "c"])
    with pytest.raises(ValueError, match="distinct"):
        rank_correlation(["a", "b", "c"], ["a", "c", "c"])


# the score at each rank


def test_equal_scores_at_every_rank_have_no_gap() -> None:
    assert score_gap_by_rank([14.09, 14.09, 12.5], [14.09, 14.09, 12.5]) == 0.0


def test_two_chunks_with_the_same_score_in_either_order_leave_no_gap() -> None:
    # the lists hold scores only, so two tied chunks standing in the other order change nothing
    assert (
        score_gap_by_rank([19.19, 12.42, 12.42, 12.42, 11.0], [19.19, 12.42, 12.42, 12.42, 11.0])
        == 0.0
    )


def test_float32_noise_is_a_tiny_gap() -> None:
    # qdrant stores float32, the oracle computes float64, the same score differs after 7 digits
    ours = [19.1853, 19.073452, 18.218664]
    reference = [19.18530003094184, 19.073451441352084, 18.218663772500353]
    gap = score_gap_by_rank(ours, reference)
    assert gap is not None
    assert gap < 1e-6


def test_a_missing_best_chunk_moves_every_score_up_a_rank_and_shows_as_a_gap() -> None:
    # the reference has [14, 13, 12, 11], ours lost its best chunk and got a new last one: a
    # comparison chunk by chunk would skip the lost chunk, rank by rank sees 13 where 14 should be
    reference = [14.0, 13.0, 12.0, 11.0]
    ours = [13.0, 12.0, 11.0, 10.5]
    assert score_gap_by_rank(ours, reference) == pytest.approx(1 / 12)


def test_the_gap_is_relative_to_the_larger_score_and_cannot_pass_one() -> None:
    assert score_gap_by_rank([10.0], [8.0]) == pytest.approx(0.2)
    assert score_gap_by_rank([8.0], [10.0]) == pytest.approx(0.2)
    assert score_gap_by_rank([0.0], [5.0]) == 1.0
    assert score_gap_by_rank([0.0], [0.0]) == 0.0


def test_a_rank_that_only_one_list_has_is_the_largest_gap() -> None:
    assert score_gap_by_rank([5.0], [5.0, 4.0]) == 1.0
    assert score_gap_by_rank([5.0, 4.0], [5.0]) == 1.0
    assert score_gap_by_rank([], [5.0]) == 1.0


def test_two_empty_lists_have_nothing_to_compare_so_it_is_skipped() -> None:
    assert score_gap_by_rank([], []) is None
