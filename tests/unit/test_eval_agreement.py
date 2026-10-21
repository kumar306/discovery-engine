"""How much two rankings of the same chunks agree: overlap at k, same first result, order."""

import pytest

from eval.metrics import overlap_at, rank_correlation, top1_equal

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


def test_a_list_shorter_than_k_cannot_reach_one() -> None:
    assert overlap_at(3, ["a"], ["a"]) == pytest.approx(1 / 3)


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
