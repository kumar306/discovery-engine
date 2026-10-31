"""The paired bootstrap: how much better one system is than another, and how sure we are."""

import random
import statistics

import pytest

from eval.metrics import PairedGain, paired_bootstrap, queries_needed


def test_two_identical_systems_have_no_gain_and_no_noise() -> None:
    scores = [0.2, 0.9, 0.5, 0.0, 1.0]
    gain = paired_bootstrap(scores, scores, resamples=200, seed=1)
    assert gain == PairedGain(mean=0.0, low=0.0, high=0.0, sd=0.0)


def test_a_system_that_is_better_by_the_same_amount_on_every_query_has_a_sure_gain() -> None:
    worse = [0.2, 0.4, 0.6, 0.8]
    better = [0.3, 0.5, 0.7, 0.9]
    gain = paired_bootstrap(worse, better, resamples=200, seed=1)
    assert gain.mean == pytest.approx(0.1)
    assert gain.low == pytest.approx(0.1)
    assert gain.high == pytest.approx(0.1)


def test_two_queries_pulling_opposite_ways_leave_the_interval_wide_and_around_zero() -> None:
    # gains are +0.1 and -0.1, a draw of two queries has a mean of -0.1, 0 or +0.1
    gain = paired_bootstrap([0.5, 0.5], [0.6, 0.4], resamples=1000, seed=1)
    assert gain.mean == pytest.approx(0.0)
    assert gain.low == pytest.approx(-0.1)
    assert gain.high == pytest.approx(0.1)


def test_the_interval_always_contains_the_observed_mean_gain() -> None:
    first = [0.1, 0.9, 0.4, 0.4, 0.7, 0.2, 0.8, 0.3]
    second = [0.3, 0.8, 0.9, 0.4, 0.6, 0.5, 0.8, 0.1]
    gain = paired_bootstrap(first, second, resamples=1000, seed=3)
    assert gain.low <= gain.mean <= gain.high


def test_the_same_seed_gives_the_same_answer_and_another_seed_may_not() -> None:
    first = [0.1, 0.9, 0.4, 0.4, 0.7, 0.2, 0.8, 0.3]
    second = [0.3, 0.8, 0.9, 0.4, 0.6, 0.5, 0.8, 0.1]
    assert paired_bootstrap(first, second, resamples=500, seed=3) == paired_bootstrap(
        first, second, resamples=500, seed=3
    )
    assert paired_bootstrap(first, second, resamples=500, seed=3) != paired_bootstrap(
        first, second, resamples=500, seed=4
    )


def test_swapping_the_systems_flips_the_sign_of_the_gain() -> None:
    first = [0.1, 0.9, 0.4, 0.4]
    second = [0.3, 0.8, 0.9, 0.4]
    forward = paired_bootstrap(first, second, resamples=500, seed=2)
    backward = paired_bootstrap(second, first, resamples=500, seed=2)
    assert backward.mean == pytest.approx(-forward.mean)


def test_scores_that_do_not_belong_to_the_same_queries_are_refused() -> None:
    with pytest.raises(ValueError, match="same queries"):
        paired_bootstrap([0.1, 0.2], [0.1], resamples=10, seed=1)


def test_no_queries_and_no_resamples_are_refused() -> None:
    with pytest.raises(ValueError, match="no queries"):
        paired_bootstrap([], [], resamples=10, seed=1)
    with pytest.raises(ValueError, match="resample"):
        paired_bootstrap([0.1], [0.2], resamples=0, seed=1)


def test_the_interval_ends_are_the_2_5th_and_97_5th_percentile_not_the_extremes() -> None:
    # one query in ten is worse by 0.1, the rest are equal: a draw of ten queries holds k of the bad
    # one with k binomial(10, 0.1), and its mean gain is -0.01 * k
    # more than 3 bad ones in a draw happen in 1.3 percent of draws, less than the 2.5 percent that
    # the interval leaves out, so the lower end is -0.03 and not the rarest draw (-0.05 or less)
    before = [0.5] * 10
    after = [0.4] + [0.5] * 9
    loss = paired_bootstrap(before, after, resamples=4000, seed=1)
    assert loss.low == pytest.approx(-0.03)
    assert loss.high == pytest.approx(0.0, abs=1e-12)
    # the mirror image: one query better by 0.1
    gain = paired_bootstrap(after, before, resamples=4000, seed=1)
    assert gain.low == pytest.approx(0.0, abs=1e-12)
    assert gain.high == pytest.approx(0.03)


def test_the_spread_of_the_gain_over_the_queries_is_reported() -> None:
    # gains +0.1 and -0.1: mean 0, sample standard deviation sqrt(0.02) = 0.1414
    gain = paired_bootstrap([0.5, 0.5], [0.6, 0.4], resamples=100, seed=1)
    assert gain.sd == pytest.approx(0.1 * 2**0.5)
    # the same gain on every query has no spread
    assert paired_bootstrap([0.2, 0.4, 0.6], [0.3, 0.5, 0.7], resamples=100, seed=1).sd == (
        pytest.approx(0.0, abs=1e-12)
    )


def test_one_query_has_no_spread_to_report_so_it_is_zero() -> None:
    assert paired_bootstrap([0.2], [0.5], resamples=10, seed=1).sd == 0.0


# ---------------- the confidence of the interval, and how many queries a gain needs

FIRST_TWELVE = [0.1, 0.9, 0.4, 0.4, 0.7, 0.2, 0.8, 0.3, 0.55, 0.65, 0.05, 0.95]
SECOND_TWELVE = [0.3, 0.8, 0.9, 0.4, 0.6, 0.5, 0.8, 0.1, 0.60, 0.85, 0.25, 0.90]


def test_the_default_confidence_gives_the_numbers_it_gave_before_the_confidence_existed() -> None:
    # pinned from the code before this change: 95 percent is still the 25th and the 975th of
    # 1000 sorted means, to the last digit
    gain = paired_bootstrap(FIRST_TWELVE, SECOND_TWELVE, resamples=1000, seed=3)
    assert gain == PairedGain(
        mean=0.08333333333333333,
        low=-0.01666666666666665,
        high=0.1958333333333333,
        sd=0.20037842985589827,
    )


def test_asking_for_95_percent_is_the_same_as_not_asking() -> None:
    assert paired_bootstrap(
        FIRST_TWELVE, SECOND_TWELVE, resamples=1000, seed=3, confidence=0.95
    ) == paired_bootstrap(FIRST_TWELVE, SECOND_TWELVE, resamples=1000, seed=3)


def test_a_higher_confidence_gives_a_wider_interval_around_the_same_mean() -> None:
    narrow = paired_bootstrap(FIRST_TWELVE, SECOND_TWELVE, resamples=1000, seed=3, confidence=0.80)
    normal = paired_bootstrap(FIRST_TWELVE, SECOND_TWELVE, resamples=1000, seed=3)
    wide = paired_bootstrap(FIRST_TWELVE, SECOND_TWELVE, resamples=1000, seed=3, confidence=0.983)
    assert narrow.mean == normal.mean == wide.mean
    assert narrow.low > normal.low > wide.low
    assert narrow.high < normal.high < wide.high


def test_a_confidence_of_zero_or_one_or_outside_is_refused() -> None:
    for confidence in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="confidence"):
            paired_bootstrap([0.1, 0.2], [0.2, 0.3], resamples=10, seed=1, confidence=confidence)


def test_queries_needed_matches_the_hand_computed_values_for_a_spread_of_0_1() -> None:
    # gains -0.1, 0 and +0.1 have a sample standard deviation of exactly 0.1
    gains = [-0.1, 0.0, 0.1]
    # ((1.96 + 0.8416) * 0.10 / 0.02) ** 2 = 196.2, rounded up
    assert queries_needed(gains, 0.02) == 197
    # the same with a target of 0.01: 784.9, rounded up
    assert queries_needed(gains, 0.01) == 785


def test_a_bigger_target_needs_fewer_queries_and_more_power_needs_more() -> None:
    gains = [-0.1, 0.0, 0.1]
    assert queries_needed(gains, 0.04) < queries_needed(gains, 0.02)
    assert queries_needed(gains, 0.02, power=0.95) > queries_needed(gains, 0.02, power=0.8)


def test_queries_needed_refuses_a_target_of_zero_a_power_outside_zero_one_and_one_gain() -> None:
    with pytest.raises(ValueError, match="target"):
        queries_needed([-0.1, 0.1], 0.0)
    with pytest.raises(ValueError, match="power"):
        queries_needed([-0.1, 0.1], 0.02, power=1.0)
    with pytest.raises(ValueError, match="spread of the gains"):
        queries_needed([0.1], 0.02)


def sorted_means(first: list[float], second: list[float], resamples: int, seed: int) -> list[float]:
    # the same draws the function makes, written out here so the places can be checked
    gains = [after - before for before, after in zip(first, second, strict=True)]
    generator = random.Random(seed)
    return sorted(
        statistics.fmean(generator.choices(gains, k=len(gains))) for _ in range(resamples)
    )


def test_the_ends_of_the_interval_are_the_26th_and_975th_of_a_thousand_sorted_means() -> None:
    first = [((n * 37) % 101) / 101 for n in range(60)]
    second = [((n * 53 + 11) % 97) / 97 for n in range(60)]
    means = sorted_means(first, second, 1000, 5)
    # the neighbours differ, so a place that is off by one is seen
    assert means[24] < means[25]
    assert means[973] < means[974] < means[975]
    gain = paired_bootstrap(first, second, resamples=1000, seed=5)
    assert gain.low == means[25]
    assert gain.high == means[974]


def test_the_ends_of_a_98_3_percent_interval_are_the_9th_and_992nd_of_a_thousand_sorted_means() -> (
    None
):
    first = [((n * 37) % 101) / 101 for n in range(60)]
    second = [((n * 53 + 11) % 97) / 97 for n in range(60)]
    means = sorted_means(first, second, 1000, 5)
    assert means[7] < means[8]
    assert means[990] < means[991] < means[992]
    gain = paired_bootstrap(first, second, resamples=1000, seed=5, confidence=0.983)
    assert gain.low == means[8]
    assert gain.high == means[991]
