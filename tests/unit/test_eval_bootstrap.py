"""The paired bootstrap: how much better one system is than another, and how sure we are."""

import pytest

from eval.metrics import PairedGain, paired_bootstrap


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
