"""C13 failure handling: what to do with an error, how long to wait, when to give up."""

import asyncio
import json
import random

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from libs.resilience.retry import (
    DependencyUnavailable,
    PermanentError,
    Rejected,
    RetryPolicy,
    TransientError,
    Verdict,
    after_dead_letter,
    backoff_delay,
    call_with_retries,
    decide,
)
from libs.schemas import UpsertRequest
from services.embed.compose import ComposeError

POLICY = RetryPolicy(max_attempts=3, base_delay=0.1, max_delay=5.0)


def a_validation_error() -> ValidationError:
    with pytest.raises(ValidationError) as exc_info:
        UpsertRequest.model_validate({})
    return exc_info.value


class BreakerAndBadInput(DependencyUnavailable, ValueError):
    """Both at once, to show that an open breaker wins."""


# 25. classification


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(TransientError("busy"), Verdict.RETRY, id="marked transient"),
        pytest.param(TimeoutError(), Verdict.RETRY, id="timeout"),
        pytest.param(ConnectionResetError(), Verdict.RETRY, id="connection reset"),
        pytest.param(OSError("disk"), Verdict.RETRY, id="other os error"),
        pytest.param(RuntimeError("boom"), Verdict.RETRY, id="unknown errors get the retries"),
        pytest.param(PermanentError("bad"), Verdict.DEAD_LETTER, id="marked permanent"),
        pytest.param(a_validation_error(), Verdict.DEAD_LETTER, id="schema validation failed"),
        pytest.param(ComposeError("header too big"), Verdict.DEAD_LETTER, id="cannot compose"),
        pytest.param(json.JSONDecodeError("bad", "{", 0), Verdict.DEAD_LETTER, id="undecodable"),
        pytest.param(DependencyUnavailable("breaker open"), Verdict.PAUSE, id="breaker open"),
        pytest.param(BreakerAndBadInput("both"), Verdict.PAUSE, id="breaker wins over bad input"),
    ],
)
def test_first_attempt_verdict(error: Exception, expected: Verdict) -> None:
    assert decide(error, attempt=1, max_attempts=3) is expected


@pytest.mark.parametrize(
    ("attempt", "expected"),
    [(1, Verdict.RETRY), (2, Verdict.RETRY), (3, Verdict.DEAD_LETTER), (4, Verdict.DEAD_LETTER)],
)
def test_a_transient_error_is_retried_until_the_attempts_run_out(
    attempt: int, expected: Verdict
) -> None:
    assert decide(TimeoutError(), attempt=attempt, max_attempts=3) is expected


@pytest.mark.parametrize("attempt", [1, 2, 3])
def test_permanent_errors_never_retry_and_an_open_breaker_never_dead_letters(attempt: int) -> None:
    assert decide(PermanentError("bad"), attempt, 3) is Verdict.DEAD_LETTER
    assert decide(DependencyUnavailable("down"), attempt, 3) is Verdict.PAUSE


# 27. when the dead letter record cannot be written the worker stalls


def test_the_offset_is_committed_only_after_the_dead_letter_record_is_written() -> None:
    assert after_dead_letter(published=True) is Verdict.COMMIT
    assert after_dead_letter(published=False) is Verdict.STALL


@given(st.integers(min_value=1, max_value=10))
def test_decide_never_commits_or_stalls_on_its_own(attempt: int) -> None:
    errors = [TimeoutError(), PermanentError("x"), DependencyUnavailable("x"), RuntimeError("x")]
    verdicts = {decide(error, attempt, 3) for error in errors}
    assert verdicts <= {Verdict.RETRY, Verdict.DEAD_LETTER, Verdict.PAUSE}


# 26. backoff: exponential ceiling with full jitter


class MaxRandom(random.Random):
    """Always draws the top of the range, so the ceiling itself can be asserted."""

    def uniform(self, a: float, b: float) -> float:
        return b


def test_the_ceiling_doubles_per_attempt_then_stops_at_the_cap() -> None:
    policy = RetryPolicy(max_attempts=9, base_delay=0.1, max_delay=1.0)
    ceilings = [backoff_delay(n, policy, MaxRandom()) for n in range(1, 7)]
    assert ceilings == [0.1, 0.2, 0.4, 0.8, 1.0, 1.0]


@given(st.integers(min_value=1, max_value=20), st.integers(min_value=0, max_value=10**6))
def test_a_delay_is_never_negative_and_never_above_the_ceiling(attempt: int, seed: int) -> None:
    delay = backoff_delay(attempt, POLICY, random.Random(seed))
    assert 0 <= delay <= min(POLICY.max_delay, POLICY.base_delay * 2 ** (attempt - 1))


def test_the_same_seed_gives_the_same_delays() -> None:
    first = [backoff_delay(4, POLICY, rng) for rng in [random.Random(7)] * 3]
    again = [backoff_delay(4, POLICY, rng) for rng in [random.Random(7)] * 3]
    assert first == again


def test_jitter_really_spreads_the_delays() -> None:
    rng = random.Random(1)
    assert len({backoff_delay(4, POLICY, rng) for _ in range(5)}) == 5


# 26. the retry loop, with a scripted operation and a sleep that never waits


class Scripted:
    """Raises or returns each outcome in turn, and keeps repeating the last one."""

    def __init__(self, *outcomes: object) -> None:
        self.outcomes = outcomes
        self.calls = 0

    async def __call__(self) -> object:
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


async def test_success_on_the_first_try_never_sleeps() -> None:
    operation, sleeps = Scripted("done"), Sleeps()
    assert await call_with_retries(operation, POLICY, sleeps, MaxRandom()) == "done"
    assert (operation.calls, sleeps.delays) == (1, [])


async def test_transient_errors_are_retried_with_growing_waits() -> None:
    operation, sleeps = Scripted(TimeoutError(), ConnectionResetError(), "done"), Sleeps()
    assert await call_with_retries(operation, POLICY, sleeps, MaxRandom()) == "done"
    assert operation.calls == 3
    assert sleeps.delays == [0.1, 0.2]


async def test_a_transient_error_that_never_clears_is_dead_lettered_after_three_attempts() -> None:
    operation, sleeps = Scripted(TimeoutError("slow")), Sleeps()
    with pytest.raises(Rejected) as exc_info:
        await call_with_retries(operation, POLICY, sleeps, MaxRandom())
    assert exc_info.value.verdict is Verdict.DEAD_LETTER
    assert exc_info.value.attempts == 3
    assert operation.calls == 3
    assert len(sleeps.delays) == 2  # no wait after the last attempt


async def test_a_permanent_error_is_dead_lettered_at_once() -> None:
    error = PermanentError("sha mismatch")
    operation, sleeps = Scripted(error), Sleeps()
    with pytest.raises(Rejected) as exc_info:
        await call_with_retries(operation, POLICY, sleeps, MaxRandom())
    assert exc_info.value.verdict is Verdict.DEAD_LETTER
    assert exc_info.value.error is error
    assert exc_info.value.__cause__ is error
    assert (operation.calls, sleeps.delays) == (1, [])


async def test_an_open_breaker_pauses_without_retrying() -> None:
    operation, sleeps = Scripted(DependencyUnavailable("qdrant down")), Sleeps()
    with pytest.raises(Rejected) as exc_info:
        await call_with_retries(operation, POLICY, sleeps, MaxRandom())
    assert exc_info.value.verdict is Verdict.PAUSE
    assert (operation.calls, sleeps.delays) == (1, [])


async def test_cancellation_is_never_swallowed() -> None:
    operation, sleeps = Scripted(asyncio.CancelledError()), Sleeps()
    with pytest.raises(asyncio.CancelledError):
        await call_with_retries(operation, POLICY, sleeps, MaxRandom())
    assert operation.calls == 1
