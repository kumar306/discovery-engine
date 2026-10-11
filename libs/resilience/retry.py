import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum


class TransientError(Exception):
    """Raise this from an adapter when a retry may help."""


class PermanentError(Exception):
    """Raise this from an adapter when a retry cannot help."""


class DependencyUnavailable(Exception):
    """A circuit breaker is open: the dependency is down, not this one message."""


class Verdict(StrEnum):
    RETRY = "retry"
    DEAD_LETTER = "dead_letter"
    PAUSE = "pause"
    COMMIT = "commit"
    STALL = "stall"


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.1
    max_delay: float = 5.0


class Rejected(Exception):
    def __init__(self, verdict: Verdict, error: Exception, attempts: int) -> None:
        super().__init__(f"{verdict} after {attempts} attempt(s): {error!r}")
        self.verdict = verdict
        self.error = error
        self.attempts = attempts


def decide(error: Exception, attempt: int, max_attempts: int) -> Verdict:
    # an open breaker says the dependency is down, so never dead letter the message for it
    if isinstance(error, DependencyUnavailable):
        return Verdict.PAUSE
    # bad input stays bad, ValueError covers pydantic validation, json and compose errors
    if isinstance(error, PermanentError | ValueError):
        return Verdict.DEAD_LETTER
    # timeouts, 5xx and anything unknown get the bounded retries, then the dead letter queue
    if attempt < max_attempts:
        return Verdict.RETRY
    return Verdict.DEAD_LETTER


def after_dead_letter(published: bool) -> Verdict:
    # offset is committed only once the dead letter record is safely written
    return Verdict.COMMIT if published else Verdict.STALL


def backoff_delay(attempt: int, policy: RetryPolicy, rng: random.Random) -> float:
    # full jitter: anywhere between zero and an exponentially growing ceiling
    ceiling = min(policy.max_delay, policy.base_delay * 2 ** (attempt - 1))
    return rng.uniform(0, ceiling)


async def call_with_retries[T](
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    sleep: Callable[[float], Awaitable[None]],
    rng: random.Random,
) -> T:
    attempt = 0
    while True:
        attempt += 1
        try:
            return await operation()
        except Exception as error:
            verdict = decide(error, attempt, policy.max_attempts)
            if verdict is not Verdict.RETRY:
                raise Rejected(verdict, error, attempt) from error
            await sleep(backoff_delay(attempt, policy, rng))
