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


# pause and stall both hold the partition, commit no offset and retry later with a capped backoff
# they stay two verdicts because the cause differs and the alert, the metric and the log line
# should say which one it was, a granular error is easier to act on than one catch all
# pause: a dependency is known to be down, an open breaker, so the message is not even tried and
# no attempt is used up, it resumes when the breaker lets a probe through
# stall: this message was tried and its retries ran out, or its dead letter record could not be
# written, so the trouble is on the path of the message and not a known outage
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
    # only an error marked permanent is the fault of the message, bad input stays bad
    # a bare ValueError is not trusted, it is as likely a bug of ours, examples we get are
    # int("abc"), uuid.UUID("not a uuid"), a pydantic ValidationError for a missing field, a
    # json.JSONDecodeError for a truncated file, and our own guard ValueError("a version with
    # no live chunks cannot be published")
    # the pipeline turns the known bad input ones into PermanentError where it parses a message
    # and everything else gets the retries and then the stall, so a bug of ours raises an alert
    # and does not push good messages into the dead letter queue
    if isinstance(error, PermanentError):
        return Verdict.DEAD_LETTER
    # timeouts, 5xx and anything unknown get the bounded retries
    if attempt < max_attempts:
        return Verdict.RETRY
    # the retries are used up and nothing says the message itself is bad, so the dependency or
    # the machine is the problem and the message stays where it is, the partition stalls and
    # alerts, moving a perfectly good message to the dead letter queue because something was
    # briefly unhealthy is an anti pattern, the message is fine and would need a manual replay
    return Verdict.STALL


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
