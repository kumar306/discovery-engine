from dataclasses import dataclass
from enum import StrEnum


class PrecheckOutcome(StrEnum):
    PROCEED = "proceed"
    ACCEPT_DUPLICATE = "accept_duplicate"
    REJECT_STALE = "reject_stale"
    REJECT_CONFLICT = "reject_conflict"


@dataclass(frozen=True)
class KnownDocument:
    latest_version: int
    content_sha: str | None  # None when the latest version is a delete


def precheck(version: int, content_sha: str | None, known: KnownDocument | None) -> PrecheckOutcome:
    if known is None or version > known.latest_version:
        return PrecheckOutcome.PROCEED
    if version < known.latest_version:
        return PrecheckOutcome.REJECT_STALE
    if content_sha == known.content_sha:
        return PrecheckOutcome.ACCEPT_DUPLICATE
    return PrecheckOutcome.REJECT_CONFLICT
