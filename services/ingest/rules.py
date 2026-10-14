from dataclasses import dataclass
from enum import StrEnum


class WritePrecheck(StrEnum):
    WRITE_PRECHECK_PROCEED = "proceed"
    WRITE_PRECHECK_ACCEPT_DUPLICATE = "accept_duplicate"
    WRITE_PRECHECK_REJECT_STALE = "reject_stale"
    WRITE_PRECHECK_REJECT_CONFLICT = "reject_conflict"


@dataclass(frozen=True)
class KnownDocument:
    latest_version: int
    content_sha: str | None  # None when the latest version is a delete


def precheck(version: int, content_sha: str | None, known: KnownDocument | None) -> WritePrecheck:
    if known is None or version > known.latest_version:
        return WritePrecheck.WRITE_PRECHECK_PROCEED
    if version < known.latest_version:
        return WritePrecheck.WRITE_PRECHECK_REJECT_STALE
    if content_sha == known.content_sha:
        return WritePrecheck.WRITE_PRECHECK_ACCEPT_DUPLICATE
    return WritePrecheck.WRITE_PRECHECK_REJECT_CONFLICT
