from dataclasses import dataclass
from enum import StrEnum


class GateDecision(StrEnum):
    PROCEED = "proceed"
    DROP = "drop"


@dataclass(frozen=True)
class GateState:
    current_version: int
    indexed: bool


def gate(version: int, state: GateState | None) -> GateDecision:
    if state is None or version > state.current_version:
        return GateDecision.PROCEED
    if version == state.current_version and not state.indexed:
        return GateDecision.PROCEED
    return GateDecision.DROP
