from dataclasses import dataclass
from enum import StrEnum


class VersionGateDecision(StrEnum):
    VERSION_GATE_PROCEED = "proceed"
    VERSION_GATE_DROP = "drop"


@dataclass(frozen=True)
class GateState:
    current_version: int
    indexed: bool


def gate(version: int, state: GateState | None) -> VersionGateDecision:
    if state is None or version > state.current_version:
        return VersionGateDecision.VERSION_GATE_PROCEED
    if version == state.current_version and not state.indexed:
        return VersionGateDecision.VERSION_GATE_PROCEED
    return VersionGateDecision.VERSION_GATE_DROP
