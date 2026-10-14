"""C2 version gate: should the worker process this event, or drop it as stale or already done?"""

import pytest

from services.embed.gate import GateState, VersionGateDecision, gate

PROCEED = VersionGateDecision.VERSION_GATE_PROCEED
DROP = VersionGateDecision.VERSION_GATE_DROP


@pytest.mark.parametrize(
    ("version", "state", "expected"),
    [
        pytest.param(0, None, PROCEED, id="unknown document, version zero"),
        pytest.param(5, None, PROCEED, id="unknown document"),
        pytest.param(6, GateState(5, indexed=True), PROCEED, id="higher than an indexed version"),
        pytest.param(6, GateState(5, indexed=False), PROCEED, id="higher than a stuck version"),
        pytest.param(5, GateState(5, indexed=False), PROCEED, id="equal, not yet indexed: a retry"),
        pytest.param(5, GateState(5, indexed=True), DROP, id="equal and indexed: a duplicate"),
        pytest.param(0, GateState(0, indexed=True), DROP, id="duplicate at version zero"),
        pytest.param(4, GateState(5, indexed=True), DROP, id="lower than an indexed version"),
        pytest.param(4, GateState(5, indexed=False), DROP, id="lower than a stuck version"),
    ],
)
def test_gate_decision(
    version: int, state: GateState | None, expected: VersionGateDecision
) -> None:
    assert gate(version, state) is expected


def test_delete_is_gated_like_any_other_version() -> None:
    tombstone = GateState(current_version=5, indexed=True)  # a delete at version 5 is done
    assert gate(6, tombstone) is PROCEED  # a newer upsert may bring the product back
    assert gate(5, tombstone) is DROP  # the delete delivered again
    assert gate(3, tombstone) is DROP  # a delayed older upsert must not resurrect it
