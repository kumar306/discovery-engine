"""Properties of the C2 gate over arbitrary event sequences (a first slice of P-ORDER)."""

from hypothesis import given
from hypothesis import strategies as st

from services.embed.gate import GateDecision, GateState, gate

versions = st.lists(st.integers(min_value=0, max_value=50), max_size=40)


def process(state: GateState | None, version: int) -> GateState | None:
    """What a worker that never crashes does with one event: pass the gate, then index it."""
    if gate(version, state) is GateDecision.PROCEED:
        return GateState(current_version=version, indexed=True)
    return state


@given(versions)
def test_current_version_never_decreases(delivered: list[int]) -> None:
    state: GateState | None = None
    for version in delivered:
        after = process(state, version)
        if state is not None:
            assert after is not None
            assert after.current_version >= state.current_version
        state = after


@given(versions)
def test_state_ends_at_the_highest_version_delivered(delivered: list[int]) -> None:
    state: GateState | None = None
    for version in delivered:
        state = process(state, version)
    if delivered:
        assert state is not None
        assert state.current_version == max(delivered)
    else:
        assert state is None
