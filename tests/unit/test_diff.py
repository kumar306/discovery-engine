"""C5 hash diff: which chunks of the new version need embedding, which stay, which go.

Chunks are named by single letters here: `uid("a")` is a fixed chunk id and `letters(...)` turns a
tuple of ids back into a readable string such as "ac". A mapping like {a: (0,), b: (1, 3)} reads
"chunk a sits at position 0, chunk b sits at positions 1 and 3" (the same text twice in a product).
"""

from collections.abc import Iterable
from uuid import UUID

import pytest
from hypothesis import given
from hypothesis import strategies as st

from services.embed.diff import Hit, Positions, diff_chunks


def uid(letter: str) -> UUID:
    return UUID(int=ord(letter))


def letters(chunk_ids: Iterable[UUID]) -> str:
    return "".join(chr(chunk_id.int) for chunk_id in chunk_ids)


def layout(**placed: Positions) -> dict[UUID, Positions]:
    return {uid(letter): positions for letter, positions in placed.items()}


def stale_of(text: str) -> set[UUID]:
    return {uid(letter) for letter in text}


# 22. classification: hit, miss, removed (and revived from the stale queue)


@pytest.mark.parametrize(
    ("new", "current", "stale", "hits", "misses", "removed", "revived"),
    [
        pytest.param(
            layout(a=(0,), b=(1,)),
            {},
            set(),
            "",
            "ab",
            "",
            "",
            id="first version: everything is new",
        ),
        pytest.param(
            layout(a=(0,), b=(1,)),
            layout(a=(0,), b=(1,)),
            set(),
            "ab",
            "",
            "",
            "",
            id="nothing changed",
        ),
        pytest.param(
            layout(a=(0,), x=(1,), c=(2,)),
            layout(a=(0,), b=(1,), c=(2,)),
            set(),
            "ac",
            "x",
            "b",
            "",
            id="one chunk edited",
        ),
        pytest.param(
            layout(a=(0,)),
            layout(a=(0,), b=(1,), c=(2,)),
            set(),
            "a",
            "",
            "bc",
            "",
            id="product got shorter",
        ),
        pytest.param(
            layout(a=(0,), b=(1,), c=(2,)),
            layout(a=(0,), b=(1,)),
            set(),
            "ab",
            "c",
            "",
            "",
            id="product got longer",
        ),
        pytest.param(
            layout(x=(0,), y=(1,)),
            layout(a=(0,), b=(1,)),
            set(),
            "",
            "xy",
            "ab",
            "",
            id="every chunk replaced, as after a title or model change",
        ),
        pytest.param(
            layout(a=(0, 2), b=(1,)),
            layout(a=(0, 2), b=(1,)),
            set(),
            "ab",
            "",
            "",
            "",
            id="a repeated chunk is one hit",
        ),
        pytest.param(
            layout(a=(0,), r=(1,)),
            layout(a=(0,)),
            {uid("r")},
            "a",
            "r",
            "",
            "r",
            id="a chunk waiting for cleanup comes back: re-embedded and revived",
        ),
        pytest.param(
            layout(a=(0,)),
            layout(a=(0,)),
            {uid("q")},
            "a",
            "",
            "",
            "",
            id="unrelated stale chunks are ignored",
        ),
    ],
)
def test_classification(
    new: dict[UUID, Positions],
    current: dict[UUID, Positions],
    stale: set[UUID],
    hits: str,
    misses: str,
    removed: str,
    revived: str,
) -> None:
    result = diff_chunks(new, current, stale)
    assert letters(hit.chunk_id for hit in result.hits) == hits
    assert letters(result.misses) == misses
    assert letters(result.removed) == removed
    assert letters(result.revived) == revived


# 24. a chunk that moved is still a hit, and says where it is now


def test_a_chunk_pushed_later_by_an_insertion_is_a_moved_hit() -> None:
    new = layout(x=(0,), a=(1,), b=(2,))
    current = layout(a=(0,), b=(1,))
    result = diff_chunks(new, current, set())
    assert result.hits == (Hit(uid("a"), (1,), moved=True), Hit(uid("b"), (2,), moved=True))
    assert letters(result.misses) == "x"


def test_a_chunk_that_stays_put_is_not_moved() -> None:
    result = diff_chunks(layout(a=(0,), x=(1,)), layout(a=(0,), b=(1,)), set())
    assert result.hits == (Hit(uid("a"), (0,), moved=False),)


def test_swapped_chunks_are_both_moved() -> None:
    result = diff_chunks(layout(b=(0,), a=(1,)), layout(a=(0,), b=(1,)), set())
    assert result.hits == (Hit(uid("b"), (0,), moved=True), Hit(uid("a"), (1,), moved=True))


def test_a_chunk_that_gains_a_repeat_is_moved() -> None:
    result = diff_chunks(layout(a=(0, 1)), layout(a=(0,)), set())
    assert result.hits == (Hit(uid("a"), (0, 1), moved=True),)


def test_results_are_in_position_order() -> None:
    new = layout(d=(3,), a=(0,), c=(2,), b=(1,))
    current = layout(z=(2,), y=(0,), x=(1,))
    result = diff_chunks(new, current, set())
    assert letters(result.misses) == "abcd"
    # removed ones come back in the order they sat in the old version
    assert letters(result.removed) == "yxz"


# properties: the three groups always account for every chunk exactly once

UNIVERSE = "abcdefgh"
# a product's chunks in order, repeats mean the same text twice
sequences = st.lists(st.sampled_from(UNIVERSE), max_size=12)


def placed(sequence: list[str]) -> dict[UUID, Positions]:
    where: dict[UUID, list[int]] = {}
    for position, letter in enumerate(sequence):
        where.setdefault(uid(letter), []).append(position)
    return {chunk_id: tuple(found) for chunk_id, found in where.items()}


@given(sequences, sequences, st.sets(st.sampled_from(UNIVERSE)))
def test_every_new_chunk_is_a_hit_or_a_miss_and_every_old_one_stays_or_goes(
    new_sequence: list[str], current_sequence: list[str], stale_letters: set[str]
) -> None:
    new, current = placed(new_sequence), placed(current_sequence)
    result = diff_chunks(new, current, {uid(letter) for letter in stale_letters})
    hit_ids = {hit.chunk_id for hit in result.hits}
    assert hit_ids | set(result.misses) == set(new)
    assert not hit_ids & set(result.misses)
    assert hit_ids <= set(current)
    assert not set(result.misses) & set(current)
    assert set(result.removed) == set(current) - set(new)
    assert set(result.revived) == set(result.misses) & {uid(letter) for letter in stale_letters}
    assert all(hit.moved == (hit.positions != current[hit.chunk_id]) for hit in result.hits)


@given(sequences, sequences)
def test_applying_the_diff_to_the_old_state_gives_the_new_state(
    new_sequence: list[str], current_sequence: list[str]
) -> None:
    new, current = placed(new_sequence), placed(current_sequence)
    result = diff_chunks(new, current, set())
    after = {chunk_id: positions for chunk_id, positions in current.items()}
    for chunk_id in result.removed:
        del after[chunk_id]
    for chunk_id in result.misses:
        after[chunk_id] = new[chunk_id]
    for hit in result.hits:
        after[hit.chunk_id] = hit.positions
    assert after == new


@given(sequences)
def test_diffing_a_state_against_itself_changes_nothing(sequence: list[str]) -> None:
    state = placed(sequence)
    result = diff_chunks(state, state, set())
    assert result.misses == ()
    assert result.removed == ()
    assert not any(hit.moved for hit in result.hits)
