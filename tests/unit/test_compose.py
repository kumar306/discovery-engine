"""C4 composer: turn one product into chunks that fit the embedding model's token limit.

Tokens are counted with a fake tokenizer (one token per whitespace separated word), so every
number below can be checked by eye. The real model tokenizer is checked against the same
behaviour in slice S3.
"""

import re
import unicodedata
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from libs.schemas import UpsertRequest
from libs.schemas.hashing import content_sha
from services.embed.compose import Chunk, ComposeError, Limits, compose


class WhitespaceTokenizer:
    def spans(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


TOKENIZER = WhitespaceTokenizer()


def make_product(**overrides: Any) -> UpsertRequest:
    body: dict[str, Any] = {
        "corpus": "esci",
        "doc_id": "B07ABC1234",
        "version": 3,
        "title": "Wireless Mouse",
        "brand": "Logitech",
        "color": "Black",
        "bullets": ["Quiet clicks", "18 month battery"],
        "description": "A compact wireless mouse.",
        "locale": "us",
    }
    return UpsertRequest.model_validate({**body, **overrides})


def run(product: UpsertRequest, budget: int = 512) -> tuple[Chunk, ...]:
    return compose(product, TOKENIZER, Limits(max_tokens=budget, margin=0))


def only_title(title: str = "T1", **overrides: Any) -> UpsertRequest:
    """A product with only a title unless the test adds fields."""
    empty: dict[str, Any] = {"brand": None, "color": None, "bullets": [], "description": None}
    return make_product(title=title, **{**empty, **overrides})


def words(count: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(count))


def bodies(chunks: tuple[Chunk, ...]) -> list[str]:
    """Each chunk's text without its first line (the header)."""
    return [chunk.text.partition("\n")[2] for chunk in chunks]


# 13. which fields are embedded


def test_the_five_embedded_fields_reach_the_chunk_text() -> None:
    (chunk,) = run(make_product())
    for part in (
        "Wireless Mouse",
        "Logitech",
        "Black",
        "Quiet clicks",
        "18 month battery",
        "compact",
    ):
        assert part in chunk.text


def test_identifiers_and_metadata_are_never_embedded() -> None:
    (chunk,) = run(make_product(doc_id="SECRETID99", locale="jp", version=77))
    for leaked in ("SECRETID99", "jp", "77", "esci"):
        assert leaked not in chunk.text


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({}, "Wireless Mouse", id="title only"),
        pytest.param({"brand": "Logitech"}, "Wireless Mouse | Logitech", id="title and brand"),
        pytest.param({"color": "Black"}, "Wireless Mouse | Black", id="title and color"),
    ],
)
def test_missing_optional_fields_leave_no_trace(overrides: dict[str, Any], expected: str) -> None:
    (chunk,) = run(only_title("Wireless Mouse", **overrides))
    assert chunk.text == expected


# 14. normalisation


COMPOSED_CAFE = "caf" + chr(0xE9)  # the accent is part of one code point
DECOMPOSED_CAFE = "cafe" + chr(0x301)  # the letter e, then a separate combining accent


def test_text_is_normalised_to_nfc() -> None:
    assert COMPOSED_CAFE != DECOMPOSED_CAFE  # they look the same but are different code points
    (chunk,) = run(make_product(title=f"{DECOMPOSED_CAFE} mouse"))
    assert chunk.text.startswith(f"{COMPOSED_CAFE} mouse")
    assert unicodedata.is_normalized("NFC", chunk.text)


def test_runs_of_whitespace_collapse_to_one_space() -> None:
    (chunk,) = run(make_product(description="A  compact\twireless\n\n mouse."))
    assert chunk.text.endswith("\nA compact wireless mouse.")


def test_formatting_differences_do_not_change_the_hash() -> None:
    (tidy,) = run(make_product(title=f"{COMPOSED_CAFE} mouse", description="A compact mouse."))
    (messy,) = run(
        make_product(title=f"{DECOMPOSED_CAFE}   mouse", description="A  compact\n\tmouse.")
    )
    assert messy.content_hash == tidy.content_hash


# 15. a product that fits is one chunk


def test_a_short_product_is_one_chunk_with_a_fixed_layout() -> None:
    (chunk,) = run(make_product())
    assert chunk.text == (
        "Wireless Mouse | Logitech | Black\n"
        "Quiet clicks\n18 month battery\n"
        "A compact wireless mouse."
    )
    assert chunk.positions == (0,)
    assert chunk.tokens == len(chunk.text.split())
    assert chunk.content_hash == content_sha(chunk.text.encode("utf-8"))


# 16. the limit: exactly at it is one chunk, one token over is a split


def test_exactly_at_the_budget_is_one_chunk() -> None:
    chunks = run(only_title(description=words(7)), budget=8)  # header 1 token + 7 words
    assert len(chunks) == 1
    assert chunks[0].tokens == 8


def test_one_token_over_the_budget_splits() -> None:
    chunks = run(only_title(description=words(8)), budget=8)  # header 1 token + 8 words
    assert len(chunks) == 2
    assert all(chunk.tokens <= 8 for chunk in chunks)


def test_the_margin_is_kept_free() -> None:
    product = only_title(description=words(8))  # 9 tokens in total
    assert len(compose(product, TOKENIZER, Limits(max_tokens=10, margin=0))) == 1
    assert len(compose(product, TOKENIZER, Limits(max_tokens=10, margin=2))) == 2


# 17. splitting an oversized description: sentence boundaries, header on every chunk, nothing lost

SENTENCES = [f"s{i}a s{i}b s{i}c." for i in range(5)]  # five sentences of three tokens


def test_an_oversized_description_splits_at_sentence_boundaries() -> None:
    chunks = run(only_title(description=" ".join(SENTENCES)), budget=7)  # room for two sentences
    assert bodies(chunks) == [
        "s0a s0b s0c. s1a s1b s1c.",
        "s2a s2b s2c. s3a s3b s3c.",
        "s4a s4b s4c.",
    ]


def test_split_chunks_all_fit_start_with_the_header_and_keep_their_order() -> None:
    chunks = run(only_title(description=" ".join(SENTENCES)), budget=7)
    assert all(chunk.tokens <= 7 for chunk in chunks)
    assert all(chunk.text.startswith("T1\n") for chunk in chunks)
    assert [chunk.positions for chunk in chunks] == [(0,), (1,), (2,)]
    assert " ".join(bodies(chunks)) == " ".join(SENTENCES)


# 18. fields are atomic: a chunk takes whole fields, the next field starts a new chunk

BULLETS = ["b1 b2", "b3 b4"]  # four tokens once joined with a newline
DESCRIPTION = "d1 d2 d3 d4 d5 d6."  # six tokens


def test_fields_that_fit_together_share_a_chunk() -> None:
    chunks = run(only_title(bullets=BULLETS, description=DESCRIPTION), budget=11)
    assert [chunk.text for chunk in chunks] == ["T1\nb1 b2\nb3 b4\nd1 d2 d3 d4 d5 d6."]


def test_a_field_that_does_not_fit_starts_the_next_chunk() -> None:
    chunks = run(only_title(bullets=BULLETS, description=DESCRIPTION), budget=10)
    assert [chunk.text for chunk in chunks] == ["T1\nb1 b2\nb3 b4", "T1\nd1 d2 d3 d4 d5 d6."]


def test_an_oversized_description_never_borrows_from_the_bullets_chunk() -> None:
    description = "e1 e2 e3. f1 f2 f3. g1 g2 g3."  # nine tokens, too big next to the header
    chunks = run(only_title(bullets=["b1 b2"], description=description), budget=6)
    assert bodies(chunks) == ["b1 b2", "e1 e2 e3.", "f1 f2 f3.", "g1 g2 g3."]


def test_editing_the_description_leaves_the_bullets_chunk_untouched() -> None:
    before = run(only_title(bullets=["b1 b2"], description="e1 e2 e3. f1 f2 f3."), budget=6)
    after = run(only_title(bullets=["b1 b2"], description="e1 e2 e3. CHANGED f2 f3."), budget=6)
    assert before[0] == after[0]
    assert before[2] != after[2]


# 19. a single sentence that is too big: cut at token boundaries, header on every piece


def test_an_oversized_sentence_is_cut_at_token_boundaries() -> None:
    chunks = run(only_title(description=words(12)), budget=6)  # room for five words next to "T1"
    assert bodies(chunks) == [words(5), "w5 w6 w7 w8 w9", "w10 w11"]
    assert all(chunk.text.startswith("T1\n") and chunk.tokens <= 6 for chunk in chunks)


def test_an_oversized_sentence_among_normal_ones() -> None:
    description = "p1 p2. l1 l2 l3 l4 l5 l6 l7 l8. q1 q2."
    chunks = run(only_title(description=description), budget=5)  # room for four tokens
    assert bodies(chunks) == ["p1 p2.", "l1 l2 l3 l4", "l5 l6 l7 l8.", "q1 q2."]


@pytest.mark.parametrize(
    ("title", "budget"),
    [
        pytest.param("one two three four five six", 5, id="header alone is too big"),
        pytest.param("one two three four", 4, id="no room left for a single body token"),
    ],
)
def test_an_unusable_header_is_a_permanent_error(title: str, budget: int) -> None:
    with pytest.raises(ComposeError):
        run(only_title(title, description="some body text"), budget=budget)


def test_a_header_over_the_budget_is_an_error_even_without_a_body() -> None:
    with pytest.raises(ComposeError):
        run(only_title("one two three four five six"), budget=5)


# 20. identical chunks inside one product collapse to one, and keep their positions


def test_identical_chunks_collapse_and_remember_every_position() -> None:
    chunks = run(only_title(description="Aa bb. Cc dd. Aa bb."), budget=4)
    assert [(c.text, c.positions) for c in chunks] == [
        ("T1\nAa bb.", (0, 2)),
        ("T1\nCc dd.", (1,)),
    ]


def test_three_identical_chunks_are_one_chunk_at_three_positions() -> None:
    (chunk,) = run(only_title(description="Same words here. Same words here. Same words here."), 5)
    assert chunk.positions == (0, 1, 2)
    assert chunk.content_hash == content_sha(chunk.text.encode("utf-8"))


# 21. properties over arbitrary text: determinism, the limit, order, nothing lost

EXTRA = map(chr, (0xE9, 0x301, 0x9F20, 0x1F600, 0xA0))  # e-acute, accent mark, CJK, emoji, NBSP
ALPHABET = [*"abcXYZ019 .!?\n\t,-", *EXTRA]
free_text = st.text(alphabet=st.sampled_from(ALPHABET), max_size=1500)


@st.composite
def products(draw: st.DrawFn) -> UpsertRequest:
    return make_product(
        title="T" + draw(st.text(alphabet=st.sampled_from(ALPHABET), max_size=60)),
        brand=draw(st.none() | free_text.map(lambda s: "B" + s[:80])),
        color=draw(st.none() | free_text.map(lambda s: "C" + s[:80])),
        bullets=draw(st.lists(free_text.map(lambda s: "b" + s[:200]), max_size=5)),
        description=draw(st.none() | free_text.map(lambda s: "d" + s)),
    )


def expected_tokens(product: UpsertRequest) -> list[str]:
    fields = [*product.bullets, product.description or ""]
    return [token for field in fields for token in unicodedata.normalize("NFC", field).split()]


@given(products(), st.integers(min_value=8, max_value=80))
def test_composition_properties(product: UpsertRequest, budget: int) -> None:
    try:
        chunks = run(product, budget)
    except ComposeError:
        return
    assert chunks == run(product, budget)  # deterministic
    headers = {chunk.text.partition("\n")[0] for chunk in chunks}
    assert len(headers) == 1  # one header, repeated on every chunk
    assert all(chunk.tokens <= budget for chunk in chunks)
    assert all(chunk.content_hash == content_sha(chunk.text.encode("utf-8")) for chunk in chunks)
    positions = sorted(p for chunk in chunks for p in chunk.positions)
    assert positions == list(range(len(positions)))  # every position exactly once
    in_order = sorted(((p, c) for c in chunks for p in c.positions), key=lambda pair: pair[0])
    body_tokens = [token for _, c in in_order for token in c.text.partition("\n")[2].split()]
    assert body_tokens == expected_tokens(product)  # nothing lost, nothing added, order kept
