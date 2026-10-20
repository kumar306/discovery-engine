import re
from collections.abc import Callable
from dataclasses import dataclass

from libs.embedding.collection import normalise
from libs.embedding.model import Limits, Tokenizer
from libs.resilience.retry import PermanentError
from libs.schemas import UpsertRequest
from libs.schemas.hashing import content_sha

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class ComposeError(PermanentError):
    """The product cannot be turned into chunks; retrying will not help (a permanent error)."""


@dataclass(frozen=True)
class Chunk:
    text: str
    content_hash: str
    tokens: int
    positions: tuple[int, ...]


@dataclass(frozen=True)
class _Field:
    units: list[str]
    separator: str

    @property
    def text(self) -> str:
        return self.separator.join(self.units)


def compose(product: UpsertRequest, tokenizer: Tokenizer, limits: Limits) -> tuple[Chunk, ...]:
    header = " | ".join(
        part
        for part in (
            normalise(product.title),
            normalise(product.brand or ""),
            normalise(product.color or ""),
        )
        if part
    )

    def count(text: str) -> int:
        return len(tokenizer.spans(text))

    def build(parts: list[str]) -> str:
        return "\n".join([header, *parts])

    def fits(parts: list[str]) -> bool:
        return count(build(parts)) <= limits.budget

    if not fits([]):  # no body parts: does the header alone fit?
        raise ComposeError("the header alone exceeds the token budget")

    texts: list[str] = []
    current: list[str] = []
    for field in _body_fields(product):
        if fits([*current, field.text]):
            current.append(field.text)
            continue
        if current:
            texts.append(build(current))
            current = []
        if fits([field.text]):
            current = [field.text]
            continue
        texts.extend(build([piece]) for piece in _split_field(field, tokenizer, fits))
    if current or not texts:
        texts.append(build(current))

    positions: dict[str, list[int]] = {}
    for index, text in enumerate(texts):
        positions.setdefault(text, []).append(index)
    return tuple(
        Chunk(text, content_sha(text.encode("utf-8")), count(text), tuple(where))
        for text, where in positions.items()
    )


def _body_fields(product: UpsertRequest) -> list[_Field]:
    bullets = [b for b in (normalise(bullet) for bullet in product.bullets) if b]
    description = normalise(product.description or "")
    sentences = [s for s in _SENTENCE_END.split(description) if s]
    fields = [_Field(bullets, "\n"), _Field(sentences, " ")]
    return [field for field in fields if field.units]


def _split_field(
    field: _Field, tokenizer: Tokenizer, fits: Callable[[list[str]], bool]
) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    for unit in field.units:
        if fits([field.separator.join([*current, unit])]):
            current.append(unit)
            continue
        if current:
            pieces.append(field.separator.join(current))
            current = []
        if fits([unit]):
            current = [unit]
        else:
            pieces.extend(_cut_at_tokens(unit, tokenizer, fits))
    if current:
        pieces.append(field.separator.join(current))
    return pieces


def _cut_at_tokens(text: str, tokenizer: Tokenizer, fits: Callable[[list[str]], bool]) -> list[str]:
    pieces: list[str] = []
    while text:
        ends = [end for _, end in tokenizer.spans(text)]
        # binary search for the largest k such that the first k tokens fit; k = 0 means none do
        low, high = 0, len(ends)
        while low < high:
            middle = (low + high + 1) // 2
            if fits([text[: ends[middle - 1]]]):
                low = middle
            else:
                high = middle - 1
        if low == 0:
            raise ComposeError("not even one token fits next to the header")
        pieces.append(text[: ends[low - 1]])
        text = text[ends[low - 1] :].lstrip()
    return pieces
