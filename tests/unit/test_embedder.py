"""The embedder's machinery with fake models: ordering, batching, deadline, errors, threads.

No tensors here on purpose, what the vectors mean is tested against the real model separately.
"""

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pytest

from libs.embedding import model as embedder_module
from libs.embedding.model import EmbedPermanent, EmbedTransient, Limits, ModelMismatch, OnnxEmbedder
from libs.resilience.retry import FailureAction, decide
from services.embed.settings import EmbedSettings
from tests.support.hash_embedder import WhitespaceTokenizer

DIM = 4


def vector(text: str) -> np.ndarray:
    # different for every text we use, so a result can be matched to the text it came from
    return np.array([float(len(text)), float(sum(map(ord, text)) % 997), 0.0, 1.0])


@dataclass
class Sparse:
    indices: np.ndarray
    values: np.ndarray


class FakeDense:
    def __init__(
        self,
        dim: int = DIM,
        fail_on: str | None = None,
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.dim = dim
        self.fail_on = fail_on
        self.error = error
        self.delay = delay
        self.batches: list[list[str]] = []
        self.threads: list[str] = []

    def embed(self, documents: list[str], **kwargs: Any) -> list[np.ndarray]:
        self.batches.append(list(documents))
        self.threads.append(threading.current_thread().name)
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None and any(self.fail_on in d for d in documents if self.fail_on):
            raise self.error
        return [np.resize(vector(d), self.dim) for d in documents]


class FakeSparse:
    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on

    def embed(self, documents: list[str], **kwargs: Any) -> list[Sparse]:
        if self.fail_on and any(self.fail_on in d for d in documents):
            raise ValueError("cannot tokenise this")
        return [
            Sparse(np.array([len(d), 7]), np.array([1.5, 2.5]))
            if d.strip()
            else Sparse(np.array([], dtype=int), np.array([]))
            for d in documents
        ]


def make(
    dense: FakeDense | None = None, sparse: FakeSparse | None = None, sub_batch: int = 16
) -> tuple[OnnxEmbedder, FakeDense]:
    dense = dense or FakeDense()
    embedder = OnnxEmbedder(
        dense=dense,
        sparse=sparse or FakeSparse(),
        tokenizer=WhitespaceTokenizer(),
        limits=Limits(max_tokens=512),
        dense_dim=DIM,
        sub_batch=sub_batch,
    )
    return embedder, dense


# ordering and batching


async def test_nothing_to_embed_never_touches_the_model() -> None:
    embedder, dense = make()
    assert await embedder.embed([], timeout_s=5) == []
    assert dense.batches == []


async def test_results_come_back_in_the_order_the_texts_were_sent() -> None:
    embedder, _ = make()
    texts = ["a b c d e f", "a", "a b c", "a b", "a b c d"]
    results = await embedder.embed(texts, timeout_s=5)
    assert [r.dense for r in results] == [vector(t).tolist() for t in texts]


async def test_the_shortest_texts_are_embedded_together_so_batches_pad_little() -> None:
    embedder, dense = make(sub_batch=2)
    await embedder.embed(["a b c d e f", "a", "a b c", "a b", "a b c d"], timeout_s=5)
    assert dense.batches == [["a", "a b"], ["a b c", "a b c d"], ["a b c d e f"]]


async def test_a_sub_batch_never_holds_more_than_the_configured_number_of_texts() -> None:
    embedder, dense = make(sub_batch=3)
    await embedder.embed([f"text number {n}" for n in range(7)], timeout_s=5)
    assert [len(batch) for batch in dense.batches] == [3, 3, 1]


async def test_the_answer_is_plain_python_numbers_for_the_qdrant_adapter() -> None:
    embedder, _ = make()
    (result,) = await embedder.embed(["one two"], timeout_s=5)
    assert type(result.dense) is list
    assert all(type(x) is float for x in result.dense)
    assert all(type(i) is int for i in result.sparse_indices)
    assert all(type(v) is float for v in result.sparse_values)


async def test_a_text_without_indexable_terms_gets_an_empty_sparse_vector() -> None:
    embedder, _ = make()
    (result,) = await embedder.embed(["   "], timeout_s=5)
    assert (result.sparse_indices, result.sparse_values) == ([], [])


# the deadline


async def test_a_deadline_that_has_already_passed_does_no_work() -> None:
    embedder, dense = make()
    with pytest.raises(EmbedTransient, match="deadline"):
        await embedder.embed(["a", "b"], timeout_s=0)
    assert dense.batches == []


async def test_the_deadline_is_checked_between_sub_batches_and_stops_the_rest() -> None:
    # every batch takes 0.1 s and the budget is 0.15 s, so two batches run and the third never does
    embedder, dense = make(FakeDense(delay=0.1), sub_batch=1)
    with pytest.raises(EmbedTransient, match="deadline"):
        await embedder.embed(["a", "b", "c"], timeout_s=0.15)
    assert len(dense.batches) == 2


# errors: which are the text's fault, which are the machine's


async def test_the_text_that_breaks_a_batch_is_found_and_named_by_its_position() -> None:
    dense = FakeDense(fail_on="POISON", error=ValueError("bad input"))
    embedder, _ = make(dense)
    with pytest.raises(EmbedPermanent) as caught:
        await embedder.embed(["good a", "POISON x", "good b"], timeout_s=5)
    assert caught.value.index == 1  # the position in what the caller sent, not in a sorted batch
    # the whole batch failed once, then the texts were tried one by one until the bad one
    assert dense.batches[0] == ["good a", "POISON x", "good b"]
    assert ["POISON x"] in dense.batches[1:]


async def test_a_single_text_that_fails_is_permanent_without_any_second_try() -> None:
    dense = FakeDense(fail_on="POISON", error=ValueError("bad input"))
    embedder, _ = make(dense)
    with pytest.raises(EmbedPermanent) as caught:
        await embedder.embed(["POISON"], timeout_s=5)
    assert caught.value.index == 0
    assert len(dense.batches) == 1


async def test_a_failing_single_text_batch_is_blamed_by_its_position_in_what_was_sent() -> None:
    # sub batches of one, and the shorter text goes first, so the bad text is alone in its batch
    # and its place in that batch (0) is not its place in what the caller sent (1)
    dense = FakeDense(fail_on="POISON", error=ValueError("bad input"))
    embedder, _ = make(dense, sub_batch=1)
    with pytest.raises(EmbedPermanent) as caught:
        await embedder.embed(["fine one", "POISON"], timeout_s=5)
    assert caught.value.index == 1


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(MemoryError(), id="python memory error"),
        pytest.param(RuntimeError("Failed to allocate memory for requested buffer"), id="onnx"),
        pytest.param(RuntimeError("CPU out of memory"), id="out of memory"),
    ],
)
async def test_running_out_of_memory_is_transient_and_is_not_blamed_on_a_text(
    error: Exception,
) -> None:
    dense = FakeDense(fail_on="a", error=error)
    embedder, _ = make(dense)
    with pytest.raises(EmbedTransient):
        await embedder.embed(["a one", "a two"], timeout_s=5)
    assert len(dense.batches) == 1  # no hunting for a guilty text, the machine is the problem


async def test_a_failure_in_the_sparse_model_is_traced_to_its_text_too() -> None:
    embedder, _ = make(sparse=FakeSparse(fail_on="BAD"))
    with pytest.raises(EmbedPermanent) as caught:
        await embedder.embed(["fine", "BAD text", "fine too"], timeout_s=5)
    assert caught.value.index == 1


async def test_a_model_of_the_wrong_size_is_a_setup_mistake_not_a_bad_text() -> None:
    dense = FakeDense(dim=8)
    embedder, _ = make(dense)
    with pytest.raises(ModelMismatch, match="8 floats, expected 4"):
        await embedder.embed(["a", "b", "c"], timeout_s=5)
    assert len(dense.batches) == 1  # not isolated text by text, no text is at fault


async def test_verify_stops_a_worker_whose_model_does_not_fit_the_collection() -> None:
    good, _ = make()
    await good.verify()
    bad, _ = make(FakeDense(dim=8))
    with pytest.raises(ModelMismatch):
        await bad.verify()


def test_the_errors_plug_into_the_existing_retry_decisions() -> None:
    assert (
        decide(EmbedPermanent(0, ValueError("x")), attempt=1, max_attempts=3)
        is FailureAction.FAILURE_DEAD_LETTER
    )
    assert decide(EmbedTransient("slow"), attempt=1, max_attempts=3) is FailureAction.FAILURE_RETRY


# threads: the model runs beside the event loop, never on it


async def test_the_model_runs_on_its_own_thread_and_the_event_loop_stays_free() -> None:
    embedder, dense = make(FakeDense(delay=0.3))
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        await embedder.embed(["a"], timeout_s=5)
    finally:
        task.cancel()
    assert dense.threads[0].startswith("embed")
    assert threading.current_thread().name != dense.threads[0]
    # the loop kept ticking for the 0.3 s the model was busy, a blocked loop would show 0 or 1
    assert ticks >= 5


async def test_two_callers_at_once_never_run_the_model_at_the_same_time() -> None:
    running = 0
    most = 0

    class Counting(FakeDense):
        def embed(self, documents: list[str], **kwargs: Any) -> list[np.ndarray]:
            nonlocal running, most
            running += 1
            most = max(most, running)
            try:
                return super().embed(documents, **kwargs)
            finally:
                running -= 1

    embedder, _ = make(Counting(delay=0.1))
    await asyncio.gather(embedder.embed(["a"], 5), embedder.embed(["b"], 5))
    assert most == 1  # the cores are shared by onnx's own threads, not by two batches at once


# the real factory: the settings reach the models


class FakeHfTokenizer:
    truncation: ClassVar[dict[str, int]] = {"max_length": 77}

    @classmethod
    def from_str(cls, text: str) -> "FakeHfTokenizer":
        return cls()

    def to_str(self) -> str:
        return "tokenizer"

    def no_truncation(self) -> None: ...

    def no_padding(self) -> None: ...


def test_the_real_models_are_built_with_the_configured_threads_length_and_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: dict[str, Any] = {}

    class FakeTextEmbedding:
        def __init__(self, name: str, threads: int) -> None:
            built["dense"] = (name, threads)
            self.model = type("Inner", (), {"tokenizer": FakeHfTokenizer()})()

    class FakeSparseTextEmbedding:
        def __init__(self, name: str, avg_len: float) -> None:
            built["sparse"] = (name, avg_len)

    monkeypatch.setattr(embedder_module, "TextEmbedding", FakeTextEmbedding)
    monkeypatch.setattr(embedder_module, "SparseTextEmbedding", FakeSparseTextEmbedding)
    settings = EmbedSettings(
        dense_model="org/some-model", embed_threads=3, bm25_avg_len=64.0, embed_sub_batch=5
    )
    embedder = embedder_module.build_onnx_embedder(settings)
    try:
        assert built == {"dense": ("org/some-model", 3), "sparse": ("Qdrant/bm25", 64.0)}
        assert embedder.limits.max_tokens == 77  # read from the model, not written down here
        assert embedder._sub_batch == 5
    finally:
        embedder.close()
