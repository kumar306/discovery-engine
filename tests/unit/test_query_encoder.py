"""The query encoder folds the text for both vectors, and the query side of the embedder."""

import asyncio
import time
from collections.abc import Callable
from typing import Any

import pytest

from libs.embedding.model import EmbedTransient, Limits, ModelMismatch, OnnxEmbedder
from services.query.encoder import OnnxQueryEncoder, SparseQuery


class RecordingEmbedder:
    """Stands in for the embedder and keeps what the encoder asked it."""

    def __init__(self) -> None:
        self.dense_calls: list[tuple[str, float]] = []
        self.sparse_calls: list[str] = []

    async def embed_query_dense(self, text: str, timeout_s: float) -> list[float]:
        self.dense_calls.append((text, timeout_s))
        return [1.0, 2.0]

    async def embed_query_sparse(self, text: str) -> tuple[list[int], list[float]]:
        self.sparse_calls.append(text)
        return [3, 7], [0.5, 1.5]


def encoder(embedder: RecordingEmbedder) -> OnnxQueryEncoder:
    return OnnxQueryEncoder(embedder, timeout_s=2.5)  # type: ignore[arg-type]


async def test_both_vectors_are_made_from_the_same_cleaned_and_folded_text() -> None:
    embedder = RecordingEmbedder()
    # "NIKE", extra spaces and an accented e written as e plus a combining accent
    messy = "  NIKE   Café  Shoes "
    await encoder(embedder).encode_dense(messy)
    await encoder(embedder).encode_sparse(messy)
    assert embedder.dense_calls[0][0] == "nike café shoes"
    assert embedder.sparse_calls == ["nike café shoes"]


async def test_nike_and_NIKE_reach_the_model_as_the_same_text() -> None:
    embedder = RecordingEmbedder()
    await encoder(embedder).encode_dense("NIKE")
    await encoder(embedder).encode_dense("nike")
    assert embedder.dense_calls[0][0] == embedder.dense_calls[1][0]


async def test_the_timeout_reaches_the_embedder() -> None:
    embedder = RecordingEmbedder()
    await encoder(embedder).encode_dense("shoes")
    assert embedder.dense_calls == [("shoes", 2.5)]


async def test_the_sparse_terms_come_back_as_indices_and_values() -> None:
    assert await encoder(RecordingEmbedder()).encode_sparse("shoes") == SparseQuery(
        [3, 7], [0.5, 1.5]
    )


# the query side of the embedder itself, with fake models


class FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


class FakeDense:
    def __init__(self, size: int = 4, delay_s: float = 0.0) -> None:
        self.size = size
        self.delay_s = delay_s
        # the texts whose model call actually started, in the order they started
        self.started: list[str] = []

    def embed(self, documents: list[str], **kwargs: Any) -> Any:
        self.started.append(documents[0])
        time.sleep(self.delay_s)
        return iter([FakeVector([1.0] * self.size) for _ in documents])

    def query_embed(self, query: Any, **kwargs: Any) -> Any:
        raise AssertionError("a dense query must take the same path as a document")


class FakeSparse:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def query_embed(self, query: str, **kwargs: Any) -> Any:
        self.queries.append(query)

        class Item:
            indices = FakeVector([1, 5])
            values = FakeVector([0.5, 1.0])

        return iter([Item()])

    def embed(self, documents: list[str], **kwargs: Any) -> Any:
        raise AssertionError("a query must never take the document path of the sparse model")


def embedder_with(
    dense: FakeDense,
    sparse: FakeSparse,
    workers: int = 1,
    wait_observer: Callable[[float], None] | None = None,
) -> OnnxEmbedder:
    return OnnxEmbedder(
        dense,
        sparse,
        tokenizer=None,  # type: ignore[arg-type]
        limits=Limits(max_tokens=64),
        dense_dim=4,
        sub_batch=4,
        workers=workers,
        wait_observer=wait_observer,
    )


async def test_a_dense_query_takes_the_same_path_as_a_document() -> None:
    dense = FakeDense()
    embedder = embedder_with(dense, FakeSparse())
    try:
        assert await embedder.embed_query_dense("shoes", 5.0) == [1.0] * 4
    finally:
        embedder.close()
    assert dense.started == ["shoes"]


async def test_a_dense_vector_of_the_wrong_size_is_a_model_mismatch() -> None:
    embedder = embedder_with(FakeDense(size=3), FakeSparse())
    try:
        with pytest.raises(ModelMismatch, match="expected 4"):
            await embedder.embed_query_dense("shoes", 5.0)
    finally:
        embedder.close()


async def test_a_slow_dense_model_is_a_transient_error_after_the_timeout() -> None:
    embedder = embedder_with(FakeDense(delay_s=0.4), FakeSparse())
    try:
        with pytest.raises(EmbedTransient, match="took more than"):
            await embedder.embed_query_dense("shoes", 0.05)
    finally:
        embedder.close()


async def test_the_sparse_query_uses_the_query_path_of_the_bm25_model() -> None:
    sparse = FakeSparse()
    embedder = embedder_with(FakeDense(), sparse)
    try:
        assert await embedder.embed_query_sparse("shoes") == ([1, 5], [0.5, 1.0])
    finally:
        embedder.close()
    assert sparse.queries == ["shoes"]


# the queue in front of the model thread: what a timeout does to jobs that are still waiting


async def test_jobs_still_queued_when_their_request_times_out_never_run() -> None:
    dense = FakeDense(delay_s=0.4)
    embedder = embedder_with(dense, FakeSparse())
    try:
        first = asyncio.create_task(embedder.embed_query_dense("first", 5.0))
        await asyncio.sleep(0.1)  # the first job is running on the one model thread
        waiting = [
            asyncio.create_task(embedder.embed_query_dense(f"waiting {n}", 0.05)) for n in range(10)
        ]
        outcomes = await asyncio.gather(*waiting, return_exceptions=True)
        assert all(isinstance(outcome, EmbedTransient) for outcome in outcomes)
        await first
        # time for any job that was wrongly kept in the queue to start
        await asyncio.sleep(0.6)
    finally:
        embedder.close()
    assert dense.started == ["first"]


async def test_under_overload_only_the_running_jobs_are_wasted_not_the_whole_queue() -> None:
    # two model threads, jobs of 0.3 s, callers that give up after 0.45 s, twenty at once
    dense = FakeDense(delay_s=0.3)
    embedder = embedder_with(dense, FakeSparse(), workers=2)
    try:
        outcomes = await asyncio.gather(
            *(embedder.embed_query_dense(f"q{n}", 0.45) for n in range(20)),
            return_exceptions=True,
        )
        await asyncio.sleep(0.8)  # the two abandoned running jobs finish
    finally:
        embedder.close()
    answered = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
    assert len(answered) == 2
    # two answered, two started and abandoned, the other sixteen never started
    assert len(dense.started) <= 4


# the wait in front of the model thread: how long a job sat in the queue before its model call began


async def test_a_job_that_starts_at_once_reports_almost_no_wait() -> None:
    waits: list[float] = []
    embedder = embedder_with(FakeDense(), FakeSparse(), wait_observer=waits.append)
    try:
        await embedder.embed_query_dense("shoes", 5.0)
    finally:
        embedder.close()
    assert len(waits) == 1
    assert 0.0 <= waits[0] < 0.05


async def test_a_job_behind_a_running_one_reports_the_time_it_waited() -> None:
    waits: list[float] = []
    embedder = embedder_with(FakeDense(delay_s=0.15), FakeSparse(), wait_observer=waits.append)
    try:
        await asyncio.gather(
            embedder.embed_query_dense("first", 5.0), embedder.embed_query_dense("second", 5.0)
        )
    finally:
        embedder.close()
    first, second = sorted(waits)
    assert first < 0.05
    # the second job waited for the whole model call of the first
    assert 0.12 <= second < 0.5


async def test_two_model_threads_remove_the_wait_of_two_jobs() -> None:
    waits: list[float] = []
    embedder = embedder_with(
        FakeDense(delay_s=0.15), FakeSparse(), workers=2, wait_observer=waits.append
    )
    try:
        await asyncio.gather(
            embedder.embed_query_dense("first", 5.0), embedder.embed_query_dense("second", 5.0)
        )
    finally:
        embedder.close()
    assert len(waits) == 2
    assert max(waits) < 0.05


async def test_a_job_that_gave_up_before_it_started_reports_no_wait() -> None:
    waits: list[float] = []
    embedder = embedder_with(FakeDense(delay_s=0.4), FakeSparse(), wait_observer=waits.append)
    try:
        first = asyncio.create_task(embedder.embed_query_dense("first", 5.0))
        await asyncio.sleep(0.1)  # the first job is running on the one model thread
        waiting = [
            asyncio.create_task(embedder.embed_query_dense(f"waiting {n}", 0.05)) for n in range(5)
        ]
        await asyncio.gather(*waiting, return_exceptions=True)
        await first
        await asyncio.sleep(0.5)  # time for a job that was wrongly kept in the queue to start
    finally:
        embedder.close()
    # only the job that really started is reported, the dropped ones never waited for the model
    assert len(waits) == 1


async def test_the_wait_is_reported_even_when_the_model_call_then_fails() -> None:
    waits: list[float] = []
    embedder = embedder_with(FakeDense(size=3), FakeSparse(), wait_observer=waits.append)
    try:
        with pytest.raises(ModelMismatch):
            await embedder.embed_query_dense("shoes", 5.0)
    finally:
        embedder.close()
    assert len(waits) == 1


async def test_the_sparse_query_reports_no_wait_it_uses_no_model_thread() -> None:
    waits: list[float] = []
    embedder = embedder_with(FakeDense(), FakeSparse(), wait_observer=waits.append)
    try:
        await embedder.embed_query_sparse("shoes")
    finally:
        embedder.close()
    assert waits == []
