import asyncio
import functools
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from fastembed import SparseTextEmbedding, TextEmbedding

from libs.embedding.settings import ModelSettings
from libs.resilience.retry import PermanentError, TransientError


class Tokenizer(Protocol):
    def spans(self, text: str) -> list[tuple[int, int]]:
        """Character span (start, end) of every content token; special tokens are not included."""
        ...


@dataclass(frozen=True)
class Limits:
    max_tokens: int
    margin: int = 16

    @property
    def budget(self) -> int:
        return self.max_tokens - self.margin


class EmbedPermanent(PermanentError):
    """One text cannot be embedded, retrying will not help, index says which one."""

    def __init__(self, index: int, cause: Exception) -> None:
        super().__init__(f"text {index} cannot be embedded: {cause!r}")
        self.index = index


# a transient error is never a reason for the dead letter queue: after its retries the partition
# stalls and alerts, moving a perfectly good message to the dead letter queue because the machine
# was briefly short of memory or time is an anti pattern, the message itself is fine
class EmbedTransient(TransientError):
    """The machine ran short of memory or time, the same texts may work a moment later."""


class ModelMismatch(RuntimeError):
    """The model returns vectors of another size than the collection holds, a setup mistake."""


@dataclass(frozen=True)
class Embedding:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class Embedder(Protocol):
    """What the worker needs, the real model and the test stub both fit."""

    tokenizer: Tokenizer
    limits: Limits

    async def embed(self, texts: Sequence[str], timeout_s: float) -> list[Embedding]: ...

    async def verify(self) -> None: ...

    def close(self) -> None: ...


class TextModel(Protocol):
    """The part of a fastembed model we use, so a fake can stand in for it in unit tests."""

    def embed(self, documents: list[str], **kwargs: Any) -> Iterable[Any]: ...

    def query_embed(self, query: str | list[str], **kwargs: Any) -> Iterable[Any]: ...


class ModelTokenizer:
    """The model's own tokenizer, asked for the character span of every content token."""

    def __init__(self, hf_tokenizer: Any) -> None:
        # the model's tokenizer cuts every text at its limit, which would hide how much overflows
        # and the composer has to see all of it to split it, so this is an untruncated copy, made
        # through the tokenizer's own class, and the model keeps its own settings untouched
        copy = type(hf_tokenizer).from_str(hf_tokenizer.to_str())
        copy.no_truncation()
        copy.no_padding()
        self._tokenizer = copy

    def spans(self, text: str) -> list[tuple[int, int]]:
        # no special tokens, the composer keeps a margin for them
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        return [(start, end) for start, end in encoding.offsets]


def _is_resource_pressure(error: Exception) -> bool:
    message = str(error).lower()
    return isinstance(error, MemoryError) or "allocate" in message or "out of memory" in message


# an ordinary embedder: texts in, vectors out
# it only differs in how it runs, the trained model is an onnx file and onnx runtime executes
# that file on the cpu, without installing pytorch
class OnnxEmbedder:
    def __init__(
        self,
        dense: TextModel,
        sparse: TextModel,
        tokenizer: Tokenizer,
        limits: Limits,
        dense_dim: int,
        sub_batch: int,
        workers: int = 1,
        wait_observer: Callable[[float], None] | None = None,
    ) -> None:
        self._dense = dense
        # told, from the model thread, how long a dense query job waited in the queue in front of
        # it (the query service turns that into a histogram), nothing happens when it is None
        self._wait_observer = wait_observer
        self._sparse = sparse
        self.tokenizer = tokenizer
        self.limits = limits
        self._dense_dim = dense_dim
        self._sub_batch = sub_batch
        # model calls run on this many worker threads (the worker that indexes uses one, the
        # query service several), and the onnx runtime inside the model has its own thread
        # count, so the cores are shared out by two plain numbers: workers times threads
        self._onnx_executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="embed")

    async def embed(self, texts: Sequence[str], timeout_s: float) -> list[Embedding]:
        if not texts:
            return []
        deadline = time.monotonic() + timeout_s
        # sort on the number of tokens per text, shortest first
        # a text here is a whole chunk of many tokens, titles and bullets and descriptions, from
        # about ten tokens up to the model limit, not a single token
        # the model pads every text in a batch to the longest one in that batch, so a ten token
        # text and a four hundred token text placed side by side are both padded to four hundred,
        # and the short one carries about three hundred and ninety positions of pure padding
        # in ascending order texts of similar length land in the same batch and little is wasted
        order = sorted(range(len(texts)), key=lambda i: len(self.tokenizer.spans(texts[i])))
        results: dict[int, Embedding] = {}
        loop = asyncio.get_running_loop()
        for start in range(0, len(order), self._sub_batch):
            # an onnx run cannot be interrupted, so the deadline is checked between sub batches
            if time.monotonic() >= deadline:
                raise EmbedTransient(f"deadline of {timeout_s}s passed with texts left")
            indices = order[start : start + self._sub_batch]
            batch = [texts[i] for i in indices]
            # off the event loop, so the loop keeps serving while the model computes
            done = await loop.run_in_executor(self._onnx_executor, self._run, batch, indices)
            results.update(zip(indices, done, strict=True))
        # back in the order the caller sent them, a missing slot would be a bug and raise
        return [results[index] for index in range(len(texts))]

    @property
    def queued_jobs(self) -> int:
        """Dense query jobs handed to the pool that no model thread has taken yet.

        This is the length of the executor's own queue. A job whose caller gave up stays in it
        until a thread takes it and finds it cancelled, which takes no time, so the number is
        right to within the jobs that are about to be dropped.
        """
        return int(self._onnx_executor._work_queue.qsize())

    async def embed_query_dense(self, text: str, timeout_s: float) -> list[float]:
        """The dense vector of one query, made the way a document is made."""
        loop = asyncio.get_running_loop()
        # the clock starts here, when the job is handed to the pool, and stops on the model thread
        run = functools.partial(self._query_dense, text, time.perf_counter())
        try:
            # the model call itself cannot be interrupted, the timeout stops the waiting, a job
            # that is still queued when its caller gives up is dropped, and the admission limit
            # of the service bounds how many can be queued at all
            return await asyncio.wait_for(
                loop.run_in_executor(self._onnx_executor, run), timeout=timeout_s
            )
        except TimeoutError as error:
            raise EmbedTransient(f"the dense query encoding took more than {timeout_s}s") from error

    async def embed_query_sparse(self, text: str) -> tuple[list[int], list[float]]:
        """The bm25 terms of one query: indices and weights. Cheap, so it needs no model thread."""
        return await asyncio.to_thread(self._query_sparse, text)

    def _query_dense(self, text: str, submitted: float) -> list[float]:
        # the job just left the queue: the time since it was handed over is the wait, which is what
        # tells a saturated encoder from a slow one (a job dropped while queued never gets here)
        if self._wait_observer is not None:
            self._wait_observer(time.perf_counter() - submitted)
        vector: list[float] = next(iter(self._dense.embed([text]))).tolist()
        if len(vector) != self._dense_dim:
            raise ModelMismatch(
                f"the model returns {len(vector)} floats, expected {self._dense_dim}"
            )
        return vector

    def _query_sparse(self, text: str) -> tuple[list[int], list[float]]:
        item = next(iter(self._sparse.query_embed(text)))
        return item.indices.tolist(), item.values.tolist()

    async def verify(self) -> None:
        # at startup: a model of the wrong size must stop the process before it reads a message
        await self.embed(["probe"], timeout_s=60.0)

    def close(self) -> None:
        self._onnx_executor.shutdown(wait=True)

    def _run(self, batch: list[str], indices: list[int]) -> list[Embedding]:
        try:
            return self._embed_many(batch)
        except ModelMismatch:
            raise
        except Exception as error:
            if _is_resource_pressure(error):
                raise EmbedTransient(str(error)) from error
            if len(batch) == 1:
                raise EmbedPermanent(indices[0], error) from error
        # the batch failed for a reason that is not memory, so find the text that causes it
        # a lone surrogate unicode character is one case, the tokenizer rejects that text with
        # a TypeError and the whole batch fails with it, so the texts are tried one by one
        return [self._embed_one(text, index) for text, index in zip(batch, indices, strict=True)]

    def _embed_one(self, text: str, index: int) -> Embedding:
        try:
            return self._embed_many([text])[0]
        except Exception as error:
            if _is_resource_pressure(error):
                raise EmbedTransient(str(error)) from error
            raise EmbedPermanent(index, error) from error

    def _embed_many(self, batch: list[str]) -> list[Embedding]:
        dense = [vector.tolist() for vector in self._dense.embed(batch)]
        sparse = list(self._sparse.embed(batch))
        for vector in dense:
            if len(vector) != self._dense_dim:
                raise ModelMismatch(
                    f"the model returns {len(vector)} floats, expected {self._dense_dim}"
                )
        return [
            Embedding(vector, item.indices.tolist(), item.values.tolist())
            for vector, item in zip(dense, sparse, strict=True)
        ]


def build_onnx_embedder(
    settings: ModelSettings,
    workers: int = 1,
    wait_observer: Callable[[float], None] | None = None,
) -> OnnxEmbedder:
    dense = TextEmbedding(settings.dense_model, threads=settings.embed_threads)
    sparse = SparseTextEmbedding("Qdrant/bm25", avg_len=settings.bm25_avg_len)
    # fastembed types its model as a base class that has no tokenizer, the onnx one has
    hf_tokenizer = dense.model.tokenizer  # type: ignore[attr-defined]
    return OnnxEmbedder(
        dense=dense,
        sparse=sparse,
        tokenizer=ModelTokenizer(hf_tokenizer),
        # the model cuts texts at this many tokens and says nothing, the composer must stay under it
        limits=Limits(max_tokens=hf_tokenizer.truncation["max_length"]),
        dense_dim=settings.dense_dim,
        sub_batch=settings.embed_sub_batch,
        workers=workers,
        wait_observer=wait_observer,
    )


# note: what the tokenizer spans are, with one input and its output
#
# input text: "wireless mouse"
#
# the model's tokenizer cuts the text into tokens and gives every token a number
#   tokens  : [CLS]   wireless   mouse    [SEP]
#   ids     : 101     9949       8000     102
#   offsets : (0, 0)  (0, 8)     (9, 14)  (0, 0)
# [CLS] and [SEP] are markers the model adds, they are not part of the text, so their offsets
# are empty, the offset of a real token says which characters of the original text it came from
#
# a span is that offset pair, without the markers
#   spans   : [(0, 8), (9, 14)]
#   text[0:8] is "wireless" and text[9:14] is "mouse"
#
# the composer needs the spans for two things
#   counting: len(spans) is the size of a chunk in the model's tokens, 2 here
#   cutting: a long text is split at a token boundary, the composer cuts at the end of a span
#   and never inside a word, for example right after character 14 here, so a cut piece is
#   always the characters of whole tokens
#
# a long or unknown word is split into several tokens, ultralightweightergonomical becomes
# ultra ##light ##weight ##er ##gon ##omic ##al, the ## marks a piece that continues the one
# before it, and the spans of those pieces are consecutive character ranges (0, 5) (5, 10) ...
#
# the model cuts any text at 512 tokens and says nothing, so the tokenizer here is an untruncated
# copy: the composer sees every span and splits the overflow itself
#
# a lone surrogate unicode character makes the tokenizer reject the whole text with a TypeError,
# see the one by one retry in _run
