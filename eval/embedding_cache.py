"""Embeddings of chunks on disk, so a long build can be resumed and nested sets share their work.

Embedding is by far the slowest part of building an index (5 to 9 chunks per second on a laptop).
The distractor ladder builds the same corpus again and again with more products each time, a
build that fails after two hours should not start from zero, and the bake off embeds the same
validation set once per model. The key is the id of the chunk, which already holds the model
version (it is made from the document, the model version and the text), so a cache file can never
give a vector of another model or of another text.

The vectors are kept exactly as the model returned them, so a build from the cache writes the very
same points as a build that embeds: a result must never depend on whether the cache was warm. The
dense vector is a float32 (that is what bge-small returns, so 4 bytes per number lose nothing and
8 bytes would only double the file), the sparse values are float64 (what the BM25 model returns)
and the sparse indices are 64 bit integers. A dense value that does not survive float32 is refused
when it is put, so a later model that returns float64 can never be rounded without anybody noticing.
One sqlite file per model version, nothing to install.
"""

import sqlite3
from array import array
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Literal

from libs.embedding.model import Embedding

# sqlite refuses more than 999 values in one query on some builds, a build asks about thousands
_ASK_AT_A_TIME = 500


def embedding_cache_path(model_version: str) -> Path:
    """The cache file of a model version: the average length of BM25 is part of the version, so
    sparse vectors made with another average length are in another file and never served."""
    return Path("data") / "cache" / f"embeddings-{model_version}.sqlite"


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            "create table if not exists embeddings ("
            "chunk_id text primary key, dense blob not null, "
            "sparse_indices blob not null, sparse_values blob not null)"
        )
        self._connection.commit()

    def get_many(self, chunk_ids: Sequence[str]) -> dict[str, Embedding]:
        """The embeddings that are in the cache, keyed by chunk id; an unknown id is left out."""
        found: dict[str, Embedding] = {}
        for start in range(0, len(chunk_ids), _ASK_AT_A_TIME):
            piece = list(chunk_ids[start : start + _ASK_AT_A_TIME])
            marks = ",".join("?" * len(piece))
            rows = self._connection.execute(
                "select chunk_id, dense, sparse_indices, sparse_values from embeddings "  # noqa: S608 - only question marks are put into the text
                f"where chunk_id in ({marks})",
                piece,
            )
            for chunk_id, dense, indices, values in rows:
                found[chunk_id] = Embedding(
                    dense=_floats(dense, "f"),
                    sparse_indices=_integers(indices),
                    sparse_values=_floats(values, "d"),
                )
        return found

    def put_many(self, embeddings: Mapping[str, Embedding]) -> None:
        """Store the embeddings, one transaction; an id that is there is replaced."""
        # every row is checked before the first one is written, a refused batch leaves no trace
        rows = [
            (
                chunk_id,
                _dense_bytes(chunk_id, embedding.dense),
                array("q", embedding.sparse_indices).tobytes(),
                array("d", embedding.sparse_values).tobytes(),
            )
            for chunk_id, embedding in embeddings.items()
        ]
        with self._connection:
            self._connection.executemany(
                "insert or replace into embeddings values (?, ?, ?, ?)", rows
            )

    def count(self) -> int:
        row = self._connection.execute("select count(*) from embeddings").fetchone()
        return int(row[0])

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _dense_bytes(chunk_id: str, dense: Sequence[float]) -> bytes:
    # python turns a value too large for float32 into inf, which fails the comparison below too
    stored = array("f", dense)
    if stored.tolist() != list(dense):
        raise ValueError(
            f"dense vector of {chunk_id} does not survive a float32 round trip, "
            "the cache would round it: the model no longer returns float32"
        )
    return stored.tobytes()


def _floats(blob: bytes, code: Literal["f", "d"]) -> list[float]:
    numbers: array[float] = array(code)
    numbers.frombytes(blob)
    return numbers.tolist()


def _integers(blob: bytes) -> list[int]:
    numbers = array("q")
    numbers.frombytes(blob)
    return numbers.tolist()
