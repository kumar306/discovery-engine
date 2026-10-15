"""Tools to kill the worker at an exact point: a call counter, and two partial failures.

A crash is simulated by raising SimulatedCrash instead of letting a call happen, or by letting
only part of a call happen and then raising. Nothing is mocked: the stores underneath are real,
so what a crash leaves behind is what a real crash would leave behind.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from services.embed.qdrant import ChunkIndex, IndexedChunk, LiveChunk, publish_operations


class SimulatedCrash(BaseException):
    """The process died here. A BaseException, so no error handling can swallow it."""


@dataclass
class CallTrace:
    """Every call the pipeline made, in order, shared by all its collaborators."""

    crash_before: int | None = None
    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def wrap(self, inner: Any) -> "_Traced":
        return _Traced(inner, self)

    def tick(self, name: str, args: tuple[Any, ...]) -> None:
        self.calls.append((name, args))
        # the call is counted first, so call number n is the n th thing the pipeline tried to do
        if len(self.calls) == self.crash_before:
            raise SimulatedCrash(f"the process died before call {len(self.calls)}, {name}")


class _Traced:
    def __init__(self, inner: Any, trace: CallTrace) -> None:
        self._inner = inner
        self._trace = trace

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._inner, name)
        if not callable(target):
            return target

        async def call(*args: Any, **kwargs: Any) -> Any:
            self._trace.tick(name, args)
            return await target(*args, **kwargs)

        return call


class PartialStageIndex(ChunkIndex):
    """Stages only the first points, then the process dies in the middle of the staging."""

    def __init__(self, inner: ChunkIndex, points: int) -> None:
        super().__init__(inner._qdrant_client, inner._qdrant_collection)
        self._points = points

    async def stage(self, chunks: Sequence[IndexedChunk]) -> None:
        await super().stage(list(chunks)[: self._points])
        raise SimulatedCrash(f"the process died after staging {self._points} point(s)")


class PartialPublishIndex(ChunkIndex):
    """Applies only the first payload updates of a publish, then the process dies."""

    def __init__(self, inner: ChunkIndex, operations: int) -> None:
        super().__init__(inner._qdrant_client, inner._qdrant_collection)
        self._operations = operations

    async def publish(
        self, version: int, live: Sequence[LiveChunk], superseded: Sequence[UUID]
    ) -> None:
        operations = publish_operations(version, live, superseded)[: self._operations]
        await self._qdrant_client.batch_update_points(
            self._qdrant_collection, update_operations=operations, wait=True
        )
        raise SimulatedCrash(f"the process died after {self._operations} publish update(s)")
