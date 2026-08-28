"""Declarative dispatch for reconstruction algorithms."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

from .policy import RECONSTRUCTION_ALGORITHM

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")
SeedT = TypeVar("SeedT")


@dataclass(frozen=True, slots=True)
class ReconstructionPipeline(Generic[RequestT, ResultT]):
    """Dispatch reconstruction requests by typed algorithm policy."""

    handlers: Mapping[
        RECONSTRUCTION_ALGORITHM, Callable[[RequestT], ResultT]
    ]

    def run(
        self, algorithm: RECONSTRUCTION_ALGORITHM, request: RequestT
    ) -> ResultT:
        try:
            handler = self.handlers[algorithm]
        except KeyError as exc:
            raise ValueError(
                f"No reconstruction handler registered for {algorithm.value!r}"
            ) from exc
        return handler(request)


@dataclass(frozen=True, slots=True)
class CrispThenIrSeed(Generic[RequestT, SeedT, ResultT]):
    """Compose a CRISP reconstruction with an optional IR-seeded extension."""

    crisp: Callable[[RequestT], SeedT]
    extension: Callable[[RequestT, SeedT], ResultT]

    def run(self, request: RequestT) -> ResultT:
        seed = self.crisp(request)
        return self.extension(request, seed)
