"""Application-facing measurement loading seam.

The format-specific readers currently remain in ``app.logic`` for compatibility;
this facade is the first Phase-E extraction point and owns loader orchestration
and replacement policy.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Generic, TypeVar

LoadedT = TypeVar("LoadedT")
ShotT = TypeVar("ShotT")


class MeasurementLoader(Generic[LoadedT, ShotT]):
    """Load measurement files without coupling callers to file formats."""

    def __init__(
        self,
        load_file: Callable[[str | Path, int | None], tuple[LoadedT, ...]],
        list_shots: Callable[[str | Path], tuple[ShotT, ...]],
        load_crisp_file: Callable[[str | Path, int | None], tuple[LoadedT, ...]] | None = None,
        load_ir_file: Callable[[str | Path, int | None], tuple[LoadedT, ...]] | None = None,
    ) -> None:
        self._load_file = load_file
        self._list_shots = list_shots
        self._load_crisp_file = load_crisp_file
        self._load_ir_file = load_ir_file

    def load(
        self, path: str | Path, *, h5_shot_index: int | None = None
    ) -> tuple[LoadedT, ...]:
        return self._load_file(path, h5_shot_index)

    def load_crisp(
        self, path: str | Path, *, h5_shot_index: int | None = None
    ) -> tuple[LoadedT, ...]:
        loaded = (
            self._load_crisp_file(path, h5_shot_index)
            if self._load_crisp_file is not None
            else self.load(path, h5_shot_index=h5_shot_index)
        )
        crisp = tuple(item for item in loaded if self._kind(item) == "crisp")
        if len(crisp) != len(loaded) or not crisp:
            raise ValueError("file does not contain a valid CRISP dataset")
        return crisp

    def load_ir(
        self, path: str | Path, *, h5_shot_index: int | None = None
    ) -> tuple[LoadedT, ...]:
        loaded = (
            self._load_ir_file(path, h5_shot_index)
            if self._load_ir_file is not None
            else self.load(path, h5_shot_index=h5_shot_index)
        )
        infrared = tuple(item for item in loaded if self._kind(item) == "ocean_nir")
        if len(infrared) != len(loaded) or not infrared:
            raise ValueError("file does not contain a valid Ocean/NIR IR dataset")
        return infrared

    def list_shots(self, path: str | Path) -> tuple[ShotT, ...]:
        return self._list_shots(path)

    @staticmethod
    def _kind(item: LoadedT) -> str:
        kind = getattr(item, "kind", None)
        value = getattr(kind, "value", None)
        return str(kind if value is None else value)

    @staticmethod
    def replace_ocean(
        current: tuple[LoadedT, ...],
        loaded: tuple[LoadedT, ...],
        *,
        kind: Callable[[LoadedT], str],
    ) -> tuple[LoadedT, ...]:
        ocean = tuple(item for item in loaded if kind(item) == "ocean_nir")
        if len(ocean) != len(loaded) or not ocean:
            raise ValueError("measurement file does not contain only Ocean/NIR data")
        preserved = tuple(item for item in current if kind(item) != "ocean_nir")
        return (*preserved, *ocean)
