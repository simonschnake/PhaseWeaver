# phase_weaver/core/reconstruction.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .base import (
    DCPhysicalRFFT,
    FormFactor,
    Grid,
    Profile,
    Transform,
)
from .constraints import (
    CenterFirstMoment,
    ClampMagnitude,
    EnforceDCOne,
    FrequencyConstraint,
    FrequencyConstraints,
    NonNegativity,
    NormalizeArea,
    TimeConstraint,
    TimeConstraints,
)

# =============================================================================
# Phase initialization
# =============================================================================


class PhaseInitializer(ABC):
    """Given a magnitude, estimate an initial phase for bins [0..Nyquist]."""

    name: str

    @abstractmethod
    def init_phase(
        self, mag_pos: np.ndarray, grid: Grid, eps_mag: float, **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError


@dataclass(slots=True)
class ZeroPhase(PhaseInitializer):
    name: str = "zero"

    def init_phase(
        self, mag_pos: np.ndarray, grid: Grid, *, eps_mag: float
    ) -> np.ndarray:
        return np.zeros_like(mag_pos, dtype=float)


@dataclass(slots=True)
class RandomPhase(PhaseInitializer):
    seed: int | None = None
    name: str = "random"

    def init_phase(
        self, mag_pos: np.ndarray, grid: Grid, *, eps_mag: float
    ) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return np.unwrap(rng.uniform(0.0, 2.0 * np.pi, size=mag_pos.shape))


class MinimumPhaseCepstrum(PhaseInitializer):
    """
    Minimum-phase estimate from magnitude using real cepstrum.

    Returns unwrapped phase for bins [0..Nyquist], length N/2+1.
    """

    name: str = "minphase"

    def init_phase(
        self, mag_pos: np.ndarray, grid: Grid, *, eps_mag: float
    ) -> np.ndarray:
        mag_pos = np.asarray(mag_pos, dtype=float)
        N = grid.N
        expected = (N // 2 + 1,)
        if mag_pos.shape != expected:
            raise ValueError(f"mag_pos must have shape {expected}, got {mag_pos.shape}")

        logmag_pos = np.log(np.maximum(mag_pos, eps_mag))

        logmag_full = np.empty(N, dtype=float)
        logmag_full[: N // 2 + 1] = logmag_pos
        # mirror interior for even-N rfft symmetry
        logmag_full[N // 2 + 1 :] = logmag_pos[1:-1][::-1]

        c = np.fft.ifft(logmag_full).real

        cmin = np.zeros_like(c)
        cmin[0] = c[0]
        cmin[1 : N // 2] = 2.0 * c[1 : N // 2]
        cmin[N // 2] = c[N // 2]  # Nyquist term for even N

        spec = np.exp(np.fft.fft(cmin))
        phase = np.angle(spec[: N // 2 + 1])
        return np.unwrap(phase)


# =============================================================================
# Stopping criteria
# =============================================================================


class StopCriterion(ABC):
    """Decide when to stop iterative reconstruction."""

    name: str

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def update(self, prof: Profile, ff: FormFactor) -> bool: ...

class StopCriteria(StopCriterion):
    """
    Combine multiple StopCriterion objects with "any" or "all" logic.
    """

    name: str = "combined"

    def __init__(self, *args: StopCriterion, logic: Literal["any", "all"] = "any"):
        self.criteria = args
        if logic not in ("any", "all"):
            raise ValueError("logic must be 'any' or 'all'")
        self.logic = logic

    def reset(self) -> None:
        for c in self.criteria:
            c.reset()

    def update(self, prof: Profile, ff: FormFactor) -> bool:
        for c in self.criteria:
            if c.update(prof, ff):
                if self.logic == "any":
                    return True
            else:
                if self.logic == "all":
                    return False
        return self.logic == "all"


@dataclass(slots=True)
class MaxIter(StopCriterion):
    n_iter: int
    name: str = "max_iter"

    def __post_init__(self) -> None:
        self._counter = 0

    def reset(self) -> None:
        self._counter = 0

    def update(self, prof: Profile, ff: FormFactor) -> bool:
        self._counter += 1
        return self._counter >= self.n_iter


@dataclass(slots=True)
class PhaseStoppedChanging(StopCriterion):
    """
    Stop if the phase hasn't changed more than tol for `patience` iterations.
    """

    tol: float = 1e-8
    patience: int = 5
    name: str = "phase_stopped_changing"

    def __post_init__(self) -> None:
        self._count = 0
        self._last_phase: np.ndarray | None = None

    def reset(self) -> None:
        self._count = 0
        self._last_phase = None

    def update(self, prof: Profile, ff: FormFactor) -> bool:
        phase = ff.phase
        mse = self.phase_diff_mse(phase)
        if mse < self.tol:
            self._count += 1
        else:
            self._count = 0
        self._last_phase = phase
        return self._count >= self.patience

    def phase_diff_mse(self, phase: np.ndarray) -> float:
        if self._last_phase is None:
            return np.inf
        return float(np.mean((phase - self._last_phase) ** 2))


# =============================================================================
# Reconstruction framework
# =============================================================================


class ReconstructionAlgorithm(ABC):
    """
    Abstract base for reconstruction algorithms.
    """

    name: str

    @abstractmethod
    def run(
        self,
        mag: np.ndarray,
        grid: Grid,
        charge_C: float,
        **kwargs,
    ) -> tuple[Profile, FormFactor]: ...


# =============================================================================
# Gerchberg–Saxton style: magnitude-only -> phase retrieval
# =============================================================================


@dataclass(slots=True)
class GSMagnitudeOnly(ReconstructionAlgorithm):
    """
    GS-style algorithm that:
      1) estimates an initial phase from magnitude
      2) repeatedly:
         - inverse transform to time
         - apply time constraints
         - forward transform to frequency
         - replace magnitude with measured magnitude
      3) returns final profile + phases/metrics

    This reproduces the behavior of your current `reconstruct_current_from_mag`.
    """

    name: str = "gs_magnitude_only"

    transform: Transform = DCPhysicalRFFT(unwrap_phase=True, dc_normalize=False)
    phase_init: PhaseInitializer = MinimumPhaseCepstrum()
    time_constraints: TimeConstraint = TimeConstraints(
        NonNegativity(), NormalizeArea(), CenterFirstMoment()
    )
    frequency_constraints: FrequencyConstraint = FrequencyConstraints(
        ClampMagnitude(eps=1e-6), EnforceDCOne()
    )
    stop: StopCriterion = StopCriteria(
        MaxIter(50), PhaseStoppedChanging(tol=1e-8, patience=5), logic="any"
    )
    eps_mag: float = 1e-30

    def run(
        self, mag: np.ndarray, grid: Grid, charge_C: float
    ) -> tuple[Profile, FormFactor]:
        mag = np.asarray(mag, dtype=float)
        expected = (grid.N // 2 + 1,)
        if mag.shape != expected:
            raise ValueError(f"mag must have shape {expected}, got {mag.shape}")

        # Prepare measurement magnitude (clamp + DC normalize etc.) WITHOUT mutating input
        ff0 = FormFactor(
            grid=grid, mag=mag.copy(), phase=np.zeros_like(mag), eps=self.eps_mag
        )
        self.frequency_constraints.apply(ff0)
        mag_meas = ff0.mag.copy()

        # init phase from processed magnitude
        phase = self.phase_init.init_phase(mag_meas, grid, eps_mag=self.eps_mag)

        # initial complex FF: measured magnitude + init phase
        ff = FormFactor(grid=grid, mag=mag_meas.copy(), phase=phase, eps=self.eps_mag)

        # initial time profile + constraints
        prof = Profile.from_form_factor(ff, transform=self.transform, charge=None)
        self.time_constraints.apply(prof)

        self.stop.reset()

        while not self.stop.update(prof, ff):
            # to frequency domain
            ff = prof.to_form_factor(transform=self.transform)

            # magnitude projection
            ff.mag = mag_meas.copy()
            self.frequency_constraints.apply(ff)

            # back to time domain + constraints
            prof = ff.to_profile(transform=self.transform, charge=None)
            self.time_constraints.apply(prof)

        # attach charge at the end (or inside Profile if you prefer)
        prof.charge = (
            charge_C  # only if Profile allows mutation; otherwise rebuild with charge
        )

        return prof, ff
