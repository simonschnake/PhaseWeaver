# phase_weaver/core/reconstruction.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    NonNegativity,
    NormalizeArea,
    TimeConstraint,
)

# =============================================================================
# Formfactor initialization
# =============================================================================


class FormFactorInitializer(ABC):
    """Given a Grid, return an initial FormFactor with mag+phase for bins [0..Nyquist]."""

    name: str

    @abstractmethod
    def __call__(self, grid: Grid, **kwargs) -> FormFactor: ...


class PhaseInitializer(ABC):
    """Given a Grid, return an initial phase"""

    name: str

    @abstractmethod
    def initialize_phase(self, grid: Grid, **kwargs) -> np.ndarray: ...

    def __call__(self, grid: Grid, **kwargs) -> FormFactor:
        mag = kwargs.get(
            "mag", None
        )  # allow passing mag if needed, but default to None
        phase = self.initialize_phase(grid, **kwargs)
        return FormFactor(grid=grid, mag=mag, phase=phase, eps=kwargs.get("eps_mag", np.finfo(float).eps))


class MagnitudeInitializer(ABC):
    """Given a Grid, return an initial magnitude"""

    name: str

    @abstractmethod
    def initialize_magnitude(self, grid: Grid, **kwargs) -> np.ndarray: ...

    def __call__(self, grid: Grid, **kwargs) -> FormFactor:
        mag = self.initialize_magnitude(grid, **kwargs)
        phase = kwargs.get(
            "phase", None
        )  # allow passing phase if needed, but default to None
        return FormFactor(grid=grid, mag=mag, phase=phase)


class GaussianInitializer(FormFactorInitializer):
    """
    Return the FormFactor corresponding to a Gaussian profile in time.

    The time-domain function is

        g(t) = (1 / (sigma * sqrt(2*pi))) * exp(-t**2 / (2*sigma**2))

    where:
        - t is time in seconds
        - sigma is the standard deviation in time

    Using the Fourier transform convention

        g_hat(f) = integral from -inf to inf of g(t) * exp(-i*2*pi*f*t) dt

    with f in Hz, the Fourier transform is

        g_hat(f) = exp(-2*pi**2 * sigma**2 * f**2)

    So a normalized zero-mean Gaussian in time remains a Gaussian in frequency.
    """

    name: str = "gaussian"

    def __call__(self, grid: Grid, **kwargs) -> FormFactor:
        try:
            sigma = kwargs["sigma"]
        except KeyError:
            raise KeyError(
                "GaussianInitializer requires 'sigma' keyword argument (in seconds)"
            )

        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        mag = np.exp(-2 * np.pi**2 * sigma**2 * grid.f_pos**2)
        phase = np.zeros_like(mag)

        return FormFactor(grid=grid, mag=mag, phase=phase, frequency_constraint=ClampMagnitude(eps=1e-6) + EnforceDCOne())


class ZeroPhase(PhaseInitializer):
    name: str = "zero_phase"

    def initialize_phase(self, grid: Grid, **kwargs) -> np.ndarray:
        return np.zeros_like(grid.f_pos, dtype=float)


class RandomPhase(PhaseInitializer):
    name: str = "random"

    def initialize_phase(self, grid: Grid, **kwargs) -> np.ndarray:
        rng = np.random.default_rng(kwargs.get("seed", None))
        return np.unwrap(rng.uniform(0.0, 2.0 * np.pi, size=grid.f_pos.shape))


class MinimumPhaseCepstrum(PhaseInitializer):
    """
    Minimum-phase estimate from magnitude using real cepstrum.

    Returns unwrapped phase for bins [0..Nyquist], length N/2+1.
    """

    name: str = "minphase"

    def initialize_phase(self, grid: Grid, **kwargs) -> np.ndarray:
        mag_pos = kwargs.get("mag", None)
        if mag_pos is None:
            raise ValueError("MinimumPhaseCepstrum requires 'mag' keyword argument")
        eps_mag = kwargs.get("eps_mag", np.finfo(float).eps)

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


class PredefinedPhase(PhaseInitializer):
    """
    Use a predefined phase array, ignoring the input magnitude.
    """

    def __init__(self, phase: np.ndarray):
        self.phase = np.asarray(phase, dtype=float)
        self.name = "predefined"

    def initialize_phase(self, grid: Grid, **kwargs) -> np.ndarray:
        expected = (grid.N // 2 + 1,)
        if self.phase.shape != expected:
            raise ValueError(
                f"predefined phase must have shape {expected}, got {self.phase.shape}"
            )
        return np.unwrap(self.phase)


# =============================================================================
# Stopping Criteria
# =============================================================================


class StopCriterion(ABC):
    """Decide when to stop iterative reconstruction."""

    name: str
    stop: bool = False
    direct_stop: bool = False

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def update(
        self,
        last_prof: Profile | None = None,
        pre_prof: Profile | None = None,
        post_prof: Profile | None = None,
        last_ff: FormFactor | None = None,
        pre_ff: FormFactor | None = None,
        post_ff: FormFactor | None = None,
    ) -> None: ...

    def __add__(self, other) -> StopCriteria:
        return StopCriteria(self, other)

    def __call__(self) -> bool:
        return self.stop or self.direct_stop


class StopCriteria(StopCriterion):
    """
    Combine multiple StopCriterion objects with "any" or "all" logic.
    """

    name: str = "combined"

    def __init__(self, *args: StopCriterion) -> None:
        self.criteria = args

    def reset(self) -> None:
        for c in self.criteria:
            c.reset()

    def update(
        self,
        last_prof: Profile | None = None,
        pre_prof: Profile | None = None,
        post_prof: Profile | None = None,
        last_ff: FormFactor | None = None,
        pre_ff: FormFactor | None = None,
        post_ff: FormFactor | None = None,
    ) -> None:
        for c in self.criteria:
            c.update(
                last_prof=last_prof,
                pre_prof=pre_prof,
                post_prof=post_prof,
                last_ff=last_ff,
                pre_ff=pre_ff,
                post_ff=post_ff,
            )

    def __add__(self, other) -> StopCriteria:
        if not isinstance(other, StopCriterion):
            raise ValueError(
                f"Can only combine with another StopCriterion, got {type(other)}"
            )
        if isinstance(other, StopCriteria):
            return StopCriteria(*self.criteria, *other.criteria)
        return StopCriteria(*self.criteria, other)

    def __call__(self) -> bool:
        all_stop = all(c.stop for c in self.criteria)
        any_direct_stop = any(c.direct_stop for c in self.criteria)
        return all_stop or any_direct_stop


@dataclass
class MinIter(StopCriterion):
    n_iter: int
    name: str = "min_iter"
    direct_stop: bool = False

    def __post_init__(self) -> None:
        self._counter = 0

    def reset(self) -> None:
        self._counter = 0

    @property
    def stop(self) -> bool:
        return self._counter >= self.n_iter

    def update(
        self,
        last_prof: Profile | None = None,
        pre_prof: Profile | None = None,
        post_prof: Profile | None = None,
        last_ff: FormFactor | None = None,
        pre_ff: FormFactor | None = None,
        post_ff: FormFactor | None = None,
    ) -> None:
        self._counter += 1


@dataclass
class MaxIter(StopCriterion):
    n_iter: int
    name: str = "max_iter"

    def __post_init__(self) -> None:
        self._counter = 0

    def reset(self) -> None:
        self._counter = 0

    @property
    def stop(self) -> bool:
        return self.direct_stop

    @property
    def direct_stop(self) -> bool:
        return self._counter >= self.n_iter

    def update(
        self,
        last_prof: Profile | None = None,
        pre_prof: Profile | None = None,
        post_prof: Profile | None = None,
        last_ff: FormFactor | None = None,
        pre_ff: FormFactor | None = None,
        post_ff: FormFactor | None = None,
    ) -> None:
        self._counter += 1


@dataclass(slots=True)
class PhaseStoppedChanging(StopCriterion):
    """
    Stop if the phase hasn't changed more than tol for `patience` iterations.
    """

    tol: float = 1e-8
    patience: int = 5
    name: str = "phase_stopped_changing"

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._count = 0
        self._mse: float = np.inf

    @property
    def stop(self) -> bool:
        if self._mse < self.tol:
            self._count += 1
            return self._count >= self.patience
        return False

    def update(
        self,
        last_prof: Profile | None = None,
        pre_prof: Profile | None = None,
        post_prof: Profile | None = None,
        last_ff: FormFactor | None = None,
        pre_ff: FormFactor | None = None,
        post_ff: FormFactor | None = None,
    ) -> None:
        if last_ff is not None and post_ff is not None:
            self._mse = float(np.mean((post_ff.phase - last_ff.phase)) ** 2)


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
    time_constraints: TimeConstraint = (
        NonNegativity() + NormalizeArea() + CenterFirstMoment()
    )

    frequency_constraints: FrequencyConstraint = (
        ClampMagnitude(eps=1e-6) + EnforceDCOne()
    )

    stop: StopCriterion = (
        MaxIter(1_000) + PhaseStoppedChanging(tol=1e-8, patience=5) + MinIter(10)
    )

    eps_mag: float = 1e-30

    def run(
        self, mag: np.ndarray, grid: Grid, charge_C: float
    ) -> tuple[Profile, FormFactor]:
        expected = (grid.N // 2 + 1,)
        if mag.shape != expected:
            raise ValueError(f"mag must have shape {expected}, got {mag.shape}")

        # Prepare measurement magnitude (clamp + DC normalize etc.) WITHOUT mutating input
        ff0 = FormFactor(
            grid=grid, mag=mag.copy(), phase=np.zeros_like(mag)
        )
        self.frequency_constraints.apply(ff0)
        mag_meas = ff0.mag.copy()

        # init phase from processed magnitude
        phase = self.phase_init.initialize_phase(grid, mag=mag_meas)

        # initial complex FF: measured magnitude + init phase
        ff = FormFactor(grid=grid, mag=mag_meas.copy(), phase=phase, eps=self.eps_mag)

        # initial time profile + constraints
        prof = Profile.from_form_factor(ff, transform=self.transform, charge=None)
        self.time_constraints.apply(prof)

        self.stop.reset()

        while not self.stop():
            last_prof = prof.copy()
            last_ff = ff.copy()

            ff = prof.to_form_factor(self.transform)
            pre_ff = ff.copy()
            ff.mag = mag_meas.copy()
            self.frequency_constraints.apply(ff)
            post_ff = ff.copy()

            prof = ff.to_profile(transform=self.transform, charge=None)
            pre_prof = prof.copy()
            self.time_constraints.apply(prof)
            post_prof = prof.copy()

            self.stop.update(
                last_prof=last_prof,
                pre_prof=pre_prof,
                post_prof=post_prof,
                last_ff=last_ff,
                pre_ff=pre_ff,
                post_ff=post_ff,
            )

        # attach charge at the end (or inside Profile if you prefer)
        prof.charge = (
            charge_C  # only if Profile allows mutation; otherwise rebuild with charge
        )

        return prof, ff


