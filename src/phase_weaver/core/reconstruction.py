from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

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
    BlendMeasuredMagnitude,
)

from .measurement import MeasuredFormFactor

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
        if mag is None:
            raise ValueError("PhaseInitializer requires 'mag' keyword argument")
        phase = self.initialize_phase(grid, **kwargs)
        return FormFactor(
            grid=grid,
            mag=mag,
            phase=phase,
            eps=kwargs.get("eps_mag", np.finfo(float).eps),
        )


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
        if phase is None:
            raise ValueError("MagnitudeInitializer requires 'phase' keyword argument")
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

        return FormFactor(
            grid=grid,
            mag=mag,
            phase=phase,
            frequency_constraint=ClampMagnitude(eps=1e-6) + EnforceDCOne(),
        )


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

    def __init__(self, *criteria: StopCriterion):
        self._criteria = criteria or (self,)

    @property
    def stop(self) -> bool:
        return False

    @property
    def direct_stop(self) -> bool:
        return False

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def update(
        self,
        prof: Profile | None = None,
        ff: FormFactor | None = None,
    ) -> None: ...

    def __add__(self, other: StopCriterion) -> StopCriterion:
        return CombinedStopCriterion(*self._criteria, *other._criteria)

    def __call__(self) -> bool:
        return self.stop or self.direct_stop


class CombinedStopCriterion(StopCriterion):
    """
    Combine multiple StopCriterion objects with "any" or "all" logic.
    """

    name: str = "combined"

    def __init__(self, *criterion: StopCriterion) -> None:
        self._criteria = criterion

    def reset(self) -> None:
        for c in self._criteria:
            c.reset()

    def update(
        self,
        prof: Profile | None = None,
        ff: FormFactor | None = None,
    ) -> None:
        for c in self._criteria:
            c.update(
                prof=prof,
                ff=ff,
            )

    def __call__(self) -> bool:
        all_stop = all(c.stop for c in self._criteria)
        any_direct_stop = any(c.direct_stop for c in self._criteria)
        return all_stop or any_direct_stop


@dataclass
class MinIter(StopCriterion):
    n_iter: int
    name: str = "min_iter"

    def __post_init__(self) -> None:
        self._counter = 0
        self._criteria = (self,)

    def reset(self) -> None:
        self._counter = 0

    @property
    def stop(self) -> bool:
        return self._counter >= self.n_iter

    def update(
        self,
        prof: Profile | None = None,
        ff: FormFactor | None = None,
    ) -> None:
        self._counter += 1


@dataclass
class MaxIter(StopCriterion):
    n_iter: int
    name: str = "max_iter"

    def __post_init__(self) -> None:
        self._criteria = (self,)
        self._counter = 0

    def reset(self) -> None:
        self._counter = 0

    @property
    def direct_stop(self) -> bool:
        return self._counter >= self.n_iter

    def update(
        self,
        prof: Profile | None = None,
        ff: FormFactor | None = None,
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
        self._criteria = (self,)
        self.reset()

    def reset(self) -> None:
        self._count = 0
        self._mse: float = np.inf
        self._last_phase: np.ndarray | None = None

    @property
    def stop(self) -> bool:
        if self._mse < self.tol:
            self._count += 1
            return self._count >= self.patience
        self._count = 0
        return False

    def update(
        self,
        prof: Profile | None = None,
        ff: FormFactor | None = None,
    ) -> None:

        if ff is None:
            return

        phase = ff.phase.copy()

        if self._last_phase is None:
            self._mse = np.inf
        else:
            self._mse = float(np.mean((phase - self._last_phase) ** 2))

        self._last_phase = phase


@dataclass(slots=True)
class MeasurementStoppedChanging(StopCriterion):
    measured_ff: MeasuredFormFactor
    tol: float = 1e-6
    patience: int = 3
    name: str = "measurement_distance_below"

    def __post_init__(self) -> None:
        self._criteria = (self,)
        self.reset()

    def reset(self) -> None:
        self._count = 0
        self._err = np.inf

    @property
    def stop(self) -> bool:
        if self._err < self.tol:
            self._count += 1
            return self._count >= self.patience
        self._count = 0
        return False

    def update(
        self,
        prof: Profile | None = None,
        ff: FormFactor | None = None,
    ) -> None:
        if ff is None:
            return

        mag_interp = np.interp(
            self.measured_ff.freq,
            ff.grid.f_pos,
            ff.mag,
        )
        denom = max(np.linalg.norm(self.measured_ff.mag), np.finfo(float).eps)
        self._err = float(np.linalg.norm(mag_interp - self.measured_ff.mag) / denom)


# =============================================================================
# Reconstruction framework
# =============================================================================


class ReconstructionAlgorithm(ABC):
    """
    Abstract base for reconstruction algorithms.
    """

    name: str

    @abstractmethod
    def run(self, grid: Grid, **kwargs) -> tuple[Profile, FormFactor]: ...


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
        self,
        grid: Grid,
        **kwargs,
    ) -> tuple[Profile, FormFactor]:
        mag = kwargs.get("mag", None)
        if mag is None or type(mag) is not np.ndarray:
            raise ValueError(
                "GSMagnitudeOnly requires 'mag' keyword argument of type np.ndarray"
            )

        if grid is None or type(grid) is not Grid:
            raise ValueError(
                "GSMagnitudeOnly requires 'grid' keyword argument of type Grid"
            )

        expected = (grid.N // 2 + 1,)
        if mag.shape != expected:
            raise ValueError(f"mag must have shape {expected}, got {mag.shape}")

        # Prepare measurement magnitude (clamp + DC normalize etc.) WITHOUT mutating input
        ff0 = FormFactor(grid=grid, mag=mag.copy(), phase=np.zeros_like(mag))
        self.frequency_constraints.apply(ff0)
        mag_meas = ff0.mag.copy()

        # init phase from processed magnitude
        phase = self.phase_init.initialize_phase(grid, mag=mag_meas)

        # initial complex FF: measured magnitude + init phase
        ff = FormFactor(grid=grid, mag=mag_meas.copy(), phase=phase, eps=self.eps_mag)

        # initial time profile + constraints
        prof = Profile.from_form_factor(ff, transform=self.transform)
        self.time_constraints.apply(prof)

        self.stop.reset()

        while not self.stop():
            ff = prof.to_form_factor(self.transform)
            ff.mag = mag_meas.copy()
            self.frequency_constraints.apply(ff)

            prof = ff.to_profile(transform=self.transform)
            self.time_constraints.apply(prof)

            self.stop.update(prof=prof, ff=ff)

        return prof, ff


@dataclass(slots=True)
class GSMeasuredMagnitude(ReconstructionAlgorithm):
    """
    GS-style algorithm for partially measured magnitude:
      1) initialize full-grid form factor from a FormFactorInitializer
         (default: GaussianInitializer)
      2) smoothly blend the measured magnitude into that initial form factor
      3) repeatedly:
         - inverse transform to time
         - apply time constraints
         - forward transform to frequency
         - smoothly blend measured magnitude into current iterate
      4) stop when all stop criteria are satisfied
    """

    measured_ff: MeasuredFormFactor
    name: str = "gs_measured_magnitude"
    transform: Transform = field(
        default_factory=lambda: DCPhysicalRFFT(unwrap_phase=True, dc_normalize=False)
    )
    formfactor_init: FormFactorInitializer = field(default_factory=GaussianInitializer)
    time_constraints: TimeConstraint = field(
        default_factory=lambda: NonNegativity() + NormalizeArea() + CenterFirstMoment()
    )
    frequency_constraints: FrequencyConstraint = field(
        default_factory=lambda: ClampMagnitude(eps=1e-6) + EnforceDCOne()
    )
    stop: StopCriterion = field(
        default_factory=lambda: (
            MaxIter(1_000) + PhaseStoppedChanging(tol=1e-8, patience=5) + MinIter(10)
        )
    )
    transition_width: float | None = None
    overlap_power: float = 2.0
    initial_sigma: float = 10e-15
    eps_mag: float = 1e-30

    def __post_init__(self) -> None:
        self.frequency_constraints = (
            BlendMeasuredMagnitude(
                measured=self.measured_ff,
                power=self.overlap_power,
                transition_width=self.transition_width,
            )
            + self.frequency_constraints
        )
        self.stop = self.stop + MeasurementStoppedChanging(
            self.measured_ff, tol=1e-4, patience=3
        )

    def run(
        self,
        grid: Grid,
        **kwargs,
    ) -> tuple[Profile, FormFactor]:

        sigma = kwargs.get("sigma", self.initial_sigma)

        # 1) initial full-grid form factor from initializer
        ff = self.formfactor_init(grid, sigma=sigma)
        self.frequency_constraints.apply(ff)

        self.stop.reset()

        prof = ff.to_profile(transform=self.transform)
        self.time_constraints.apply(prof)

        while not self.stop():
            # back to time domain
            prof = ff.to_profile(transform=self.transform)
            self.time_constraints.apply(prof)

            ff = prof.to_form_factor(self.transform)
            self.frequency_constraints.apply(ff)

            self.stop.update(prof=prof, ff=ff)

        return prof, ff
