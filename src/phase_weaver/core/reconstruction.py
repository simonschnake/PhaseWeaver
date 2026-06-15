from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .base import (
    DCPhysicalRFFT,
    FormFactor,
    Grid,
    Profile,
)
from .constraints import (
    CenterFirstMoment,
    ClampMagnitude,
    CutAfterNthZeroFromPeak,
    EnforceDCOne,
    NonNegativity,
    NormalizeArea,
    BlendMeasuredMagnitude,
)

from .measurement import MeasuredFormFactor

# from .utils import exponential_extend
# from .utils import quadratic_log_extend
from .utils import gaussian_extend, interpolate_between_functions


class PhaseInitState(Protocol):
    phase_init_mode: object


def minimum_phase_from_mag(
    grid: Grid, mag: np.ndarray, eps: float = np.finfo(float).eps
) -> np.ndarray:
    N = grid.N
    expected = (N // 2 + 1,)
    if mag.shape != expected:
        raise ValueError(f"mag_pos must have shape {expected}, got {mag.shape}")

    logmag_pos = np.log(np.maximum(mag, eps))

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


def _extension_min_positive(y_source: np.ndarray) -> float:
    """
    Choose a small positive floor for log-space tail fitting.

    The measured magnitudes are allowed to contain zeros, but the Gaussian
    extension fits in log-space and therefore needs strictly positive values.
    We keep the floor tiny relative to the signal scale so the initial guess
    remains close to the measured data.
    """
    y_source = np.asarray(y_source, dtype=float)
    positive_peak = float(np.max(y_source)) if y_source.size else 0.0
    if not np.isfinite(positive_peak) or positive_peak <= 0.0:
        return np.finfo(float).eps
    return max(np.finfo(float).eps, positive_peak * 1e-12)


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


@dataclass(slots=True)
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
class MeasurementsStoppedChanging(StopCriterion):
    measured_ff: tuple[MeasuredFormFactor, ...]
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

        err = np.empty(len(self.measured_ff), dtype=float)

        for i, meas in enumerate(self.measured_ff):
            mag_interp = np.interp(
                meas.freq,
                ff.grid.f_pos,
                ff.mag,
            )
            denom = max(np.linalg.norm(meas.mag), np.finfo(float).eps)
            self._err = float(np.linalg.norm(mag_interp - meas.mag) / denom)
            denom = max(np.linalg.norm(meas.mag), np.finfo(float).eps)
            err[i] = float(np.linalg.norm(mag_interp - meas.mag) / denom)

        self._err = float(np.sqrt(np.mean(err**2)))


# =============================================================================
# Reconstruction framework
# =============================================================================


class ReconstructionAlgorithm(ABC):
    """
    Abstract base for reconstruction algorithms.
    """

    name: str

    @abstractmethod
    def run(self) -> tuple[Profile, FormFactor]: ...


# =============================================================================
# Gerchberg–Saxton style: magnitude-only -> phase retrieval
# =============================================================================


class GerchbergSaxton(ReconstructionAlgorithm):
    """ """

    name: str = "GerchbergSaxton"

    def __init__(
        self,
        grid: Grid,
        measurements: tuple[MeasuredFormFactor, ...],
        reconstruction_state: PhaseInitState,
        formfactor_input: FormFactor | None = None,
        phase_last: np.ndarray | None = None,
    ):

        self.grid = grid
        self.measurements = measurements
        self.reconstruction_state = reconstruction_state
        self.formfactor_input = formfactor_input
        self.phase_last = phase_last
        self.last_iterations = 0
        self.last_stop_reason = "not_run"
        self.last_measurement_error: float | None = None

        self.transform = DCPhysicalRFFT(
            unwrap_phase=True,
            dc_normalize=False,
        )

        self.formfactor_init = self._calculate_init_formfactor(
            grid,
            measurements,
            reconstruction_state,
            formfactor_input=formfactor_input,
            phase_last=phase_last,
        )

        self.time_constraints = (
            CutAfterNthZeroFromPeak(n=3)
            + NonNegativity()
            + NormalizeArea()
            + CenterFirstMoment()
        )

        self.frequency_constraints = (
            ClampMagnitude()
            + EnforceDCOne()
            + BlendMeasuredMagnitude(measurements, scale=True)
        )
        self.stop = (
            MaxIter(n_iter=1_000)
            + PhaseStoppedChanging(tol=1e-8, patience=5)
            + MinIter(n_iter=10)
            + MeasurementsStoppedChanging(
                measured_ff=measurements, tol=1e-4, patience=3
            )
        )

    def run(self) -> tuple[Profile, FormFactor]:

        ff = self.formfactor_init
        self.frequency_constraints.apply(ff)

        self.stop.reset()
        self.last_iterations = 0
        self.last_stop_reason = "not_run"
        self.last_measurement_error = None

        prof = ff.to_profile(transform=self.transform)
        self.time_constraints.apply(prof)

        while not self.stop():
            # back to time domain
            prof = ff.to_profile(transform=self.transform)
            self.time_constraints.apply(prof)

            ff = prof.to_form_factor(self.transform)
            self.frequency_constraints.apply(ff)

            self.stop.update(prof=prof, ff=ff)
            self.last_iterations += 1

        self.last_stop_reason = self._describe_stop_reason()
        self.last_measurement_error = self._measurement_error(ff)
        return prof, ff

    def _describe_stop_reason(self) -> str:
        reasons: list[str] = []
        for criterion in self.stop._criteria:
            if criterion.direct_stop or criterion.stop:
                reasons.append(criterion.name)
        return "+".join(reasons) if reasons else "unknown"

    def _measurement_error(self, ff: FormFactor) -> float:
        err = np.empty(len(self.measurements), dtype=float)
        for i, meas in enumerate(self.measurements):
            mag_interp = np.interp(meas.freq, ff.grid.f_pos, ff.mag)
            denom = max(np.linalg.norm(meas.mag), np.finfo(float).eps)
            err[i] = float(np.linalg.norm(mag_interp - meas.mag) / denom)
        return float(np.sqrt(np.mean(err**2)))

    def _calculate_init_formfactor(
        self,
        grid: Grid,
        measurements: tuple[MeasuredFormFactor, ...],
        reconstruction_state: PhaseInitState,
        formfactor_input: FormFactor | None = None,
        phase_last: np.ndarray | None = None,
    ) -> FormFactor:
        from phase_weaver.app.state import PHASE_INIT_MODE

        min_positive = _extension_min_positive(measurements[0].mag)
        mag_init = gaussian_extend(
            grid.f_pos,
            measurements[0].freq,
            measurements[0].mag,
            min_positive=min_positive,
        )
        freq_left = measurements[0].freq[0]

        for meas in measurements[1:]:
            freq_right = meas.freq[0]
            if freq_right < freq_left:
                freq_left, freq_right = freq_right, freq_left

            mag_init = interpolate_between_functions(
                grid.f_pos,
                mag_init,
                gaussian_extend(
                    grid.f_pos,
                    meas.freq,
                    meas.mag,
                    min_positive=_extension_min_positive(meas.mag),
                ),
                freq_left,
                freq_right,
            )

        formfactor_init = FormFactor(
            grid=grid,
            mag=mag_init,
            phase=np.zeros_like(mag_init),
        )

        phase_init_mode = reconstruction_state.phase_init_mode

        if phase_init_mode == PHASE_INIT_MODE.ZERO:
            # default case
            pass

        elif phase_init_mode == PHASE_INIT_MODE.REAL:
            if formfactor_input is None:
                print("Input form factor is required for REAL phase initialization")
            else:
                if formfactor_input.phase.shape != formfactor_init.phase.shape:
                    raise ValueError(
                        "input form factor phase must match initialized phase shape"
                    )
                formfactor_init.phase[:] = formfactor_input.phase

        elif phase_init_mode == PHASE_INIT_MODE.MINPHASE:
            formfactor_init.phase = minimum_phase_from_mag(grid, mag_init)

        elif phase_init_mode == PHASE_INIT_MODE.LAST:
            if phase_last is not None:
                if phase_last.shape != formfactor_init.phase.shape:
                    raise ValueError(
                        "last phase must match initialized phase shape"
                    )
                formfactor_init.phase[:] = phase_last
            else:
                print(
                    "Warning: no previous phase found for LAST initialization, falling back to Zero Phase"
                )

        else:
            raise ValueError(f"unknown phase initialization mode: {phase_init_mode!r}")

        return formfactor_init
