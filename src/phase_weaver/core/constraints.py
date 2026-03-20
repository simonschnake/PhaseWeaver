# phase_weaver/core/constraints.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .utils import smooth_overlap, trapz_uniform

if TYPE_CHECKING:
    from .base import CurrentProfile, FormFactor, Grid
    from .measurement import MeasuredFormFactor


class TimeConstraint(ABC):
    def __init__(self, *constraints: TimeConstraint):
        self._constraints = constraints or (self,)

    @abstractmethod
    def apply(self, prof: "CurrentProfile") -> None: ...

    def __add__(self, other: TimeConstraint) -> TimeConstraint:
        return CombinedTimeConstraint(*self._constraints, *other._constraints)


class CombinedTimeConstraint(TimeConstraint):
    def __init__(self, *constraints: TimeConstraint):
        self._constraints = constraints

    def apply(self, prof: "CurrentProfile") -> None:
        for constraint in self._constraints:
            constraint.apply(prof)


class FrequencyConstraint(ABC):
    def __init__(self, *constraints: FrequencyConstraint):
        self._constraints = constraints or (self,)

    @abstractmethod
    def apply(self, ff: "FormFactor") -> None: ...

    def __add__(self, other: FrequencyConstraint) -> FrequencyConstraint:
        return CombinedFrequencyConstraint(*self._constraints, *other._constraints)


class CombinedFrequencyConstraint(FrequencyConstraint):
    def __init__(self, *constraints: FrequencyConstraint):
        self._constraints = constraints

    def apply(self, ff: "FormFactor") -> None:
        for constraint in self._constraints:
            constraint.apply(ff)


class NonNegativity(TimeConstraint):
    def apply(self, prof: "CurrentProfile") -> None:
        prof.values = np.maximum(prof.values, 0.0)


class NormalizeArea(TimeConstraint):
    def apply(self, prof: "CurrentProfile") -> None:
        area = trapz_uniform(prof.grid.dt, prof.values)
        if area == 0:
            print("Warning: area is zero, cannot normalize.")
            return
        prof.values /= area


class CenterFirstMoment(TimeConstraint):
    def apply(self, prof: "CurrentProfile") -> None:
        g = prof.grid
        mean_t = trapz_uniform(g.dt, prof.values * g.t)
        shift = int(np.round(mean_t / g.dt))
        prof.values = np.roll(prof.values, -shift)


class ClampMagnitude(FrequencyConstraint):
    def __init__(self, eps: float = np.finfo(float).eps):
        super().__init__()
        self.eps = eps

    def apply(self, ff: "FormFactor") -> None:
        ff.mag = np.maximum(ff.mag, self.eps)


class EnforceDCOne(FrequencyConstraint):
    def __init__(self, eps: float = np.finfo(float).eps):
        super().__init__()
        self.eps = eps

    def apply(self, ff: "FormFactor") -> None:
        ff.mag /= np.max([ff.mag[0], self.eps])


class ReplacePhaseEndLinear(FrequencyConstraint):
    def __init__(self, grid: "Grid", start_freq: float, alpha: float):
        super().__init__()
        self.grid = grid
        self.start_freq = start_freq
        self.alpha = alpha

        if not np.isfinite(self.start_freq) or self.start_freq < 0:
            raise ValueError("start_freq must be finite and non-negative")
        if start_freq > grid.f_pos[-1]:
            raise ValueError("start_freq is above the maximum frequency in the grid")

        self._replace_mask = self.grid.f_pos >= self.start_freq

        self._values = self.alpha * self.grid.f_pos[self._replace_mask]

    def apply(self, ff: "FormFactor") -> None:
        ff.phase[self._replace_mask] = self._values


class ReplacePhaseEndLinearShift(FrequencyConstraint):
    def __init__(self, grid: "Grid", start_freq: float, freq_x: float, freq_y: float):
        super().__init__()
        self.grid = grid
        self.start_freq = start_freq
        self.freq_x = freq_x
        self.freq_y = freq_y

        if self.start_freq < 0:
            raise ValueError("start_freq must be non-negative")

        self._replace_mask = self.grid.f_pos >= self.start_freq
        if not np.any(self._replace_mask):
            raise ValueError("start_freq is above the maximum frequency in the grid")

        if self.freq_x <= self.start_freq:
            raise ValueError("freq_x must be greater than start_freq")

        if self.freq_x > self.grid.f_pos[-1]:
            raise ValueError(
                "freq_x must be less than or equal to the maximum frequency in the grid"
            )

        self._freqs_mask = self.grid.f_pos[self._replace_mask]
        self._start_x = self._freqs_mask[0]

    def apply(self, ff: "FormFactor") -> None:
        self._start_y = ff.phase[self._replace_mask][0]
        slope = (self.freq_y - self._start_y) / (self.freq_x - self._start_x)
        values = slope * (self._freqs_mask - self._start_x) + self._start_y
        ff.phase[self._replace_mask] = values


class ReplacePhaseEndLinearSmooth(FrequencyConstraint):
    """
    Smoothly blend in a linear high-frequency phase tail.

    The inserted tail is the line through (0, 0) and (freq_x, freq_y), applied
    from `start_freq` onward and smoothly merged into the original phase.
    """

    def __init__(
        self,
        start_freq: float,
        freq_x: float,
        freq_y: float,
        power: float = 2.0,
        transition_width: float | None = None,
        *constraints: FrequencyConstraint,
    ):
        super().__init__()
        self.start_freq = start_freq
        self.freq_x = freq_x
        self.freq_y = freq_y
        self.power = power
        self.transition_width = transition_width

        if not np.isfinite(self.start_freq) or self.start_freq < 0:
            raise ValueError("start_freq must be finite and non-negative")
        if not np.isfinite(self.freq_x):
            raise ValueError("freq_x must be finite")
        if not np.isfinite(self.freq_y):
            raise ValueError("freq_y must be finite")
        if not np.isfinite(self.power) or self.power <= 0:
            raise ValueError("power must be positive")
        if self.transition_width is not None and (
            not np.isfinite(self.transition_width) or self.transition_width < 0
        ):
            raise ValueError("transition_width must be finite and non-negative")

        if self.freq_x <= self.start_freq:
            raise ValueError("freq_x must be greater than start_freq")

        self.slope = self.freq_y / self.freq_x

    def apply(self, ff: "FormFactor") -> None:
        f_pos = ff.grid.f_pos

        if self.freq_x > f_pos[-1]:
            raise ValueError(
                "freq_x must be less than or equal to the maximum frequency in the grid"
            )

        replace_mask = f_pos >= self.start_freq

        f_mask = np.asarray(f_pos[replace_mask], dtype=float)
        lin_phase = self.slope * f_mask

        ff.phase = smooth_overlap(
            x_target=f_pos,
            y_target=ff.phase,
            x_source=f_mask,
            y_source=lin_phase,
            power=self.power,
            transition_width=self.transition_width,
        )


class BlendMeasuredMagnitude(FrequencyConstraint):
    """
    Smoothly blend a measured magnitude into ff.mag.

    Parameters
    ----------
    measured:
        Measured form factor.
    favor:
        Blend strength in [0, 1]. Higher means closer to measured data.
    transition_width:
        Width of the smooth transition at the overlap boundaries, in frequency
        units. If None, defaults to 20% of overlap span.
    """

    def __init__(
        self,
        measured: MeasuredFormFactor,
        power: float = 2.0,
        transition_width: float | None = None,
    ):
        super().__init__()
        if not np.isfinite(power) or power <= 0:
            raise ValueError("power must be positive")
        if transition_width is not None and (
            not np.isfinite(transition_width) or transition_width < 0
        ):
            raise ValueError("transition_width must be finite and non-negative")

        self.measured = measured
        self.power = float(power)
        self.transition_width = (
            float(transition_width) if transition_width is not None else None
        )

    def apply(self, ff: "FormFactor") -> None:
        ff.mag = smooth_overlap(
            x_target=ff.grid.f_pos,
            y_target=ff.mag,
            x_source=self.measured.freq,
            y_source=self.measured.mag,
            power=self.power,
            transition_width=self.transition_width,
        )


class CutAfterNthZeroFromPeak(TimeConstraint):
    def __init__(
        self, n: int = 10, threshold: float = 0.0, keep_crossing: bool = False
    ):
        super().__init__()

        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite")

        self.n = n
        self.threshold = threshold
        self.keep_crossing = keep_crossing

    def apply(self, prof: "CurrentProfile") -> None:
        x = np.asarray(prof.values, dtype=float)

        if x.ndim != 1:
            raise ValueError("CutAfterNthZeroFromPeak only supports 1D profiles")
        if x.size == 0:
            return

        imax = int(np.argmax(x))
        y = x.copy()

        # Links vom Maximum:
        # Alle Indizes mit x <= threshold, dann die n-te von innen nach außen.
        left_hits = np.where(x[:imax] <= self.threshold)[0]
        if left_hits.size >= self.n:
            left_cut = int(left_hits[-self.n])
            if self.keep_crossing:
                y[:left_cut] = 0.0
            else:
                y[: left_cut + 1] = 0.0

        # Rechts vom Maximum:
        # Alle Indizes mit x <= threshold, dann die n-te nach außen.
        right_hits = np.where(x[imax + 1 :] <= self.threshold)[0]
        if right_hits.size >= self.n:
            right_cut = int(imax + 1 + right_hits[self.n - 1])
            if self.keep_crossing:
                y[right_cut + 1 :] = 0.0
            else:
                y[right_cut:] = 0.0

        prof.values = y
