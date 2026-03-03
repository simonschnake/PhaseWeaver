# phase_weaver/core/constraints.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .utils import trapz_uniform

if TYPE_CHECKING:
    from .base import CurrentProfile, FormFactor, Grid


class TimeConstraint(ABC):
    @abstractmethod
    def apply(self, prof: "CurrentProfile") -> None:
        raise NotImplementedError

    def __add__(self, other: "TimeConstraint") -> "TimeConstraints":
        if isinstance(other, TimeConstraint):
            return TimeConstraints(self, other)
        raise NotImplementedError(
            "Can only add TimeConstraint to another TimeConstraint"
        )


class TimeConstraints(TimeConstraint):
    def __init__(self, *args: TimeConstraint):
        self.constraints = args

    def apply(self, prof: "CurrentProfile") -> None:
        for constraint in self.constraints:
            constraint.apply(prof)

    def __add__(self, other: "TimeConstraint") -> "TimeConstraints":
        if isinstance(other, TimeConstraints):
            return TimeConstraints(*self.constraints, *other.constraints)
        elif isinstance(other, TimeConstraint):
            return TimeConstraints(*self.constraints, other)
        else:
            raise NotImplementedError(
                "Can only add TimeConstraint to another TimeConstraint"
            )


class FrequencyConstraint(ABC):
    @abstractmethod
    def apply(self, ff: "FormFactor") -> None:
        raise NotImplementedError

    def __add__(self, other: "FrequencyConstraint") -> "FrequencyConstraints":
        if isinstance(other, FrequencyConstraint):
            return FrequencyConstraints(self, other)
        raise NotImplementedError(
            "Can only add FrequencyConstraint to another FrequencyConstraint"
        )


class FrequencyConstraints(FrequencyConstraint):
    def __init__(self, *args: FrequencyConstraint):
        self.constraints = args

    def __add__(self, other: "FrequencyConstraint") -> "FrequencyConstraints":
        if isinstance(other, FrequencyConstraints):
            return FrequencyConstraints(*self.constraints, *other.constraints)
        elif isinstance(other, FrequencyConstraint):
            return FrequencyConstraints(*self.constraints, other)
        else:
            raise NotImplementedError(
                "Can only add FrequencyConstraint to another FrequencyConstraint"
            )

    def apply(self, ff: "FormFactor") -> None:
        for constraint in self.constraints:
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
        self.eps = eps

    def apply(self, ff: "FormFactor") -> None:
        ff.mag = np.maximum(ff.mag, self.eps)


class EnforceDCOne(FrequencyConstraint):
    def __init__(self, eps: float = np.finfo(float).eps):
        self.eps = eps

    def apply(self, ff: "FormFactor") -> None:
        ff.mag /= np.max([ff.mag[0], self.eps])

class ReplacePhaseEndLinear(FrequencyConstraint):
    def __init__(self, grid: "Grid", start_freq: float, alpha: float):
        self.grid = grid
        self.start_freq = start_freq
        self.alpha = alpha

        if self.start_freq < 0:
            raise ValueError("start_freq must be non-negative")

        self._replace_mask = (self.grid.f_pos >= self.start_freq)
        if not np.any(self._replace_mask):
            raise ValueError("start_freq is above the maximum frequency in the grid")

        self._values = self.alpha * self.grid.f_pos[self._replace_mask]

    def apply(self, ff: "FormFactor") -> None:
        ff.phase[self._replace_mask] = self._values


class ReplacePhaseEndLinearShift(FrequencyConstraint):
    def __init__(self, grid: "Grid", start_freq: float, freq_x: float, freq_y: float):
        self.grid = grid
        self.start_freq = start_freq
        self.freq_x = freq_x
        self.freq_y = freq_y

        if self.start_freq < 0:
            raise ValueError("start_freq must be non-negative")

        self._replace_mask = (self.grid.f_pos >= self.start_freq)
        if not np.any(self._replace_mask):
            raise ValueError("start_freq is above the maximum frequency in the grid")

        if self.freq_x <= self.start_freq:
            raise ValueError("freq_x must be greater than start_freq")
        
        if self.freq_x > self.grid.f_pos[-1]:
            raise ValueError("freq_x must be less than or equal to the maximum frequency in the grid")

        self._freqs_mask = self.grid.f_pos[self._replace_mask]
        self._start_x = self._freqs_mask[0]

    def apply(self, ff: "FormFactor") -> None:
        self._start_y = ff.phase[self._replace_mask][0]
        slope = (self.freq_y - self._start_y) / (self.freq_x - self._start_x)
        values = slope * (self._freqs_mask - self._start_x) + self._start_y
        ff.phase[self._replace_mask] = values


        

