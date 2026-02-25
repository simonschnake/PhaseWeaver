# phase_weaver/core/constraints.py
from abc import ABC, abstractmethod

import numpy as np
from typing import TYPE_CHECKING

from .utils import trapz_uniform

if TYPE_CHECKING:
    from .base import CurrentProfile, FormFactor


class TimeConstraint(ABC):
    @abstractmethod
    def apply(self, prof: "CurrentProfile") -> None:
        raise NotImplementedError

class TimeConstraints(TimeConstraint):
    def __init__(self, *args: TimeConstraint):
        self.constraints = args

    def apply(self, prof: "CurrentProfile") -> None:
        for constraint in self.constraints:
            constraint.apply(prof)

class FrequencyConstraint(ABC):
    @abstractmethod
    def apply(self, ff: "FormFactor") -> None:
        raise NotImplementedError

class FrequencyConstraints(FrequencyConstraint):
    def __init__(self, *args: FrequencyConstraint):
        self.constraints = args

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
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def apply(self, ff: "FormFactor") -> None:
        ff.mag = np.maximum(ff.mag, self.eps)

class EnforceDCOne(FrequencyConstraint):
    def apply(self, ff: "FormFactor") -> None:
        ff.mag /= np.max([ff.mag[0], ff.eps])
    


