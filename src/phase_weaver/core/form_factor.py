from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from .grid import Grid
from .transform import Transform, DCPhysicalRFFT

if TYPE_CHECKING:
    from .current_profile import CurrentProfile


class FormFactor:
    def __init__(self, grid: Grid, mag: np.ndarray, phase: np.ndarray, eps: float | None = None):
        self.grid = grid
        self.eps = eps if eps is not None else np.finfo(float).eps
        self.mag = np.maximum(mag, self.eps)
        self.phase = phase

    @classmethod
    def from_profile(cls, profile: "CurrentProfile", transform: Transform | None = None) -> "FormFactor":
        if transform is None:
            transform = DCPhysicalRFFT()
        grid, mag, phase, eps = transform.profile_to_form_factor(profile)
        return cls(grid=grid, mag=mag, phase=phase, eps=eps)

    def to_profile(self, transform: Transform | None = None, charge: float | None = None):
        from .current_profile import CurrentProfile
        return CurrentProfile.from_form_factor(self, transform=transform, charge=charge)