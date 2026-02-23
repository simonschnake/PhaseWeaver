# phase_weaver/core/transform.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np

from .grid import Grid

if TYPE_CHECKING:
    from .current_profile import CurrentProfile
    from .form_factor import FormFactor


class Transform(ABC):
    """
    A transform defines the mapping between a current profile and its form factor, and vice versa.
    Subclasses encode scaling, shifting, one-sided vs full spectrum, etc.
    """

    name: str

    @abstractmethod
    def profile_to_form_factor(
        self, profile: "CurrentProfile"
    ) -> Tuple[Grid, np.ndarray, np.ndarray, float]:
        raise NotImplementedError

    @abstractmethod
    def form_factor_to_profile(
        self, form_factor: "FormFactor"
    ) -> Tuple[Grid, np.ndarray]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class DCPhysicalRFFT(Transform):
    unwrap_phase: bool = True
    dc_normalize: bool = True
    eps_mag: float = np.finfo(float).eps

    def profile_to_form_factor(
        self, profile: "CurrentProfile"
    ) -> Tuple[Grid, np.ndarray, np.ndarray, float]:
        x_shift = np.fft.ifftshift(profile.density)
        F_pos = np.fft.rfft(x_shift) * profile.grid.dt
        mag = np.abs(F_pos)

        if self.dc_normalize:
            mag0 = max(mag[0], self.eps_mag)
            mag = mag / mag0

        phase = np.angle(F_pos)
        if self.unwrap_phase:
            phase = np.unwrap(phase)

        return profile.grid, mag, phase, self.eps_mag

    def form_factor_to_profile(self, form_factor: "FormFactor") -> Tuple[Grid, np.ndarray]:
        mag = np.maximum(form_factor.mag, self.eps_mag)
        F_pos = mag * np.exp(1j * form_factor.phase)

        x_shift = np.fft.irfft(F_pos, n=form_factor.grid.N) / form_factor.grid.dt
        x = np.fft.fftshift(x_shift)

        x = np.maximum(x, 0)
        integral = np.trapezoid(x, form_factor.grid.t)
        if integral <= 0:
            raise ValueError("Reconstructed profile has non-positive integral, cannot normalize charge.")
        density = x / integral

        return form_factor.grid, density

