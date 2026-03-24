# phase_weaver/core/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil, log2
from typing import Tuple

import numpy as np

from .constraints import (
    FrequencyConstraint,
    NonNegativity,
    NormalizeArea,
    TimeConstraint,
    CombinedTimeConstraint,
)


def next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << int(ceil(log2(n)))


@dataclass(frozen=True, slots=True)
class Grid:
    N: int
    dt: float

    def __post_init__(self):
        if self.N <= 0:
            raise ValueError("N must be positive.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")

    @classmethod
    def from_dt_tmax(
        cls, *, dt: float, t_max: float, snap_pow2: bool = True, min_N: int = 64
    ) -> "Grid":
        if dt <= 0 or t_max <= 0:
            raise ValueError("dt and t_max must be positive.")
        N_raw = int(ceil(2.0 * t_max / dt))
        N_raw = max(N_raw, min_N)
        N = next_pow2(N_raw) if snap_pow2 else N_raw
        if N % 2 == 1:
            N += 1
        return cls(N=int(N), dt=float(dt))

    @classmethod
    def from_df_fmax(
        cls, df: float, f_max: float, *, snap_pow2: bool = True, min_N: int = 64
    ) -> "Grid":
        if df <= 0 or f_max <= 0:
            raise ValueError("df and f_max must be positive.")
        N_raw = int(ceil(2.0 * f_max / df))
        N_raw = max(N_raw, min_N)
        N = next_pow2(N_raw) if snap_pow2 else N_raw
        if N % 2 == 1:
            N += 1
        dt = 1.0 / (N * df)
        return cls(N=int(N), dt=float(dt))

    @property
    def df(self) -> float:
        return 1.0 / (self.N * self.dt)

    @property
    def f_nyq(self) -> float:
        return 1.0 / (2.0 * self.dt)

    @property
    def t(self) -> np.ndarray:
        half = self.N // 2
        return self.dt * np.arange(-half, half, dtype=float)

    @property
    def f_pos(self) -> np.ndarray:
        half = self.N // 2
        return self.df * np.arange(0, half + 1, dtype=float)

    @property
    def f_shift(self) -> np.ndarray:
        half = self.N // 2
        return self.df * np.arange(-half, half, dtype=float)


class Profile:
    def __init__(
        self,
        grid: Grid,
        values: np.ndarray,
        charge: float | None = None,
        time_constraint: TimeConstraint | None = None,
    ):
        self.grid = grid
        self.values = values
        self.charge = charge
        self._validate()
        self.time_constraint = time_constraint
        if self.time_constraint is not None:
            self.time_constraint.apply(self)

    def copy(self) -> "Profile":
        return Profile(
            grid=self.grid,
            values=self.values.copy(),
            charge=self.charge,
            time_constraint=self.time_constraint,
        )

    @classmethod
    def from_form_factor(
        cls,
        form_factor: FormFactor,
        transform: Transform | None = None,
        charge: float | None = None,
    ) -> CurrentProfile:
        if transform is None:
            transform = DCPhysicalRFFT()
        grid, values = transform.form_factor_to_profile(form_factor)
        return cls(grid=grid, values=values, charge=charge)

    def to_form_factor(self, transform: Transform | None = None) -> FormFactor:
        return FormFactor.from_profile(self, transform=transform)

    def _validate(self):
        if self.values.shape != (self.grid.N,):
            raise ValueError(f"values must have shape ({self.grid.N},)")

        if self.charge is not None and self.charge <= 0:
            raise ValueError("Charge must be positive if specified.")


class CurrentProfile(Profile):
    def __init__(
        self,
        grid: Grid,
        values: np.ndarray,
        charge: float | None = None,
    ):
        super().__init__(
            grid=grid,
            values=values,
            charge=charge,
            time_constraint=CombinedTimeConstraint(NonNegativity(), NormalizeArea()),
        )

    def copy(self) -> "CurrentProfile":
        return CurrentProfile(
            grid=self.grid,
            values=self.values.copy(),
            charge=self.charge,
        )

    @property
    def current(self) -> np.ndarray:
        if self.charge is None:
            raise ValueError("Charge is not set, cannot compute current.")
        return self.charge * self.values

    @classmethod
    def from_profile(cls, profile: Profile) -> "CurrentProfile":
        return cls(
            grid=profile.grid,
            values=profile.values,
            charge=profile.charge,
        )


class FormFactor:
    def __init__(
        self,
        grid: Grid,
        mag: np.ndarray,
        phase: np.ndarray,
        eps: float | None = None,
        frequency_constraint: FrequencyConstraint | None = None,
    ):
        self.grid = grid
        self.phase = phase
        self.mag = mag
        self.frequency_constraint = frequency_constraint
        if self.frequency_constraint is not None:
            self.frequency_constraint.apply(self)

    @classmethod
    def from_profile(
        cls, profile: Profile, transform: Transform | None = None
    ) -> FormFactor:
        if transform is None:
            transform = DCPhysicalRFFT()
        grid, mag, phase = transform.profile_to_form_factor(profile)
        return cls(grid=grid, mag=mag, phase=phase)

    def to_profile(
        self, transform: Transform | None = None, charge: float | None = None
    ) -> Profile:
        return Profile.from_form_factor(self, transform=transform, charge=charge)

    @property
    def value(self) -> np.ndarray:
        return self.mag * np.exp(1j * self.phase)

    def copy(self) -> "FormFactor":
        return FormFactor(
            grid=self.grid,
            mag=self.mag.copy(),
            phase=self.phase.copy(),
            frequency_constraint=self.frequency_constraint,
        )


class Transform(ABC):
    """
    A transform defines the mapping between a current profile and its form factor, and vice versa.
    Subclasses encode scaling, shifting, one-sided vs full spectrum, etc.
    """

    name: str

    @abstractmethod
    def profile_to_form_factor(
        self, profile: CurrentProfile
    ) -> Tuple[Grid, np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def form_factor_to_profile(
        self, form_factor: FormFactor
    ) -> Tuple[Grid, np.ndarray]:
        raise NotImplementedError


class DCPhysicalRFFT(Transform):
    def __init__(
        self,
        unwrap_phase: bool = True,
        dc_normalize: bool = True,
        eps_mag: float | None = None,
    ):
        self.unwrap_phase = unwrap_phase
        self.dc_normalize = dc_normalize
        self.eps_mag = eps_mag if eps_mag is not None else np.finfo(float).eps

    def profile_to_form_factor(
        self, profile: CurrentProfile
    ) -> Tuple[Grid, np.ndarray, np.ndarray]:
        x_shift = np.fft.ifftshift(profile.values)
        F_pos = np.fft.rfft(x_shift) * profile.grid.dt
        mag = np.abs(F_pos)

        if self.dc_normalize:
            mag0 = max(mag[0], self.eps_mag)
            mag = mag / mag0

        phase = np.angle(F_pos)
        if self.unwrap_phase:
            phase = np.unwrap(phase)

        return profile.grid, mag, phase

    def form_factor_to_profile(
        self, form_factor: FormFactor
    ) -> Tuple[Grid, np.ndarray]:
        mag = np.maximum(form_factor.mag, self.eps_mag)
        F_pos = mag * np.exp(1j * form_factor.phase)

        x_shift = np.fft.irfft(F_pos, n=form_factor.grid.N) / form_factor.grid.dt
        x = np.fft.fftshift(x_shift)

        return form_factor.grid, x


class BandLimitedDCPhysicalRFFT(DCPhysicalRFFT):
    def __init__(
        self,
        f_cut: float,
        compact: bool = True,
        unwrap_phase: bool = True,
        dc_normalize: bool = True,
        eps_mag: float | None = None,
    ):
        super().__init__(
            unwrap_phase=unwrap_phase, dc_normalize=dc_normalize, eps_mag=eps_mag
        )
        self.f_cut = f_cut
        self.compact = compact
        if self.f_cut <= 0:
            raise ValueError("f_cut must be positive.")

    def _k_cut(self, grid: Grid) -> int:
        # Largest retained positive-frequency bin index
        if self.f_cut >= grid.f_nyq:
            return grid.N // 2
        return int(np.floor(self.f_cut / grid.df))

    def profile_to_form_factor(
        self, profile: CurrentProfile
    ) -> tuple[Grid, np.ndarray, np.ndarray]:
        grid, mag, phase = super().profile_to_form_factor(profile)
        k_cut = self._k_cut(grid)

        if self.compact:
            # Keep only DC .. cutoff
            return grid, mag[: k_cut + 1].copy(), phase[: k_cut + 1].copy()

        # Keep full length, but hard-zero everything above cutoff
        mag_lp = mag.copy()
        phase_lp = phase.copy()
        mag_lp[k_cut + 1 :] = 0.0
        phase_lp[k_cut + 1 :] = 0.0
        return grid, mag_lp, phase_lp

    def form_factor_to_profile(
        self, form_factor: FormFactor
    ) -> tuple[Grid, np.ndarray]:
        mag = np.asarray(form_factor.mag, dtype=float)
        phase = np.asarray(form_factor.phase, dtype=float)

        if mag.ndim != 1 or phase.ndim != 1:
            raise ValueError("mag and phase must be 1D arrays.")
        if mag.shape != phase.shape:
            raise ValueError("mag and phase must have the same shape.")

        n_full = form_factor.grid.N // 2 + 1
        if len(mag) > n_full:
            raise ValueError("Too many positive-frequency bins for this grid.")

        # If compact, zero-pad the missing high-frequency bins
        if len(mag) < n_full:
            mag_full = np.zeros(n_full, dtype=float)
            phase_full = np.zeros(n_full, dtype=float)
            mag_full[: len(mag)] = mag
            phase_full[: len(phase)] = phase

            F_pos = np.maximum(mag_full, self.eps_mag) * np.exp(1j * phase_full)
            x_shift = np.fft.irfft(F_pos, n=form_factor.grid.N) / form_factor.grid.dt
            x = np.fft.fftshift(x_shift)
            return form_factor.grid, x

        # Already full-length
        return super().form_factor_to_profile(form_factor)

    def f_pos_used(self, grid: Grid) -> np.ndarray:
        """
        Frequency axis corresponding to the retained bins in compact mode.
        """
        k_cut = self._k_cut(grid)
        return grid.f_pos[: k_cut + 1]
