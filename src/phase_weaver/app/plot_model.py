from dataclasses import dataclass

import numpy as np

from phase_weaver.core.base import CurrentProfile, FormFactor, Profile
from phase_weaver.core.utils import FWHMResult, fwhm_highest_peak


@dataclass(slots=True)
class TimePlotModel:
    profile_input: CurrentProfile
    profile_recon: CurrentProfile | None = None
    t_min: float | None = None
    t_max: float | None = None

    def __post_init__(self):
        if type(self.profile_input) is Profile:
            self.profile_input = CurrentProfile.from_profile(self.profile_input)
        if type(self.profile_recon) is Profile:
            self.profile_recon = CurrentProfile.from_profile(self.profile_recon)

    def update(
        self,
        profile_input: CurrentProfile | Profile | None = None,
        profile_recon: CurrentProfile | Profile | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> None:
        if profile_input is not None:
            if type(profile_input) is Profile:
                self.profile_input = CurrentProfile.from_profile(profile_input)
            elif type(profile_input) is CurrentProfile:
                self.profile_input = profile_input
            else:
                raise ValueError(
                    "profile_input must be of type CurrentProfile or Profile"
                )
        if profile_recon is not None:
            if type(profile_recon) is Profile:
                self.profile_recon = CurrentProfile.from_profile(profile_recon)
            elif type(profile_recon) is CurrentProfile:
                self.profile_recon = profile_recon
            else:
                raise ValueError(
                    "profile_recon must be of type CurrentProfile or Profile"
                )
        if t_min is not None:
            self.t_min = t_min
        if t_max is not None:
            self.t_max = t_max

    @property
    def t(self) -> np.ndarray:
        return self.profile_input.grid.t

    @property
    def current_input(self) -> np.ndarray:
        return self.profile_input.current

    @property
    def current_recon(self) -> np.ndarray | None:
        if self.profile_recon is not None:
            return self.profile_recon.current
        else:
            return None

    @property
    def current_input_kA(self) -> np.ndarray:
        return self.current_input * 1e-3

    @property
    def current_recon_kA(self) -> np.ndarray | None:
        if self.current_recon is not None:
            return self.current_recon * 1e-3
        else:
            return None

    @property
    def t_fs(self) -> np.ndarray:
        return self.t * 1e15

    @property
    def t_ui(self) -> np.ndarray:
        return self.t_fs[self._mask]

    @property
    def current_input_ui(self) -> np.ndarray:
        return self.current_input_kA[self._mask]

    @property
    def current_recon_ui(self) -> np.ndarray | None:
        if self.current_recon_kA is not None:
            return self.current_recon_kA[self._mask]
        else:
            return None

    @property
    def current_input_fwhm(self) -> FWHMResult:
        return fwhm_highest_peak(self.t_ui, self.current_input_ui)

    @property
    def current_recon_fwhm(self) -> FWHMResult | None:
        if self.current_recon_ui is not None:
            return fwhm_highest_peak(self.t_ui, self.current_recon_ui)
        else:
            return None

    @property
    def t_min_ui(self) -> float | None:
        if self.t_min is not None:
            return self.t_min * 1e15
        else:
            return np.min(self.t_fs)

    @property
    def t_max_ui(self) -> float | None:
        if self.t_max is not None:
            return self.t_max * 1e15
        else:
            return np.max(self.t_fs)

    @property
    def _mask(self) -> np.ndarray:
        mask = np.ones_like(self.t, dtype=bool)
        if self.t_min is not None:
            mask &= self.t >= self.t_min
        if self.t_max is not None:
            mask &= self.t <= self.t_max
        return mask


@dataclass(slots=True)
class SpectrumPlotModel:
    formfactor_input: FormFactor
    formfactor_recon: FormFactor | None = None
    f_min: float | None = 0.0
    f_max: float | None = 333e12

    def update(
        self,
        formfactor_input: FormFactor | None = None,
        formfactor_recon: FormFactor | None = None,
        f_min: float | None = None,
        f_max: float | None = None,
    ) -> None:
        if formfactor_input is not None:
            self.formfactor_input = formfactor_input
        if formfactor_recon is not None:
            self.formfactor_recon = formfactor_recon
        if f_min is not None:
            self.f_min = f_min
        if f_max is not None:
            self.f_max = f_max

    @property
    def f(self) -> np.ndarray:
        return self.formfactor_input.grid.f_pos

    @property
    def mag_input(self) -> np.ndarray:
        return self.formfactor_input.mag

    @property
    def phase_input(self) -> np.ndarray:
        return self.formfactor_input.phase

    @property
    def mag_recon(self) -> np.ndarray | None:
        return self.formfactor_recon.mag if self.formfactor_recon is not None else None

    @property
    def phase_recon(self) -> np.ndarray | None:
        return (
            self.formfactor_recon.phase if self.formfactor_recon is not None else None
        )

    @property
    def f_THz(self) -> np.ndarray:
        return self.f * 1e-12

    @property
    def f_ui(self) -> np.ndarray:
        return self.f_THz[self._mask]

    @property
    def mag_input_ui(self) -> np.ndarray:
        return self.mag_input[self._mask]

    @property
    def mag_recon_ui(self) -> np.ndarray | None:
        if self.mag_recon is not None:
            return self.mag_recon[self._mask]
        else:
            return None

    @property
    def phase_input_ui(self) -> np.ndarray:
        return self.phase_input[self._mask]

    @property
    def phase_recon_ui(self) -> np.ndarray | None:
        if self.phase_recon is not None:
            return self.phase_recon[self._mask]
        else:
            return None

    @property
    def _mask(self) -> np.ndarray:
        mask = np.ones_like(self.f, dtype=bool)
        if self.f_min is not None:
            mask &= self.f >= self.f_min
        if self.f_max is not None:
            mask &= self.f <= self.f_max
        return mask

    @property
    def f_min_ui(self) -> float | None:
        if self.f_min is not None:
            return self.f_min * 1e-12
        else:
            return np.min(self.f_THz)

    @property
    def f_max_ui(self) -> float | None:
        if self.f_max is not None:
            return self.f_max * 1e-12
        else:
            return np.max(self.f_THz)


@dataclass(slots=True)
class PlotModels:
    time: TimePlotModel
    spectrum: SpectrumPlotModel
