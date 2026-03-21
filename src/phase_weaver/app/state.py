from __future__ import annotations
from dataclasses import dataclass, field

from phase_weaver.core import CurrentProfile, Grid
from phase_weaver.model.profiles import AsymSuperGaussParams, asymmetric_super_gaussian

from .config import CHARGE_C, DT, T_MAX


def _default_background() -> AsymSuperGaussParams:
    return AsymSuperGaussParams(
        center=0.0,  # fixed at 0
        width=2.191914604893585e-14,  # width_scale
        skew=0.0,
        order=1.0,
        amplitude=5819.0,
    )


def _default_peak() -> AsymSuperGaussParams:
    # label in GUI will be "Spike"
    return AsymSuperGaussParams(
        center=-1e-14,
        width=7.21e-16,
        skew=0.0,
        order=0.5,
        amplitude=13000.0,
    )


def _default_peak2() -> AsymSuperGaussParams:
    # label in GUI will be "Spike 2"
    return AsymSuperGaussParams(
        center=1e-14,
        width=7.21e-16,
        skew=0.0,
        order=0.5,
        amplitude=13000.0,
    )


@dataclass(slots=True)
class ProfileModelState:
    dt: float = DT
    t_max: float = T_MAX
    charge: float = CHARGE_C

    background: AsymSuperGaussParams = field(default_factory=_default_background)
    peak: AsymSuperGaussParams = field(default_factory=_default_peak)

    peak2_enabled: bool = False
    peak2: AsymSuperGaussParams = field(default_factory=_default_peak2)


class ProfileModel:
    def __init__(self, state: ProfileModelState | None = None):
        self.state = state or ProfileModelState()
        self.grid = Grid.from_dt_tmax(
            dt=self.state.dt,
            t_max=self.state.t_max,
            snap_pow2=True,
            min_N=64,
        )

    def compute_profile(self) -> CurrentProfile:
        t = self.grid.t

        bg = self.state.background
        pk = self.state.peak
        pk2 = self.state.peak2

        density = asymmetric_super_gaussian(
            t,
            center=0.0,  # background is always centered
            width=bg.width,
            skew=bg.skew,
            order=bg.order,
            amplitude=bg.amplitude,
        )

        density += asymmetric_super_gaussian(
            t,
            center=pk.center,
            width=pk.width,
            skew=pk.skew,
            order=pk.order,
            amplitude=pk.amplitude,
        )

        if self.state.peak2_enabled:
            density += asymmetric_super_gaussian(
                t,
                center=pk2.center,
                width=pk2.width,
                skew=pk2.skew,
                order=pk2.order,
                amplitude=pk2.amplitude,
            )

        return CurrentProfile(
            grid=self.grid,
            values=density,
            charge=self.state.charge,
        )


@dataclass(slots=True)
class ReconstructionState:
    mag_init_mode: str
    phase_init_mode: str
    band_limit: bool
    band_limit_f_cut_hz: float


@dataclass(slots=True)
class MeasurementState:
    enabled: bool
    mode: str
    f_min_hz: float
    f_max_hz: float
    overlap_width_hz: float


@dataclass
class ControlsState:
    scenario: ProfileModelState
    measurement: MeasurementState
    reconstruction: ReconstructionState
