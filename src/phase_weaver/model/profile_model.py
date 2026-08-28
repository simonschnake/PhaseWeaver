"""Toy current-profile scenario model."""
from __future__ import annotations

from dataclasses import dataclass, field

from phase_weaver.core import CurrentProfile, Grid
from phase_weaver.model.profiles import AsymSuperGaussParams, asymmetric_super_gaussian

DT = 1e-16
T_MAX = 1e-12
CHARGE_C = 250e-12


def _default_background() -> AsymSuperGaussParams:
    return AsymSuperGaussParams(
        center=0.0,
        width=2.191914604893585e-14,
        skew=0.0,
        order=1.0,
        amplitude=5819.0,
    )


def _default_peak() -> AsymSuperGaussParams:
    return AsymSuperGaussParams(
        center=-1e-14,
        width=7.21e-16,
        skew=0.0,
        order=0.5,
        amplitude=13000.0,
    )


def _default_peak2() -> AsymSuperGaussParams:
    return AsymSuperGaussParams(
        center=1e-14,
        width=7.21e-16,
        skew=0.0,
        order=0.5,
        amplitude=13000.0,
    )


@dataclass(slots=True)
class ScenarioState:
    dt: float = DT
    t_max: float = T_MAX
    charge: float = CHARGE_C
    background: AsymSuperGaussParams = field(default_factory=_default_background)
    peak: AsymSuperGaussParams = field(default_factory=_default_peak)
    peak2_enabled: bool = False
    peak2: AsymSuperGaussParams = field(default_factory=_default_peak2)


ProfileModelState = ScenarioState


class ProfileModel:
    def __init__(self, state: ScenarioState | None = None):
        self.state = state or ScenarioState()
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
            center=0.0,
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
        return CurrentProfile(grid=self.grid, values=density, charge=self.state.charge)
