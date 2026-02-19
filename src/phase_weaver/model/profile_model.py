from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from phase_weaver.core.grid import Grid
from phase_weaver.core.profiles import AsymSuperGaussParams, asymmetric_super_gaussian


def _default_background() -> AsymSuperGaussParams:
    return AsymSuperGaussParams(
        center=0.0,                             # fixed at 0
        width=2.191914604893585e-14,            # width_scale
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

@dataclass(slots=True)
class ProfileModelState:
    dt: float = 1e-16
    t_max: float = 1e-13

    background: AsymSuperGaussParams = field(default_factory=_default_background)
    peak: AsymSuperGaussParams = field(default_factory=_default_peak)


class ProfileModel:
    def __init__(self, state: ProfileModelState | None = None):
        self.state = state or ProfileModelState()
        self._grid: Grid | None = None
        self._t: np.ndarray | None = None
        self._grid_key: tuple[float, float] | None = None

    def grid(self) -> Grid:
        key = (self.state.dt, self.state.t_max)
        if self._grid is None or self._grid_key != key:
            self._grid = Grid.from_dt_tmax(dt=self.state.dt, t_max=self.state.t_max)
            self._t = self._grid.t
            self._grid_key = key
        return self._grid

    def compute_profile(self) -> tuple[np.ndarray, np.ndarray]:
        _ = self.grid()
        t = self._t
        assert t is not None

        bg = self.state.background
        pk = self.state.peak

        bg.center = 0.0  # enforce background centered at 0

        y = (
            asymmetric_super_gaussian(
                t, center=bg.center, width=bg.width, skew=bg.skew, order=bg.order, amplitude=bg.amplitude
            )
            + asymmetric_super_gaussian(
                t, center=pk.center, width=pk.width, skew=pk.skew, order=pk.order, amplitude=pk.amplitude
            )
        )
        return t, y