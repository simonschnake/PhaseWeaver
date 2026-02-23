from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .grid import Grid
from .transform import DCPhysicalRFFT, Transform

if TYPE_CHECKING:
    from .form_factor import FormFactor


class CurrentProfile:
    def __init__(
        self, grid: Grid, values: np.ndarray, charge: float | None = None
    ) -> None:
        self.grid = grid
        self.density = np.maximum(values, 0)
        integral = np.trapezoid(self.density, self.grid.t)
        if integral <= 0:
            raise ValueError("Charge must be positive")
        self.density /= integral
        self.charge = charge

    @classmethod
    def from_form_factor(
        cls,
        form_factor: "FormFactor",
        transform: Transform | None = None,
        charge: float | None = None,
    ) -> "CurrentProfile":
        if transform is None:
            transform = DCPhysicalRFFT()
        grid, density = transform.form_factor_to_profile(form_factor)
        return cls(grid=grid, values=density, charge=charge)

    def to_form_factor(self, transform: Transform | None = None) -> "FormFactor":
        # local import avoids circular dependency at import time
        from .form_factor import FormFactor

        return FormFactor.from_profile(self, transform=transform)

    @property
    def current(self) -> np.ndarray:
        if self.charge is None:
            raise ValueError("Charge is not set, cannot compute current.")
        return self.charge * self.density

