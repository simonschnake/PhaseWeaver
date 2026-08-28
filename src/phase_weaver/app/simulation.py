"""Application seam for detector forward simulations."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


class SimulationService:
    """Keep detector simulation orchestration behind an injectable facade."""

    def __init__(
        self,
        crisp_simulator: Callable[..., Any],
        ocean_simulator: Callable[..., Any],
    ) -> None:
        self._crisp_simulator = crisp_simulator
        self._ocean_simulator = ocean_simulator

    def simulate_crisp(self, input_profile: Any, measurement_state: Any) -> Any:
        return self._crisp_simulator(input_profile, measurement_state)

    def simulate_ocean(self, form_factor: Any, measurement_state: Any) -> Any:
        return self._ocean_simulator(form_factor, measurement_state)
