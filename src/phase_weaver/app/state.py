from __future__ import annotations
from dataclasses import dataclass, field

from phase_weaver.core import CurrentProfile, Grid
from phase_weaver.model import profile_model as _profile_model
from phase_weaver.model.profile_model import ScenarioState
from phase_weaver.model.profiles import AsymSuperGaussParams, asymmetric_super_gaussian

from .config import (
    CRISP_SIMULATION_MODE,
    IR_SIMULATION_MODE,
    PHASE_INIT_MODE,
    RECONSTRUCTION_ALGORITHM,
    RECONSTRUCTION_ALGORITHM_DEFAULT,
    RECON_FREQUENCY_CONSTRAINT,
    RECON_FREQUENCY_CONSTRAINT_DEFAULT,
    RECON_STOP_CONDITION,
    RECON_STOP_CONDITION_DEFAULT,
    RECON_TIME_CONSTRAINT,
    RECON_TIME_CONSTRAINT_DEFAULT,
    PHASE_INIT_DEFAULT,
)

DT = _profile_model.DT
T_MAX = _profile_model.T_MAX
CHARGE_C = _profile_model.CHARGE_C


def _default_background() -> AsymSuperGaussParams:
    return _profile_model._default_background()


def _default_peak() -> AsymSuperGaussParams:
    return _profile_model._default_peak()


def _default_peak2() -> AsymSuperGaussParams:
    return _profile_model._default_peak2()


class ProfileModel(_profile_model.ProfileModel):
    """Compatibility facade for callers that still import from ``app.state``."""

    def __init__(self, state: ProfileModelState | None = None):
        self._sync_model_symbols()
        super().__init__(state)

    def compute_profile(self) -> CurrentProfile:
        self._sync_model_symbols()
        return super().compute_profile()

    @staticmethod
    def _sync_model_symbols() -> None:
        _profile_model.Grid = Grid
        _profile_model.CurrentProfile = CurrentProfile
        _profile_model.asymmetric_super_gaussian = asymmetric_super_gaussian


ProfileModelState = ScenarioState

@dataclass(slots=True)
class ReconstructionState:
    algorithm: RECONSTRUCTION_ALGORITHM = RECONSTRUCTION_ALGORITHM_DEFAULT
    phase_init_mode: PHASE_INIT_MODE = PHASE_INIT_DEFAULT
    use_ir_relative_constraint: bool = False
    use_fixed_ir_scale: bool = False
    fixed_ir_scale: float = 1.0
    time_constraints: set[RECON_TIME_CONSTRAINT] = field(
        default_factory=lambda: set(RECON_TIME_CONSTRAINT_DEFAULT)
    )
    frequency_constraints: set[RECON_FREQUENCY_CONSTRAINT] = field(
        default_factory=lambda: set(RECON_FREQUENCY_CONSTRAINT_DEFAULT)
    )
    stop_conditions: set[RECON_STOP_CONDITION] = field(
        default_factory=lambda: set(RECON_STOP_CONDITION_DEFAULT)
    )


@dataclass(slots=True)
class MeasurementState:
    crisp: bool = False
    infrared: bool = False
    crisp_scale: float = 1.0
    infrared_scale: float = 1.0
    crisp_simulation_mode: CRISP_SIMULATION_MODE = CRISP_SIMULATION_MODE.IDEAL
    crisp_n_shots: int = 1
    crisp_noise_seed: int = 0
    infrared_simulation_mode: IR_SIMULATION_MODE = IR_SIMULATION_MODE.IDEAL
    infrared_n_shots: int = 1
    infrared_noise_seed: int = 0


@dataclass
class ControlsState:
    scenario: ScenarioState
    measurement: MeasurementState
    reconstruction: ReconstructionState
