from dataclasses import dataclass

import phase_weaver.app.config as config
from phase_weaver.model.profile_model import ProfileModelState


@dataclass(slots=True)
class ReconstructionState:
    phase_init_mode: config.PHASE_INIT_MODES_T
    linear_phase_end: bool
    linear_phase_end_start_freq_hz: float
    linear_phase_end_phase_at_300_thz: float
    band_limit: bool
    band_limit_f_cut_hz: float


@dataclass
class AppState:
    scenario: ProfileModelState
    reconstruction: ReconstructionState

    @classmethod
    def default(cls) -> "AppState":
        return cls(
            scenario=ProfileModelState(
                dt=config.DT,
                t_max=config.T_MAX,
                charge=config.CHARGE_C,
                background=config.BACKGROUND_SPEC,
                peak=config.SPIKE_SPEC,
                peak2_enabled=config.PEAK2_ENABLED,
                peak2=config.SPIKE_SPEC,
            ),
            reconstruction=ReconstructionState(
                phase_init_mode=config.PHASE_INIT_DEFAULT,
                linear_phase_end=config.LINEAR_PHASE_END_ENABLED,
                linear_phase_end_start_freq_hz=config.LINEAR_PHASE_END_START_FREQ_HZ,
                linear_phase_end_phase_at_300_thz=config.LINEAR_PHASE_END_PHASE_AT_300_THZ,
                band_limit=config.BAND_LIMIT_ENABLED,
                band_limit_f_cut_hz=config.BAND_LIMIT_F_CUT_HZ,
            ),
        )