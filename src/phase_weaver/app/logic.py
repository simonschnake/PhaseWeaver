from dataclasses import dataclass

import numpy as np

from phase_weaver.app import config
from phase_weaver.core import (
    BandLimitedDCPhysicalRFFT,
    CurrentProfile,
    DCPhysicalRFFT,
    Transform,
    FormFactor,
    Profile,
)
from phase_weaver.core.constraints import (
    CenterFirstMoment,
    ClampMagnitude,
    EnforceDCOne,
    FrequencyConstraint,
    NonNegativity,
    NormalizeArea,
    ReplacePhaseEndLinearSmooth,
)
from phase_weaver.core.reconstruction import (
    GSMagnitudeOnly,
    MaxIter,
    MinimumPhaseCepstrum,
    MinIter,
    PhaseStoppedChanging,
    PredefinedPhase,
    ZeroPhase,
)
from phase_weaver.model.profile_model import ProfileModel

from .state import AppState, ReconstructionState


class AppLogic:
    def __init__(self):
        self.phase_last: np.ndarray | None = None
        self.center_prof = CenterFirstMoment()
        # --- Reconstruction core ---
        self._recon_time_constraints = (
            NonNegativity() + NormalizeArea() + CenterFirstMoment()
        )
        self._recon_stop_condition = (
            MaxIter(1_000) + PhaseStoppedChanging(tol=1e-8, patience=5) + MinIter(10)
        )

    def compute_initial(self, app_state: AppState) -> tuple[Profile, FormFactor]:
        prof_input = self._compute_input_profile(app_state)
        ff_input = self._compute_input_formfactor(
            prof_input, self._create_transform(app_state.reconstruction)
        )

        return prof_input, ff_input

    def compute_reconstruction(
        self, app_state: AppState, formfactor_input: FormFactor
    ) -> tuple[Profile, FormFactor]:

        reconstruction = self._build_reconstruction(app_state, formfactor_input)

        prof_recon, ff_recon = reconstruction.run(
            formfactor_input.mag, formfactor_input.grid
        )
        self.phase_last = ff_recon.phase
        return prof_recon, ff_recon

    def _compute_input_profile(self, app_state: AppState) -> CurrentProfile:
        profile_model = ProfileModel(app_state.scenario)
        prof = profile_model.compute_profile()
        if self.center_prof is not None:
            self.center_prof.apply(prof)
        return prof

    def _compute_input_formfactor(self, prof, transform: Transform):
        ff = prof.to_form_factor(transform=transform)
        return ff

    def _create_transform(self, recon_state: ReconstructionState) -> Transform:
        if recon_state.band_limit:
            f_cut = float(max(recon_state.band_limit_f_cut_hz, 0.0))
            transform = BandLimitedDCPhysicalRFFT(
                f_cut=f_cut, compact=False, unwrap_phase=True, dc_normalize=True
            )
        else:
            transform = DCPhysicalRFFT(unwrap_phase=True, dc_normalize=True)

        return transform

    def _build_reconstruction(
        self, app_state: AppState, ff_input: FormFactor
    ) -> GSMagnitudeOnly:
        return GSMagnitudeOnly(
            transform=self._create_transform(app_state.reconstruction),
            phase_init=self._make_phase_init(app_state, phase_in=ff_input.phase),
            time_constraints=self._recon_time_constraints,
            frequency_constraints=self._make_frequency_constraints(
                app_state=app_state, ff=ff_input
            ),
            stop=self._recon_stop_condition,
        )

    def _make_phase_init(self, app_state: AppState, phase_in: np.ndarray):

        mode = app_state.reconstruction.phase_init_mode

        if mode == "zero":
            return ZeroPhase()

        if mode == "real":
            return PredefinedPhase(phase=phase_in)

        if mode == "minphase":
            return MinimumPhaseCepstrum()

        if mode == "last":
            phase = self.phase_last if self.phase_last is not None else phase_in
            return PredefinedPhase(phase=phase)

        return MinimumPhaseCepstrum()

    def _make_frequency_constraints(
        self, app_state: AppState, ff: FormFactor
    ) -> FrequencyConstraint:

        constraints = ClampMagnitude(eps=1e-6) + EnforceDCOne()

        if app_state.reconstruction.linear_phase_end:
            constraints = constraints + ReplacePhaseEndLinearSmooth(
                grid=ff.grid,
                start_freq=app_state.reconstruction.linear_phase_end_start_freq_hz,
                freq_x=config.PHASE_END_REF_FREQ_HZ,
                freq_y=app_state.reconstruction.linear_phase_end_phase_at_300_thz,
                transition_width=30e12,  # 30 THz transition width for smoothness
            )

        return constraints
