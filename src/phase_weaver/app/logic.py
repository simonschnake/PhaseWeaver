import numpy as np

from phase_weaver.app import config
from phase_weaver.core import (
    BandLimitedDCPhysicalRFFT,
    CurrentProfile,
    DCPhysicalRFFT,
    FormFactor,
    Grid,
    Profile,
    Transform,
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
from phase_weaver.core.measurement import MeasuredFormFactor
from phase_weaver.core.reconstruction import (
    GSMeasuredMagnitude,
    MaxIter,
    MinimumPhaseCepstrum,
    MinIter,
    PhaseStoppedChanging,
    PredefinedPhase,
    ReconstructionAlgorithm,
    ZeroPhase,
)

from .state import AppState, MeasurementState, ReconstructionState, ProfileModel


class AppLogic:
    def __init__(self):
        self.phase_last: np.ndarray | None = None
        self.center_prof = CenterFirstMoment()

        self._recon_time_constraints = (
            NonNegativity() + NormalizeArea() + CenterFirstMoment()
        )
        self._recon_stop_condition = (
            MaxIter(1_000) + PhaseStoppedChanging(tol=1e-8, patience=5) + MinIter(10)
        )

    def compute_initial(
        self, app_state: AppState
    ) -> tuple[Profile, FormFactor, MeasuredFormFactor]:
        transform = self._create_transform(app_state.reconstruction)
        prof_input = self._compute_input_profile(app_state)
        ff_input = self._compute_input_formfactor(prof_input, transform=transform)
        measurement = self._make_measured_formfactor(app_state, ff_true=ff_input)
        return prof_input, ff_input, measurement

    def compute_reconstruction(
        self,
        app_state: AppState,
        measured_ff: MeasuredFormFactor,
        grid: Grid,
    ) -> tuple[Profile, FormFactor]:
        reconstruction = self._build_reconstruction(app_state, measured_ff)

        prof_recon, ff_recon = reconstruction.run(
            grid=grid,
            sigma=app_state.reconstruction.initial_gaussian_sigma_s,
        )

        self.phase_last = ff_recon.phase.copy()
        return prof_recon, ff_recon

    def _compute_input_profile(self, app_state: AppState) -> CurrentProfile:
        profile_model = ProfileModel(app_state.scenario)
        prof = profile_model.compute_profile()
        self.center_prof.apply(prof)
        return prof

    def _compute_input_formfactor(
        self,
        prof: Profile,
        transform: Transform,
    ) -> FormFactor:
        return prof.to_form_factor(transform=transform)

    def _create_transform(self, recon_state: ReconstructionState) -> Transform:
        if recon_state.band_limit:
            f_cut = float(max(recon_state.band_limit_f_cut_hz, 0.0))
            return BandLimitedDCPhysicalRFFT(
                f_cut=f_cut,
                compact=False,
                unwrap_phase=True,
                dc_normalize=True,
            )

        return DCPhysicalRFFT(
            unwrap_phase=True,
            dc_normalize=True,
        )

    def _build_reconstruction(
        self,
        app_state: AppState,
        measured_ff: MeasuredFormFactor,
    ) -> ReconstructionAlgorithm:
        return GSMeasuredMagnitude(
            measured_ff=measured_ff,
            transform=self._create_transform(app_state.reconstruction),
            time_constraints=self._recon_time_constraints,
            frequency_constraints=self._make_frequency_constraints(app_state),
            stop=self._recon_stop_condition,
            transition_width=app_state.measurement.overlap_width_hz,
            overlap_power=2.0,
            initial_sigma=app_state.reconstruction.initial_gaussian_sigma_s,
        )

    def _make_phase_init(
        self,
        app_state: AppState,
        phase_in: np.ndarray,
    ):
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
        self,
        app_state: AppState,
    ) -> FrequencyConstraint:
        constraints = ClampMagnitude() + EnforceDCOne()

        if app_state.reconstruction.linear_phase_end:
            constraints = constraints + ReplacePhaseEndLinearSmooth(
                start_freq=app_state.reconstruction.linear_phase_end_start_freq_hz,
                freq_x=config.PHASE_END_REF_FREQ_HZ,
                freq_y=app_state.reconstruction.linear_phase_end_phase_at_300_thz,
                transition_width=30e12,
            )

        return constraints

    def _make_measurement_mask(
        self,
        measurement_state: MeasurementState,
        ff_true: FormFactor,
    ) -> np.ndarray:
        freq = ff_true.grid.f_pos
        mode = measurement_state.mode

        if (not measurement_state.enabled) or mode == "full":
            return np.ones_like(freq, dtype=bool)

        if mode == "below_cut":
            f_cut = float(max(measurement_state.f_max_hz, 0.0))
            return freq <= f_cut

        if mode == "between":
            f_min = float(max(measurement_state.f_min_hz, 0.0))
            f_max = float(max(measurement_state.f_max_hz, f_min))
            return (freq >= f_min) & (freq <= f_max)

        raise ValueError(f"Unknown measurement mode: {mode}")

    def _make_measured_formfactor(
        self,
        app_state: AppState,
        ff_true: FormFactor,
    ) -> MeasuredFormFactor:
        mask = self._make_measurement_mask(app_state.measurement, ff_true)

        return MeasuredFormFactor(
            freq=ff_true.grid.f_pos[mask],
            mag=ff_true.mag[mask],
        )
