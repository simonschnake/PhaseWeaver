from tokenize import Exponent

import numpy as np

from phase_weaver.app.config import MAG_INIT_MODE, MEASUREMENT_MODE, PHASE_INIT_MODE
from phase_weaver.core import (
    BandLimitedDCPhysicalRFFT,
    CurrentProfile,
    DCPhysicalRFFT,
    FormFactor,
    Profile,
    Transform,
)
from phase_weaver.core.constraints import (
    CenterFirstMoment,
    ClampMagnitude,
    CutAfterNthZeroFromPeak,
    NonNegativity,
    NormalizeArea,
)
from phase_weaver.core.measurement import MeasuredFormFactor
from phase_weaver.core.reconstruction import (
    ExponentialExtendMeasurement,
    ExponentialInitializer,
    FormFactorInitializer,
    GSMeasuredMagnitude,
    MaxIter,
    MinimumPhaseCepstrum,
    MinIter,
    PhaseStoppedChanging,
    PredefinedMagnitude,
    PredefinedPhase,
    ReconstructionAlgorithm,
    ZeroPhase,
)

from .state import ControlsState, MeasurementState, ProfileModel, ReconstructionState


class AppLogic:
    def __init__(self):
        self.phase_last: np.ndarray | None = None
        self.center_prof = CenterFirstMoment()

        self._recon_time_constraints = (
            CutAfterNthZeroFromPeak()
            + NonNegativity()
            # + NormalizeArea()
            + CenterFirstMoment()
        )

        self._recon_frequency_constraints = ClampMagnitude()

        self._recon_stop_condition = (
            MaxIter(1_000) + PhaseStoppedChanging(tol=1e-8, patience=5) + MinIter(10)
        )

    def compute_initial(
        self, controls_state: ControlsState
    ) -> tuple[Profile, FormFactor, MeasuredFormFactor | None]:
        transform = self._create_transform(controls_state.reconstruction)
        prof_input = self._compute_input_profile(controls_state)
        ff_input = self._compute_input_formfactor(prof_input, transform=transform)
        measurement = self._make_measured_formfactor(controls_state, ff_input=ff_input)
        return prof_input, ff_input, measurement

    def compute_reconstruction(
        self,
        controls_state: ControlsState,
        measured_ff: MeasuredFormFactor | None,
        ff_input: FormFactor,
    ) -> tuple[Profile, FormFactor]:
        reconstruction = self._build_reconstruction(
            controls_state, measured_ff, ff_input
        )

        prof_recon, ff_recon = reconstruction.run(
            grid=ff_input.grid, mag=ff_input.mag, measured_ff=measured_ff
        )

        self.phase_last = ff_recon.phase.copy()
        return prof_recon, ff_recon

    def _compute_input_profile(self, app_state: ControlsState) -> CurrentProfile:
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
                dc_normalize=False,
            )

        return DCPhysicalRFFT(
            unwrap_phase=True,
            dc_normalize=False,
        )

    def _build_reconstruction(
        self,
        controls_state: ControlsState,
        measured_ff: MeasuredFormFactor | None,
        ff_input: FormFactor,
    ) -> ReconstructionAlgorithm:
        recon_state = controls_state.reconstruction

        return GSMeasuredMagnitude(
            measured_ff=measured_ff,
            transform=self._create_transform(recon_state),
            formfactor_init=self._make_formfactor_init(recon_state, ff_input),
            time_constraints=self._recon_time_constraints,
            frequency_constraints=self._recon_frequency_constraints,
            # stop=self._recon_stop_condition,
            transition_width=controls_state.measurement.overlap_width_hz,
            overlap_power=2.0,
        )

    def _make_formfactor_init(
        self,
        recon_state: ReconstructionState,
        ff_input: FormFactor,
    ):
        mag_mode = recon_state.mag_init_mode
        phase_mode = recon_state.phase_init_mode

        if mag_mode == MAG_INIT_MODE.EXP:
            mag_init = ExponentialInitializer()
        elif mag_mode == MAG_INIT_MODE.REAL:
            mag_init = PredefinedMagnitude(mag=ff_input.mag)
        elif mag_mode == MAG_INIT_MODE.MEASEXT:
            mag_init = ExponentialExtendMeasurement()
        elif mag_mode == MAG_INIT_MODE.CRISP:
            print(
                "Warning: CRISP magnitude initialization is not implemented. Falling back to exponential initialization."
            )
            mag_init = ExponentialInitializer()

        if phase_mode == PHASE_INIT_MODE.ZERO:
            phase_init = ZeroPhase()
        elif phase_mode == PHASE_INIT_MODE.REAL:
            phase_init = PredefinedPhase(phase=ff_input.phase)
        elif phase_mode == PHASE_INIT_MODE.MINPHASE:
            phase_init = MinimumPhaseCepstrum()
        elif phase_mode == PHASE_INIT_MODE.LAST:
            phase = self.phase_last if self.phase_last is not None else ff_input.phase
            phase_init = PredefinedPhase(phase=phase)
        elif phase_mode == "CRISP":
            print(
                "Warning: CRISP phase initialization is not implemented. Falling back to minimum phase cepstrum initialization."
            )
            phase_init = PredefinedPhase(phase=ff_input.phase)

        return FormFactorInitializer(mag_init=mag_init, phase_init=phase_init)

    def _make_measurement_mask(
        self,
        measurement_state: MeasurementState,
        ff_true: FormFactor,
    ) -> np.ndarray:
        freq = ff_true.grid.f_pos
        mode = measurement_state.mode

        if mode == MEASUREMENT_MODE.FULL:
            return np.ones_like(freq, dtype=bool)

        if mode == MEASUREMENT_MODE.BELOW_CUT:
            f_cut = float(max(measurement_state.f_max_hz, 0.0))
            return freq <= f_cut

        if mode == MEASUREMENT_MODE.BETWEEN:
            f_min = float(max(measurement_state.f_min_hz, 0.0))
            f_max = float(max(measurement_state.f_max_hz, f_min))
            return (freq >= f_min) & (freq <= f_max)

        raise ValueError(f"Unknown measurement mode: {mode}")

    def _make_measured_formfactor(
        self,
        controls_state: ControlsState,
        ff_input: FormFactor,
    ) -> MeasuredFormFactor | None:
        if controls_state.measurement.mode == "full":
            return None
        mask = self._make_measurement_mask(controls_state.measurement, ff_input)

        if not np.any(mask):
            print("Warning: Measurement mask is empty. No measurements will be made.")
            return None

        scale = controls_state.measurement.scale
        if scale is None:
            scale = 1.0

        return MeasuredFormFactor(
            freq=ff_input.grid.f_pos[mask],
            mag=ff_input.mag[mask] * scale,
        )
