import numpy as np

from phase_weaver.app.config import (
    MAG_INIT_MODE,
    PHASE_INIT_MODE,
    CRISP_MIN_HZ,
    CRISP_MAX_HZ,
    IR_MIN_HZ,
    IR_MAX_HZ,
)
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
    GerchbergSaxton,
    MaxIter,
    MinimumPhaseCepstrum,
    MinIter,
    PhaseStoppedChanging,
    PredefinedMagnitude,
    PredefinedPhase,
    ReconstructionAlgorithm,
    ZeroPhase,
)

from .plot_model import SpectrumPlotModel, TimePlotModel
from .state import ControlsState, MeasurementState, ProfileModel, ReconstructionState


class AppLogic:
    def __init__(self):
        self.phase_last: np.ndarray | None = None
        self.center_prof = CenterFirstMoment()
        self._recon_time_constraints = CenterFirstMoment() + NormalizeArea()

        self._recon_time_constraints = (
            CutAfterNthZeroFromPeak(n=3)
            + NonNegativity()
            + NormalizeArea()
            + CenterFirstMoment()
        )

        self._recon_frequency_constraints = ClampMagnitude()

        self._recon_stop_condition = (
            MaxIter(1_000) + PhaseStoppedChanging(tol=1e-8, patience=5) + MinIter(10)
        )

    def compute_initial(
        self, controls_state: ControlsState
    ) -> tuple[
        Profile,
        FormFactor,
        MeasuredFormFactor | tuple[MeasuredFormFactor, MeasuredFormFactor] | None,
    ]:
        transform = self._create_transform(controls_state.reconstruction)
        prof_input = self._compute_input_profile(controls_state)
        ff_input = self._compute_input_formfactor(prof_input, transform=transform)
        measurement = self._make_measured_formfactor(controls_state, ff_input=ff_input)
        return prof_input, ff_input, measurement

    def compute_reconstruction(
        self,
        controls_state: ControlsState,
        measured_ff: MeasuredFormFactor
        | tuple[MeasuredFormFactor, MeasuredFormFactor]
        | None,
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

    def _compute_input_profile(self, app_state: ControlsState) -> Profile:
        profile_model = ProfileModel(app_state.scenario)
        prof = profile_model.compute_profile()
        self._recon_time_constraints.apply(prof)
        return prof

    def _compute_input_formfactor(
        self, prof: Profile, transform: Transform | None = None
    ) -> FormFactor:
        return prof.to_form_factor(transform=transform)

    def _create_transform(self, recon_state: ReconstructionState) -> Transform:
        return DCPhysicalRFFT(
            unwrap_phase=True,
            dc_normalize=False,
        )

    def _build_reconstruction(
        self,
        controls_state: ControlsState,
        measured_ff: MeasuredFormFactor
        | tuple[MeasuredFormFactor, MeasuredFormFactor]
        | None,
        ff_input: FormFactor,
    ) -> ReconstructionAlgorithm:
        recon_state = controls_state.reconstruction

        return GerchbergSaxton(
            measured_ff=measured_ff,
            transform=self._create_transform(recon_state),
            formfactor_init=self._make_formfactor_init(recon_state, ff_input),
            time_constraints=self._recon_time_constraints,
            frequency_constraints=self._recon_frequency_constraints,
            # stop=self._recon_stop_condition,
            # transition_width=controls_state.measurement.overlap_width_hz,
            # overlap_power=2.0,
        )

    def _make_formfactor_init(
        self,
        recon_state: ReconstructionState,
        ff_input: FormFactor,
    ):
        phase_mode = recon_state.phase_init_mode

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

        return FormFactorInitializer(phase_init=phase_init)

    def _make_measurement_mask(
        self, form_factor: FormFactor, f_min_hz: float, f_max_hz: float
    ) -> np.ndarray:
        freq = form_factor.grid.f_pos
        return (freq >= f_min_hz) & (freq <= f_max_hz)

    def _make_measured_formfactor(
        self,
        controls_state: ControlsState,
        ff_input: FormFactor,
    ) -> MeasuredFormFactor | tuple[MeasuredFormFactor, MeasuredFormFactor] | None:
        res: (
            MeasuredFormFactor | tuple[MeasuredFormFactor, MeasuredFormFactor] | None
        ) = None
        if controls_state.measurement.crisp:
            mask = self._make_measurement_mask(ff_input, CRISP_MIN_HZ, CRISP_MAX_HZ)
            res = MeasuredFormFactor(
                freq=ff_input.grid.f_pos[mask],
                mag=ff_input.mag[mask] * controls_state.measurement.crisp_scale,
            )
        if controls_state.measurement.infrared:
            mask = self._make_measurement_mask(ff_input, IR_MIN_HZ, IR_MAX_HZ)
            ir_meas = MeasuredFormFactor(
                freq=ff_input.grid.f_pos[mask],
                mag=ff_input.mag[mask] * controls_state.measurement.infrared_scale,
            )
            if res is None:
                res = ir_meas
            else:
                res = (res, ir_meas)

        return res

    def export_npz(
        self,
        time_model: TimePlotModel | None,
        spectrum_model: SpectrumPlotModel | None,
    ) -> None:
        payload = {
            "t": time_model.t_ui if time_model is not None else np.array([]),
            "current_recon": time_model.current_recon_ui
            if time_model is not None
            else np.array([]),
            "current_input": time_model.current_input_ui
            if time_model is not None
            else np.array([]),
            "f": spectrum_model.f_ui if spectrum_model is not None else np.array([]),
            "mag_recon": spectrum_model.mag_recon_ui
            if spectrum_model is not None
            else np.array([]),
            "phase_recon": spectrum_model.phase_recon_ui
            if spectrum_model is not None
            else np.array([]),
            "mag_input": spectrum_model.mag_input_ui
            if spectrum_model is not None
            else np.array([]),
            "phase_input": spectrum_model.phase_input_ui
            if spectrum_model is not None
            else np.array([]),
        }
        np.savez(file="export.npz", **payload)
