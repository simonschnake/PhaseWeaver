import numpy as np

from phase_weaver.app.config import (
    CRISP_MIN_HZ,
    CRISP_MAX_HZ,
    IR_MIN_HZ,
    IR_MAX_HZ,
)
from phase_weaver.core import (
    FormFactor,
    Grid,
    Profile,
)
from phase_weaver.core.constraints import (
    CenterFirstMoment,
)
from phase_weaver.core.measurement import MeasuredFormFactor
from phase_weaver.core.reconstruction import (
    GerchbergSaxton,
    ReconstructionAlgorithm,
)

from .plot_model import SpectrumPlotModel, TimePlotModel
from .state import ControlsState, MeasurementState, ProfileModel, ReconstructionState


class AppLogic:
    def __init__(self):
        self.phase_last: np.ndarray | None = None
        self.center_prof = CenterFirstMoment()

    def compute_initial(
        self, controls_state: ControlsState
    ) -> tuple[
        Profile,
        FormFactor,
        tuple[MeasuredFormFactor, ...],
    ]:
        prof_input = self.compute_input_profile(controls_state)
        ff_input = self.compute_input_formfactor(prof_input)
        measurements = self.compute_measured_formfactor(
            ff_input, controls_state.measurement
        )
        return prof_input, ff_input, measurements

    def compute_reconstruction(
        self,
        grid: Grid,
        measurements: tuple[MeasuredFormFactor, ...],
        controls_state: ControlsState,
        ff_input: FormFactor | None,
    ) -> tuple[Profile, FormFactor]:
        reconstruction = GerchbergSaxton(
            grid=grid,
            measurements=measurements,
            reconstruction_state=controls_state.reconstruction,
            formfactor_input=ff_input,
            phase_last=self.phase_last,
        )

        prof_recon, ff_recon = reconstruction.run()

        self.phase_last = ff_recon.phase.copy()
        return prof_recon, ff_recon

    def compute_input_profile(self, app_state: ControlsState) -> Profile:
        profile_model = ProfileModel(app_state.scenario)
        prof = profile_model.compute_profile()
        return prof

    def compute_input_formfactor(
        self,
        prof: Profile,
    ) -> FormFactor:
        return prof.to_form_factor()

    def compute_measured_formfactor(
        self, form_factor: FormFactor, measurement_state: MeasurementState
    ) -> tuple[MeasuredFormFactor, ...]:
        freq = form_factor.grid.f_pos
        mag = form_factor.mag
        if not measurement_state.crisp and not measurement_state.infrared:
            return (MeasuredFormFactor(freq=freq, mag=mag),)

        measured: list[MeasuredFormFactor] = []

        if measurement_state.crisp:
            mask_crisp = (freq >= CRISP_MIN_HZ) & (freq <= CRISP_MAX_HZ)
            freq_crisp = freq[mask_crisp]
            mag_crisp = mag[mask_crisp] * measurement_state.crisp_scale
            meas_crisp = MeasuredFormFactor(freq=freq_crisp, mag=mag_crisp)

            measured.append(meas_crisp)

        if measurement_state.infrared:
            mask_ir = (freq >= IR_MIN_HZ) & (freq <= IR_MAX_HZ)
            freq_ir = freq[mask_ir]
            mag_ir = mag[mask_ir] * measurement_state.infrared_scale
            meas_ir = MeasuredFormFactor(freq=freq_ir, mag=mag_ir)

            measured.append(meas_ir)

        return tuple(measured)

    def _build_reconstruction(
        self,
        grid: Grid,
        measurements: tuple[MeasuredFormFactor, ...],
        recon_state: ReconstructionState,
        form_factor_input: FormFactor,
    ) -> ReconstructionAlgorithm:

        return GerchbergSaxton(
            grid=grid,
            measurements=measurements,
            reconstruction_state=recon_state,
            formfactor_input=form_factor_input,
        )

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
