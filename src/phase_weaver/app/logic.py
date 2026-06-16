from dataclasses import dataclass
from pathlib import Path

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
    ReconstructionHistory,
)

from .plot_model import SpectrumPlotModel, TimePlotModel
from .state import ControlsState, MeasurementState, ProfileModel, ReconstructionState


@dataclass(slots=True)
class LoadedMeasurement:
    label: str
    measured: MeasuredFormFactor


@dataclass(slots=True)
class ReconstructionSummary:
    measurement_source: str = "simulated"
    measurement_count: int = 0
    iterations: int = 0
    stop_reason: str = "not_run"
    measurement_error: float | None = None
    status: str = "not_run"
    history: ReconstructionHistory | None = None


def _decode_label(value: object, fallback: str) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)
    return fallback


def load_measurements_npz(path: str | Path) -> tuple[LoadedMeasurement, ...]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)

        if {"freq_hz", "mag"} <= keys:
            label = _decode_label(data["label"], "measurement 1") if "label" in keys else "measurement 1"
            return (
                LoadedMeasurement(
                    label=label,
                    measured=MeasuredFormFactor(freq=data["freq_hz"], mag=data["mag"]),
                ),
            )

        indices: set[int] = set()
        for key in keys:
            if key.startswith("freq_hz_"):
                suffix = key.removeprefix("freq_hz_")
                if suffix.isdigit():
                    indices.add(int(suffix))
            elif key.startswith("mag_"):
                suffix = key.removeprefix("mag_")
                if suffix.isdigit():
                    indices.add(int(suffix))

        if not indices:
            raise ValueError(
                "measurement npz must contain freq_hz/mag or indexed freq_hz_N/mag_N arrays"
            )

        measurements: list[LoadedMeasurement] = []
        for index in sorted(indices):
            freq_key = f"freq_hz_{index}"
            mag_key = f"mag_{index}"
            if freq_key not in keys or mag_key not in keys:
                raise ValueError(
                    f"measurement {index} must contain both {freq_key!r} and {mag_key!r}"
                )

            label_key = f"label_{index}"
            label = (
                _decode_label(data[label_key], f"measurement {index + 1}")
                if label_key in keys
                else f"measurement {index + 1}"
            )
            measurements.append(
                LoadedMeasurement(
                    label=label,
                    measured=MeasuredFormFactor(freq=data[freq_key], mag=data[mag_key]),
                )
            )

    return tuple(measurements)


class AppLogic:
    def __init__(self):
        self.phase_last: np.ndarray | None = None
        self.center_prof = CenterFirstMoment()
        self.loaded_measurements: tuple[LoadedMeasurement, ...] = ()
        self.reconstruction_summary = ReconstructionSummary()

    def load_measurements(self, path: str | Path) -> tuple[LoadedMeasurement, ...]:
        self.loaded_measurements = load_measurements_npz(path)
        self.phase_last = None
        self.reconstruction_summary = ReconstructionSummary(
            measurement_source="loaded",
            measurement_count=len(self.loaded_measurements),
            status="not_run",
        )
        return self.loaded_measurements

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
        measurement_source: str = "simulated",
    ) -> tuple[Profile, FormFactor, ReconstructionSummary]:
        reconstruction = GerchbergSaxton(
            grid=grid,
            measurements=measurements,
            reconstruction_state=controls_state.reconstruction,
            formfactor_input=ff_input,
            phase_last=self.phase_last,
        )

        prof_recon, ff_recon = reconstruction.run()

        self.phase_last = ff_recon.phase.copy()
        summary = ReconstructionSummary(
            measurement_source=measurement_source,
            measurement_count=len(measurements),
            iterations=reconstruction.last_iterations,
            stop_reason=reconstruction.last_stop_reason,
            measurement_error=reconstruction.last_measurement_error,
            status="finished",
            history=reconstruction.history,
        )
        self.reconstruction_summary = summary
        return prof_recon, ff_recon, summary

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

    def active_measurements(
        self, form_factor: FormFactor, measurement_state: MeasurementState
    ) -> tuple[tuple[MeasuredFormFactor, ...], str]:
        if self.loaded_measurements:
            return tuple(item.measured for item in self.loaded_measurements), "loaded"
        return self.compute_measured_formfactor(form_factor, measurement_state), "simulated"

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
        path: str | Path,
        time_model: TimePlotModel | None,
        spectrum_model: SpectrumPlotModel | None,
        controls_state: ControlsState | None = None,
        summary: ReconstructionSummary | None = None,
    ) -> None:
        summary = summary or self.reconstruction_summary
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
            "measurement_source": np.array(summary.measurement_source),
            "measurement_count": np.array(summary.measurement_count),
            "reconstruction_status": np.array(summary.status),
            "reconstruction_iterations": np.array(summary.iterations),
            "reconstruction_stop_reason": np.array(summary.stop_reason),
            "reconstruction_measurement_error": np.array(
                np.nan
                if summary.measurement_error is None
                else summary.measurement_error
            ),
        }
        if summary.history is not None:
            payload.update(summary.history.as_arrays())

        if controls_state is not None:
            payload.update(self._state_payload(controls_state))

        for i, item in enumerate(self.loaded_measurements):
            payload[f"measurement_label_{i}"] = np.array(item.label)
            payload[f"measurement_freq_hz_{i}"] = item.measured.freq
            payload[f"measurement_mag_{i}"] = item.measured.mag

        np.savez(file=path, **payload)

    def _state_payload(self, controls_state: ControlsState) -> dict[str, np.ndarray]:
        scenario = controls_state.scenario
        reconstruction = controls_state.reconstruction
        payload: dict[str, np.ndarray] = {
            "phase_init_mode": np.array(reconstruction.phase_init_mode.value),
            "reconstruction_time_constraints": np.array(
                sorted(option.value for option in reconstruction.time_constraints)
            ),
            "reconstruction_frequency_constraints": np.array(
                sorted(option.value for option in reconstruction.frequency_constraints)
            ),
            "reconstruction_stop_conditions": np.array(
                sorted(option.value for option in reconstruction.stop_conditions)
            ),
            "profile_dt_s": np.array(scenario.dt),
            "profile_t_max_s": np.array(scenario.t_max),
            "profile_charge_c": np.array(scenario.charge),
            "profile_peak2_enabled": np.array(scenario.peak2_enabled),
        }
        for prefix, params in (
            ("background", scenario.background),
            ("peak", scenario.peak),
            ("peak2", scenario.peak2),
        ):
            payload[f"{prefix}_center_s"] = np.array(params.center)
            payload[f"{prefix}_width_s"] = np.array(params.width)
            payload[f"{prefix}_skew"] = np.array(params.skew)
            payload[f"{prefix}_order"] = np.array(params.order)
            payload[f"{prefix}_amplitude"] = np.array(params.amplitude)
        return payload
