from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
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

CRISP_FORMFACTOR_XY_KEY = (
    "XFEL.SDIAG__THZ_SPECTROMETER.FORMFACTOR__CRD.1934.TL__FORMFACTOR.XY"
)
TIMESTAMP_KEY = "timestamp"
THZ_TO_HZ = 1e12


@dataclass(slots=True)
class LoadedMeasurement:
    label: str
    measured: MeasuredFormFactor


@dataclass(frozen=True, slots=True)
class H5MeasurementShot:
    index: int
    timestamp: float
    measured_at: str


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


def _validate_h5_measurement_data(
    xy: h5py.Dataset, timestamps: np.ndarray
) -> None:
    if xy.ndim != 3 or xy.shape[2] != 2:
        raise ValueError(
            f"CRISP dataset must have shape (shots, points, 2), got {xy.shape}"
        )
    if timestamps.ndim != 1:
        raise ValueError("timestamp must be a 1D array")
    if xy.shape[0] != len(timestamps):
        raise ValueError("CRISP dataset shot count must match timestamp array length")
    if len(timestamps) == 0:
        raise ValueError("measurement h5 must contain at least one timestamp")
    if np.any(~np.isfinite(timestamps)):
        raise ValueError("timestamp must contain only finite values")


def _format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def list_h5_measurement_shots(path: str | Path) -> tuple[H5MeasurementShot, ...]:
    path = Path(path)
    with h5py.File(path, "r") as data:
        if CRISP_FORMFACTOR_XY_KEY not in data:
            raise ValueError(
                f"measurement h5 must contain CRISP dataset {CRISP_FORMFACTOR_XY_KEY!r}"
            )
        if TIMESTAMP_KEY not in data:
            raise ValueError(f"measurement h5 must contain {TIMESTAMP_KEY!r}")

        xy = data[CRISP_FORMFACTOR_XY_KEY]
        timestamps = np.asarray(data[TIMESTAMP_KEY], dtype=float)
        _validate_h5_measurement_data(xy, timestamps)

    return tuple(
        H5MeasurementShot(
            index=index,
            timestamp=float(timestamp),
            measured_at=_format_timestamp(float(timestamp)),
        )
        for index, timestamp in enumerate(timestamps)
    )


def load_measurements_h5(
    path: str | Path, shot_index: int | None = None
) -> tuple[LoadedMeasurement, ...]:
    path = Path(path)
    with h5py.File(path, "r") as data:
        if CRISP_FORMFACTOR_XY_KEY not in data:
            raise ValueError(
                f"measurement h5 must contain CRISP dataset {CRISP_FORMFACTOR_XY_KEY!r}"
            )
        if TIMESTAMP_KEY not in data:
            raise ValueError(f"measurement h5 must contain {TIMESTAMP_KEY!r}")

        xy = data[CRISP_FORMFACTOR_XY_KEY]
        timestamps = np.asarray(data[TIMESTAMP_KEY], dtype=float)
        _validate_h5_measurement_data(xy, timestamps)

        selected_index = int(np.argmax(timestamps)) if shot_index is None else shot_index
        if selected_index < 0 or selected_index >= len(timestamps):
            raise ValueError(
                f"shot index must be between 0 and {len(timestamps) - 1}, got {shot_index}"
            )
        shot = np.asarray(xy[selected_index], dtype=float)
        timestamp = float(timestamps[selected_index])

    measured_at = _format_timestamp(timestamp)
    label = f"CRISP latest {measured_at}" if shot_index is None else f"CRISP {measured_at}"
    return (
        LoadedMeasurement(
            label=label,
            measured=MeasuredFormFactor(
                freq=shot[:, 0] * THZ_TO_HZ,
                mag=np.sqrt(shot[:, 1]),
            ),
        ),
    )


def load_measurements_file(
    path: str | Path, h5_shot_index: int | None = None
) -> tuple[LoadedMeasurement, ...]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_measurements_npz(path)
    if suffix in {".h5", ".hdf5"}:
        return load_measurements_h5(path, shot_index=h5_shot_index)
    raise ValueError("measurement file must be a .npz, .h5, or .hdf5 file")


class AppLogic:
    def __init__(self):
        self.phase_last: np.ndarray | None = None
        self.center_prof = CenterFirstMoment()
        self.loaded_measurements: tuple[LoadedMeasurement, ...] = ()
        self.reconstruction_summary = ReconstructionSummary()

    def load_measurements(
        self, path: str | Path, h5_shot_index: int | None = None
    ) -> tuple[LoadedMeasurement, ...]:
        self.loaded_measurements = load_measurements_file(
            path, h5_shot_index=h5_shot_index
        )
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

    def visible_measurements(
        self, form_factor: FormFactor, measurement_state: MeasurementState
    ) -> tuple[LoadedMeasurement, ...]:
        if self.loaded_measurements:
            return self.loaded_measurements

        visible: list[LoadedMeasurement] = []
        freq = form_factor.grid.f_pos
        mag = form_factor.mag

        if measurement_state.crisp:
            mask_crisp = (freq >= CRISP_MIN_HZ) & (freq <= CRISP_MAX_HZ)
            visible.append(
                LoadedMeasurement(
                    label="CRISP",
                    measured=MeasuredFormFactor(
                        freq=freq[mask_crisp],
                        mag=mag[mask_crisp] * measurement_state.crisp_scale,
                    ),
                )
            )

        if measurement_state.infrared:
            mask_ir = (freq >= IR_MIN_HZ) & (freq <= IR_MAX_HZ)
            visible.append(
                LoadedMeasurement(
                    label="IR",
                    measured=MeasuredFormFactor(
                        freq=freq[mask_ir],
                        mag=mag[mask_ir] * measurement_state.infrared_scale,
                    ),
                )
            )

        return tuple(visible)

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
