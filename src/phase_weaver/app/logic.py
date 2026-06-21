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
    PHASE_INIT_MODE,
    RECONSTRUCTION_ALGORITHM,
)
from phase_weaver.core import (
    FormFactor,
    Grid,
    Profile,
)
from phase_weaver.core.constraints import (
    CenterFirstMoment,
)
from phase_weaver.core.crisp_reconstruction import (
    CrispDiagnostics,
    CrispReconstruction,
    CrispReconstructionConfig,
    CrispReconstructionInput,
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
CRISP_INPUT_FFSQ_KEY = (
    "XFEL.SDIAG__THZ_SPECTROMETER.RECONSTRUCTION__CRD.1934.TL.SA1__INPUT_FFSQ"
)
CRISP_INPUT_FFSQ_STD_KEY = (
    "XFEL.SDIAG__THZ_SPECTROMETER.RECONSTRUCTION__CRD.1934.TL.SA1__INPUT_FFSQ_STD"
)
CRISP_INPUT_FFSQ_DETECTION_LIMIT_KEY = (
    "XFEL.SDIAG__THZ_SPECTROMETER.RECONSTRUCTION__CRD.1934.TL.SA1__INPUT_FFSQ_DETECTION_LIMIT"
)
CRISP_SA1_CURRENT_PROFILE_KEY = (
    "XFEL.SDIAG__THZ_SPECTROMETER.RECONSTRUCTION__CRD.1934.TL.SA1__CURRENT_PROFILE"
)
CRISP_CHARGE_KEY = "XFEL.DIAG__BPM__BPMA.2218.T2__CHARGE.ALL"
OCEAN_SPECTRUM_KEY = "XFEL.DIAG__SPECTROMETER__SPEC.2219.T2__SPECTRUM"
TIMESTAMP_KEY = "timestamp"
THZ_TO_HZ = 1e12
C_NM_THZ = 299792.458
OCEAN_NORMALIZATION_PERCENTILE = 95.0


@dataclass(slots=True)
class ReferenceCurrentProfile:
    label: str
    time_s: np.ndarray
    current_a: np.ndarray
    inferred_max_frequency_thz: float


@dataclass(slots=True)
class LoadedMeasurement:
    label: str
    measured: MeasuredFormFactor
    crisp_input: CrispReconstructionInput | None = None
    reference_current: ReferenceCurrentProfile | None = None
    kind: str = "unknown"
    calibration: str = "unknown"
    use_in_reconstruction: bool = True


@dataclass(frozen=True, slots=True)
class H5MeasurementShot:
    index: int
    timestamp: float
    measured_at: str


@dataclass(slots=True)
class ReconstructionSummary:
    algorithm: str = "Gerchberg-Saxton"
    measurement_source: str = "simulated"
    measurement_count: int = 0
    iterations: int = 0
    stop_reason: str = "not_run"
    measurement_error: float | None = None
    status: str = "not_run"
    history: ReconstructionHistory | None = None
    crisp_diagnostics: CrispDiagnostics | None = None
    ir_relative_constraint_used: bool = False
    relative_measurement_count: int = 0


def _decode_label(value: object, fallback: str) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)
    return fallback


def _ocean_relative_measurement_from_wavelength_signal(
    wavelength_nm: np.ndarray,
    signal: np.ndarray,
) -> LoadedMeasurement | None:
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    signal = np.maximum(np.asarray(signal, dtype=float), 0.0)
    valid = (
        np.isfinite(wavelength_nm)
        & np.isfinite(signal)
        & (wavelength_nm > 0.0)
    )
    if not np.any(valid):
        return None

    frequency_hz = (C_NM_THZ / wavelength_nm[valid]) * THZ_TO_HZ
    signal = signal[valid]
    band = (
        np.isfinite(frequency_hz)
        & (frequency_hz >= IR_MIN_HZ)
        & (frequency_hz <= IR_MAX_HZ)
        & (signal > 0.0)
    )
    if not np.any(band):
        return None

    mag_like = np.sqrt(signal[band])
    scale = float(np.percentile(mag_like, OCEAN_NORMALIZATION_PERCENTILE))
    if not np.isfinite(scale) or scale <= 0.0:
        return None

    return LoadedMeasurement(
        label="Ocean NIR relative |F|",
        measured=MeasuredFormFactor(
            freq=frequency_hz[band],
            mag=np.clip(mag_like / scale, 0.0, 1.0),
        ),
        kind="ocean_nir",
        calibration="relative_shape",
        use_in_reconstruction=False,
    )


def load_measurements_npz(path: str | Path) -> tuple[LoadedMeasurement, ...]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)

        if {"e_axis", "average"} <= keys:
            measurement = _ocean_relative_measurement_from_wavelength_signal(
                data["e_axis"],
                data["average"],
            )
            if measurement is None:
                raise ValueError("waterflow npz does not contain usable Ocean/NIR signal")
            return (measurement,)

        if {"phen_scale", "spec_hist"} <= keys:
            spectrum_history = np.asarray(data["spec_hist"], dtype=float)
            wavelength_nm = np.asarray(data["phen_scale"], dtype=float)
            if spectrum_history.ndim != 2:
                raise ValueError("cor2d npz spec_hist must be a 2D array")
            if spectrum_history.shape[1] < len(wavelength_nm):
                raise ValueError("cor2d npz spec_hist must cover phen_scale length")
            measurement = _ocean_relative_measurement_from_wavelength_signal(
                wavelength_nm,
                np.nanmean(spectrum_history[:, : len(wavelength_nm)], axis=0),
            )
            if measurement is None:
                raise ValueError("cor2d npz does not contain usable Ocean/NIR signal")
            return (measurement,)

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


def _h5_shot_to_measured_formfactor(shot: np.ndarray) -> MeasuredFormFactor:
    freq_thz = shot[:, 0]
    formfactor_squared = shot[:, 1]
    valid = (
        np.isfinite(freq_thz)
        & np.isfinite(formfactor_squared)
        & (formfactor_squared >= 0.0)
    )
    if not np.any(valid):
        raise ValueError("CRISP shot does not contain any finite non-negative points")

    return MeasuredFormFactor(
        freq=freq_thz[valid] * THZ_TO_HZ,
        mag=np.sqrt(formfactor_squared[valid]),
    )


def _h5_shot_to_crisp_input(
    shot: np.ndarray,
    *,
    ffsq_input: np.ndarray,
    ffsq_std: np.ndarray,
    detection_limit: np.ndarray,
    charge_c: float,
    max_frequency_thz: float | None,
    shot_index: int,
    timestamp: float,
) -> CrispReconstructionInput:
    freq_thz = np.asarray(shot[:, 0], dtype=float)
    valid_freq = np.isfinite(freq_thz)
    if not np.any(valid_freq):
        raise ValueError("CRISP shot does not contain any finite frequencies")
    return CrispReconstructionInput(
        freq_hz=freq_thz[valid_freq] * THZ_TO_HZ,
        ffsq=ffsq_input[valid_freq],
        ffsq_std=ffsq_std[valid_freq],
        detection_limit=detection_limit[valid_freq],
        charge_c=charge_c,
        max_frequency_thz=max_frequency_thz,
        shot_index=shot_index,
        timestamp=timestamp,
    )


def _h5_charge_c(data: h5py.File, selected_index: int) -> float:
    if CRISP_CHARGE_KEY not in data:
        return 250e-12
    charge_nc = np.asarray(data[CRISP_CHARGE_KEY], dtype=float)
    if charge_nc.ndim != 1 or selected_index >= len(charge_nc):
        return 250e-12
    value = float(charge_nc[selected_index])
    if not np.isfinite(value) or value <= 0.0:
        return 250e-12
    return value * 1e-9


def _h5_optional_shot_array(
    data: h5py.File,
    key: str,
    *,
    selected_index: int,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    if key not in data:
        return np.zeros(expected_shape[1], dtype=float)

    values = np.asarray(data[key], dtype=float)
    if values.shape != expected_shape:
        raise ValueError(
            f"CRISP dataset {key!r} must have shape {expected_shape}, got {values.shape}"
        )
    return values[selected_index]


def _h5_optional_reference_current(
    data: h5py.File,
    *,
    selected_index: int,
    num_shots: int,
    charge_c: float,
) -> ReferenceCurrentProfile | None:
    if CRISP_SA1_CURRENT_PROFILE_KEY not in data:
        return None

    values = np.asarray(data[CRISP_SA1_CURRENT_PROFILE_KEY], dtype=float)
    if values.ndim != 2 or values.shape[0] != num_shots:
        raise ValueError(
            f"CRISP dataset {CRISP_SA1_CURRENT_PROFILE_KEY!r} must have shape "
            f"({num_shots}, points), got {values.shape}"
        )

    current_a = values[selected_index]
    if not np.all(np.isfinite(current_a)):
        raise ValueError("CRISP SA1 current profile must contain only finite values")

    current_sum = float(np.sum(current_a))
    if current_sum <= 0.0:
        raise ValueError("CRISP SA1 current profile must have positive area")

    dt_s = charge_c / current_sum
    max_frequency_thz = 0.5 / dt_s / THZ_TO_HZ
    config = CrispReconstructionConfig(
        num_output_points=len(current_a),
        max_frequency_thz=max_frequency_thz,
    )
    time_s = Grid(N=len(current_a), dt=config.dt_s).t
    return ReferenceCurrentProfile(
        label="CRISP SA1",
        time_s=time_s,
        current_a=current_a,
        inferred_max_frequency_thz=max_frequency_thz,
    )


def _h5_optional_ocean_measurement(
    data: h5py.File,
    *,
    selected_index: int,
    num_shots: int,
) -> LoadedMeasurement | None:
    if OCEAN_SPECTRUM_KEY not in data:
        return None

    spectrum = np.asarray(data[OCEAN_SPECTRUM_KEY], dtype=float)
    if spectrum.ndim != 3 or spectrum.shape[2] != 2 or spectrum.shape[0] != num_shots:
        raise ValueError(
            f"Ocean spectrum dataset {OCEAN_SPECTRUM_KEY!r} must have shape "
            f"({num_shots}, points, 2), got {spectrum.shape}"
        )

    shot = spectrum[selected_index]
    return _ocean_relative_measurement_from_wavelength_signal(
        shot[:, 0],
        shot[:, 1],
    )


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
        expected_optional_shape = (xy.shape[0], xy.shape[1])
        ffsq_input = _h5_optional_shot_array(
            data,
            CRISP_INPUT_FFSQ_KEY,
            selected_index=selected_index,
            expected_shape=expected_optional_shape,
        )
        if CRISP_INPUT_FFSQ_KEY not in data:
            ffsq_input = np.asarray(shot[:, 1], dtype=float)
        ffsq_std = _h5_optional_shot_array(
            data,
            CRISP_INPUT_FFSQ_STD_KEY,
            selected_index=selected_index,
            expected_shape=expected_optional_shape,
        )
        detection_limit = _h5_optional_shot_array(
            data,
            CRISP_INPUT_FFSQ_DETECTION_LIMIT_KEY,
            selected_index=selected_index,
            expected_shape=expected_optional_shape,
        )
        timestamp = float(timestamps[selected_index])
        charge_c = _h5_charge_c(data, selected_index)
        reference_current = _h5_optional_reference_current(
            data,
            selected_index=selected_index,
            num_shots=xy.shape[0],
            charge_c=charge_c,
        )
        ocean_measurement = _h5_optional_ocean_measurement(
            data,
            selected_index=selected_index,
            num_shots=xy.shape[0],
        )

    measured_at = _format_timestamp(timestamp)
    label = f"CRISP latest {measured_at}" if shot_index is None else f"CRISP {measured_at}"
    loaded = [
        LoadedMeasurement(
            label=label,
            measured=_h5_shot_to_measured_formfactor(shot),
            crisp_input=_h5_shot_to_crisp_input(
                shot,
                ffsq_input=ffsq_input,
                ffsq_std=ffsq_std,
                detection_limit=detection_limit,
                charge_c=charge_c,
                max_frequency_thz=(
                    reference_current.inferred_max_frequency_thz
                    if reference_current is not None
                    else None
                ),
                shot_index=selected_index,
                timestamp=timestamp,
            ),
            reference_current=reference_current,
            kind="crisp",
            calibration="absolute_or_calibrated",
        ),
    ]
    if ocean_measurement is not None:
        loaded.append(ocean_measurement)
    return tuple(loaded)


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

    def replace_loaded_ocean_measurements(
        self, path: str | Path
    ) -> tuple[LoadedMeasurement, ...]:
        loaded = load_measurements_file(path)
        ocean_measurements = tuple(
            item for item in loaded if item.kind == "ocean_nir"
        )
        if not ocean_measurements:
            raise ValueError("measurement file does not contain Ocean/NIR data")

        preserved = tuple(
            item for item in self.loaded_measurements if item.kind != "ocean_nir"
        )
        self.loaded_measurements = (*preserved, *ocean_measurements)
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
        algorithm = controls_state.reconstruction.algorithm
        if algorithm == RECONSTRUCTION_ALGORITHM.CRISP:
            crisp_input = self._active_crisp_input(
                measurements,
                controls_state,
                measurement_source=measurement_source,
            )
            reconstruction = CrispReconstruction(crisp_input)
            result = reconstruction.run()
            relative_measurements = self.relative_measurements_for_reconstruction(
                controls_state.reconstruction
            )
            if relative_measurements:
                extension_state = self._crisp_extension_reconstruction_state(
                    controls_state.reconstruction
                )
                extension = GerchbergSaxton(
                    grid=result.form_factor.grid,
                    measurements=measurements,
                    reconstruction_state=extension_state,
                    formfactor_input=result.form_factor,
                    phase_last=result.form_factor.phase,
                    relative_measurements=relative_measurements,
                    relative_anchor_formfactor=result.form_factor,
                    use_formfactor_input_magnitude=True,
                )
                profile, form_factor = extension.run()
                profile.charge = result.profile.charge
                summary = ReconstructionSummary(
                    algorithm=f"{algorithm.value} + IR",
                    measurement_source=measurement_source,
                    measurement_count=len(measurements),
                    iterations=result.diagnostics.num_iterations
                    + extension.last_iterations,
                    stop_reason=(
                        f"crisp:{result.stop_reason}"
                        f"+ir:{extension.last_stop_reason}"
                    ),
                    measurement_error=extension.last_measurement_error,
                    status="finished",
                    history=extension.history,
                    crisp_diagnostics=result.diagnostics,
                    ir_relative_constraint_used=True,
                    relative_measurement_count=len(relative_measurements),
                )
                self.phase_last = form_factor.phase.copy()
                self.reconstruction_summary = summary
                return profile, form_factor, summary

            summary = ReconstructionSummary(
                algorithm=algorithm.value,
                measurement_source=measurement_source,
                measurement_count=len(measurements),
                iterations=result.diagnostics.num_iterations,
                stop_reason=result.stop_reason,
                measurement_error=None,
                status="finished",
                crisp_diagnostics=result.diagnostics,
            )
            self.phase_last = result.form_factor.phase.copy()
            self.reconstruction_summary = summary
            return result.profile, result.form_factor, summary

        reconstruction = GerchbergSaxton(
            grid=grid,
            measurements=measurements,
            reconstruction_state=controls_state.reconstruction,
            formfactor_input=ff_input,
            phase_last=self.phase_last,
            relative_measurements=self.relative_measurements_for_reconstruction(
                controls_state.reconstruction
            ),
            relative_anchor_formfactor=self.relative_anchor_formfactor(
                controls_state.reconstruction,
                measurement_source=measurement_source,
            ),
        )

        prof_recon, ff_recon = reconstruction.run()

        self.phase_last = ff_recon.phase.copy()
        summary = ReconstructionSummary(
            algorithm=algorithm.value,
            measurement_source=measurement_source,
            measurement_count=len(measurements),
            iterations=reconstruction.last_iterations,
            stop_reason=reconstruction.last_stop_reason,
            measurement_error=reconstruction.last_measurement_error,
            status="finished",
            history=reconstruction.history,
            ir_relative_constraint_used=bool(reconstruction.relative_measurements),
            relative_measurement_count=len(reconstruction.relative_measurements),
        )
        self.reconstruction_summary = summary
        return prof_recon, ff_recon, summary

    def _crisp_extension_reconstruction_state(
        self, reconstruction_state: ReconstructionState
    ) -> ReconstructionState:
        return ReconstructionState(
            algorithm=RECONSTRUCTION_ALGORITHM.GERCHBERG_SAXTON,
            phase_init_mode=PHASE_INIT_MODE.LAST,
            use_ir_relative_constraint=True,
            use_fixed_ir_scale=reconstruction_state.use_fixed_ir_scale,
            fixed_ir_scale=reconstruction_state.fixed_ir_scale,
            time_constraints=set(reconstruction_state.time_constraints),
            frequency_constraints=set(reconstruction_state.frequency_constraints),
            stop_conditions=set(reconstruction_state.stop_conditions),
        )

    def relative_measurements_for_reconstruction(
        self, reconstruction_state: ReconstructionState
    ) -> tuple[MeasuredFormFactor, ...]:
        if not reconstruction_state.use_ir_relative_constraint:
            return ()

        relative = tuple(
            item.measured
            for item in self.loaded_measurements
            if item.kind == "ocean_nir"
        )
        if not relative:
            raise ValueError("No IR measurement loaded for extended CRISP reconstruction")
        return relative

    def relative_anchor_formfactor(
        self,
        reconstruction_state: ReconstructionState,
        *,
        measurement_source: str,
    ) -> FormFactor | None:
        if not reconstruction_state.use_ir_relative_constraint:
            return None
        if measurement_source != "loaded":
            return None
        for item in self.loaded_measurements:
            if item.crisp_input is not None:
                return CrispReconstruction(item.crisp_input).run().form_factor
        return None

    def _active_crisp_input(
        self,
        measurements: tuple[MeasuredFormFactor, ...],
        controls_state: ControlsState,
        *,
        measurement_source: str,
    ) -> CrispReconstructionInput:
        if measurement_source == "loaded":
            for item in self.loaded_measurements:
                if item.crisp_input is not None:
                    return item.crisp_input

        if not measurements:
            raise ValueError("CRISP reconstruction requires at least one measurement")

        measured = measurements[0]
        positive = measured.freq > 0.0
        return CrispReconstructionInput(
            freq_hz=measured.freq[positive],
            ffsq=np.square(measured.mag[positive]),
            ffsq_std=np.zeros(np.count_nonzero(positive), dtype=float),
            detection_limit=np.zeros(np.count_nonzero(positive), dtype=float),
            charge_c=controls_state.scenario.charge,
        )

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
            return (
                tuple(
                    item.measured
                    for item in self.loaded_measurements
                    if item.use_in_reconstruction
                ),
                "loaded",
            )
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
            relative_measurements=self.relative_measurements_for_reconstruction(
                recon_state
            ),
            relative_anchor_formfactor=self.relative_anchor_formfactor(
                recon_state,
                measurement_source="loaded" if self.loaded_measurements else "simulated",
            ),
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
            "reconstruction_algorithm": np.array(summary.algorithm),
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
            "reconstruction_ir_relative_constraint_used": np.array(
                summary.ir_relative_constraint_used
            ),
            "reconstruction_relative_measurement_count": np.array(
                summary.relative_measurement_count
            ),
        }
        if summary.history is not None:
            payload.update(summary.history.as_arrays())
        if summary.crisp_diagnostics is not None:
            payload.update(self._crisp_diagnostics_payload(summary.crisp_diagnostics))

        if controls_state is not None:
            payload.update(self._state_payload(controls_state))

        for i, item in enumerate(self.loaded_measurements):
            payload[f"measurement_label_{i}"] = np.array(item.label)
            payload[f"measurement_freq_hz_{i}"] = item.measured.freq
            payload[f"measurement_mag_{i}"] = item.measured.mag
            payload[f"measurement_kind_{i}"] = np.array(item.kind)
            payload[f"measurement_calibration_{i}"] = np.array(item.calibration)
            payload[f"measurement_use_in_reconstruction_{i}"] = np.array(
                item.use_in_reconstruction
            )
            if item.reference_current is not None:
                payload[f"measurement_reference_current_label_{i}"] = np.array(
                    item.reference_current.label
                )
                payload[f"measurement_reference_current_time_s_{i}"] = (
                    item.reference_current.time_s
                )
                payload[f"measurement_reference_current_a_{i}"] = (
                    item.reference_current.current_a
                )
                payload[f"measurement_reference_current_max_frequency_thz_{i}"] = (
                    np.array(item.reference_current.inferred_max_frequency_thz)
                )

        np.savez(file=path, **payload)

    def _crisp_diagnostics_payload(
        self, diagnostics: CrispDiagnostics
    ) -> dict[str, np.ndarray]:
        payload: dict[str, np.ndarray] = {
            "crisp_kramers_kronig_profile": diagnostics.kramers_kronig_profile,
            "crisp_kramers_kronig_phase": diagnostics.kramers_kronig_phase,
            "crisp_intermediate_frequencies_thz": (
                diagnostics.intermediate_frequencies_thz
            ),
            "crisp_intermediate_ffsq": diagnostics.intermediate_ffsq,
            "crisp_interpolated_frequencies_thz": (
                diagnostics.interpolated_frequencies_thz
            ),
            "crisp_interpolated_ffabs": diagnostics.interpolated_ffabs,
            "crisp_num_input_points": np.array(diagnostics.num_input_points),
            "crisp_num_filtered_input_points": np.array(
                diagnostics.num_filtered_input_points
            ),
            "crisp_max_input_frequency_thz": np.array(
                diagnostics.max_input_frequency_thz
            ),
            "crisp_num_iterations": np.array(diagnostics.num_iterations),
            "crisp_peak_current_a": np.array(diagnostics.peak_current_a),
            "crisp_fwhm_fs": np.array(diagnostics.fwhm_fs),
            "crisp_rms_width_fs": np.array(diagnostics.rms_width_fs),
            "crisp_skewness": np.array(diagnostics.skewness),
        }
        for i, profile in enumerate(diagnostics.iteration_profiles, start=1):
            payload[f"crisp_current_profile_{i}"] = profile
        return payload

    def _state_payload(self, controls_state: ControlsState) -> dict[str, np.ndarray]:
        scenario = controls_state.scenario
        reconstruction = controls_state.reconstruction
        payload: dict[str, np.ndarray] = {
            "reconstruction_algorithm": np.array(reconstruction.algorithm.value),
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
            "reconstruction_use_ir_relative_constraint": np.array(
                reconstruction.use_ir_relative_constraint
            ),
            "reconstruction_use_fixed_ir_scale": np.array(
                reconstruction.use_fixed_ir_scale
            ),
            "reconstruction_fixed_ir_scale": np.array(reconstruction.fixed_ir_scale),
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
