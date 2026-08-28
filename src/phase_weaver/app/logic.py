from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from phase_weaver.app.config import (
    CRISP_MIN_HZ,
    CRISP_MAX_HZ,
    CRISP_SIMULATION_MODE,
    IR_SIMULATION_MODE,
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
)
from phase_weaver.core.crisp_simulation import (
    CrispSimulationConfig,
    simulate_crisp_measurement,
)
from phase_weaver.core.measurement import (
    CalibrationStatus,
    Measurement,
    MeasurementKind,
    SquaredMagnitudeMeasurement,
)
from phase_weaver.core.ocean_simulation import (
    OceanSimulationConfig,
    simulate_ocean_measurement,
)
from phase_weaver.core.reconstruction import (
    GerchbergSaxton,
    ReconstructionAlgorithm,
    ReconstructionHistory,
)
from phase_weaver.core.pipeline import CrispThenIrSeed, ReconstructionPipeline

from .loading import MeasurementLoader
from .export import NpzExporter
from .plot_model import SpectrumPlotModel, TimePlotModel
from .simulation import SimulationService
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
    measured: Measurement
    crisp_input: SquaredMagnitudeMeasurement | None = None
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


@dataclass(frozen=True, slots=True)
class ReconstructionRequest:
    grid: Grid
    measurements: tuple[Measurement, ...]
    controls_state: ControlsState
    ff_input: FormFactor | None
    measurement_source: str
    input_profile: Profile | None


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
    signal_std: np.ndarray | None = None,
) -> LoadedMeasurement | None:
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if wavelength_nm.ndim != 1 or signal.ndim != 1:
        raise ValueError("Ocean/NIR wavelength and signal must be 1D arrays")
    if wavelength_nm.shape != signal.shape:
        raise ValueError("Ocean/NIR wavelength and signal must have matching shapes")
    if np.any(~np.isfinite(wavelength_nm)) or np.any(wavelength_nm <= 0.0):
        raise ValueError("Ocean/NIR wavelength must be finite and positive")
    if np.any(~np.isfinite(signal)) or np.any(signal < 0.0):
        raise ValueError("Ocean/NIR signal must be finite and non-negative")
    if signal_std is not None:
        signal_std = np.asarray(signal_std, dtype=float)
        if signal_std.shape != signal.shape:
            raise ValueError("Ocean signal uncertainty must match the signal shape")
    valid = np.ones(wavelength_nm.shape, dtype=bool)
    if signal_std is not None:
        valid &= np.isfinite(signal_std) & (signal_std >= 0.0)
    if not np.any(valid):
        return None

    frequency_hz = (C_NM_THZ / wavelength_nm[valid]) * THZ_TO_HZ
    signal = signal[valid]
    if signal_std is not None:
        signal_std = signal_std[valid]
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

    mag_std = None
    detection_limit = None
    if signal_std is not None:
        band_signal_std = signal_std[band]
        intensity_detection_limit = 3.0 * band_signal_std
        propagation_floor = np.maximum(signal[band], intensity_detection_limit)
        mag_std = (
            0.5
            * band_signal_std
            / np.sqrt(np.maximum(propagation_floor, np.finfo(float).eps))
            / scale
        )
        detection_limit = np.sqrt(intensity_detection_limit) / scale

    return LoadedMeasurement(
        label="Ocean NIR relative |F|",
        measured=Measurement(
            freq=frequency_hz[band],
            mag=np.clip(mag_like / scale, 0.0, 1.0),
            mag_std=mag_std,
            detection_limit=detection_limit,
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
            signal_std = None
            if "map" in keys:
                spectrum_map = np.asarray(data["map"], dtype=float)
                wavelength_size = np.asarray(data["e_axis"]).size
                if spectrum_map.ndim == 2:
                    if spectrum_map.shape[0] == wavelength_size:
                        signal_std = np.nanstd(spectrum_map, axis=1) / np.sqrt(
                            spectrum_map.shape[1]
                        )
                    elif spectrum_map.shape[1] == wavelength_size:
                        signal_std = np.nanstd(spectrum_map, axis=0) / np.sqrt(
                            spectrum_map.shape[0]
                        )
            measurement = _ocean_relative_measurement_from_wavelength_signal(
                data["e_axis"],
                data["average"],
                signal_std=signal_std,
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
            # Cor2d samples are background-corrected and can therefore contain
            # small negative noise values.  They do not represent negative
            # intensity; clamp the averaged signal before taking its square root.
            signal = np.maximum(
                np.nanmean(spectrum_history[:, : len(wavelength_nm)], axis=0),
                0.0,
            )
            measurement = _ocean_relative_measurement_from_wavelength_signal(
                wavelength_nm,
                signal,
                signal_std=(
                    np.nanstd(
                        spectrum_history[:, : len(wavelength_nm)],
                        axis=0,
                    )
                    / np.sqrt(spectrum_history.shape[0])
                ),
            )
            if measurement is None:
                raise ValueError("cor2d npz does not contain usable Ocean/NIR signal")
            return (measurement,)

        if {"freq_hz", "mag"} <= keys:
            label = _decode_label(data["label"], "measurement 1") if "label" in keys else "measurement 1"
            return (
                LoadedMeasurement(
                    label=label,
                    measured=Measurement(freq=data["freq_hz"], mag=data["mag"]),
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
                    measured=Measurement(freq=data[freq_key], mag=data[mag_key]),
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


def _h5_shot_to_measured_formfactor(shot: np.ndarray) -> Measurement:
    freq_thz = np.asarray(shot[:, 0], dtype=float)
    formfactor_squared = np.asarray(shot[:, 1], dtype=float)
    valid = (
        np.isfinite(freq_thz)
        & np.isfinite(formfactor_squared)
        & (formfactor_squared >= 0.0)
    )
    if not np.any(valid):
        raise ValueError("CRISP shot does not contain any finite non-negative points")

    return Measurement(
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
) -> SquaredMagnitudeMeasurement:
    freq_thz = np.asarray(shot[:, 0], dtype=float)
    valid_freq = np.isfinite(freq_thz)
    if not np.any(valid_freq):
        raise ValueError("CRISP shot does not contain any finite frequencies")
    num_valid = int(np.count_nonzero(valid_freq))
    freq_hz = freq_thz[valid_freq] * THZ_TO_HZ
    mag_sq = np.asarray(ffsq_input, dtype=float)[valid_freq]
    return SquaredMagnitudeMeasurement(
        freq=freq_hz,
        mag=mag_sq,
        mag_std=(
            np.asarray(ffsq_std, dtype=float)[valid_freq]
            if ffsq_std is not None
            else np.zeros(num_valid)
        ),
        detection_limit=(
            np.asarray(detection_limit, dtype=float)[valid_freq]
            if detection_limit is not None
            else np.zeros(num_valid)
        ),
        kind=MeasurementKind.CRISP,
        calibration=CalibrationStatus.ABSOLUTE,
        label="CRISP",
        source=f"h5:shot={shot_index}",
        charge_c=charge_c,
        max_frequency_thz=max_frequency_thz,
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
    allow_raw_values: bool = False,
) -> np.ndarray:
    if key not in data:
        return np.zeros(expected_shape[1], dtype=float)

    values = np.asarray(data[key], dtype=float)
    if values.shape != expected_shape:
        raise ValueError(
            f"CRISP dataset {key!r} must have shape {expected_shape}, got {values.shape}"
        )
    selected = values[selected_index]
    if not allow_raw_values and (
        np.any(~np.isfinite(selected)) or np.any(selected < 0.0)
    ):
        raise ValueError(f"CRISP dataset {key!r} must be finite and non-negative")
    return selected


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

        xy = cast(h5py.Dataset, data[CRISP_FORMFACTOR_XY_KEY])
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
    path: str | Path, shot_index: int | None = None, *, include_ocean: bool = True
) -> tuple[LoadedMeasurement, ...]:
    path = Path(path)
    with h5py.File(path, "r") as data:
        if CRISP_FORMFACTOR_XY_KEY not in data:
            raise ValueError(
                f"measurement h5 must contain CRISP dataset {CRISP_FORMFACTOR_XY_KEY!r}"
            )
        if TIMESTAMP_KEY not in data:
            raise ValueError(f"measurement h5 must contain {TIMESTAMP_KEY!r}")

        xy = cast(h5py.Dataset, data[CRISP_FORMFACTOR_XY_KEY])
        timestamps = np.asarray(data[TIMESTAMP_KEY], dtype=float)
        _validate_h5_measurement_data(xy, timestamps)
        num_shots = int(np.asarray(xy).shape[0])

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
            allow_raw_values=True,
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
            num_shots=num_shots,
            charge_c=charge_c,
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
    if include_ocean:
        with h5py.File(path, "r") as data:
            ocean_measurement = _h5_optional_ocean_measurement(
                data,
                selected_index=selected_index,
                num_shots=num_shots,
            )
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


def load_crisp_measurements_file(
    path: str | Path, h5_shot_index: int | None = None
) -> tuple[LoadedMeasurement, ...]:
    path = Path(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return load_measurements_h5(
            path,
            shot_index=h5_shot_index,
            include_ocean=False,
        )
    raise ValueError("CRISP measurements must be loaded from an HDF5 recording")


def load_ir_measurements_file(
    path: str | Path, h5_shot_index: int | None = None
) -> tuple[LoadedMeasurement, ...]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_measurements_npz(path)
    if suffix in {".h5", ".hdf5"}:
        loaded = load_measurements_h5(path, shot_index=h5_shot_index)
        infrared = tuple(item for item in loaded if item.kind == "ocean_nir")
        if not infrared:
            raise ValueError("HDF5 recording does not contain a usable Ocean/NIR IR dataset")
        return infrared
    raise ValueError("IR measurements must be loaded from an Ocean/NIR NPZ or HDF5 file")


class AppLogic:
    def __init__(self):
        self.phase_last: np.ndarray | None = None
        self.center_prof = CenterFirstMoment()
        self.loaded_measurements: tuple[LoadedMeasurement, ...] = ()
        self.reconstruction_summary = ReconstructionSummary()
        self.measurement_loader = MeasurementLoader[LoadedMeasurement, H5MeasurementShot](
            load_measurements_file,
            list_h5_measurement_shots,
            load_crisp_measurements_file,
            load_ir_measurements_file,
        )
        self.npz_exporter = NpzExporter()
        self.simulation_service = SimulationService(
            self._simulate_crisp_detector_raw,
            self._simulate_ocean_detector_raw,
        )
        self._reconstruction_pipeline = ReconstructionPipeline(
            {
                RECONSTRUCTION_ALGORITHM.GERCHBERG_SAXTON: self._run_gs_pipeline,
                RECONSTRUCTION_ALGORITHM.CRISP: self._run_crisp_pipeline,
            }
        )

    def load_measurements(
        self, path: str | Path, h5_shot_index: int | None = None
    ) -> tuple[LoadedMeasurement, ...]:
        self.loaded_measurements = self.measurement_loader.load(
            path,
            h5_shot_index=h5_shot_index,
        )
        self.phase_last = None
        self.reconstruction_summary = ReconstructionSummary(
            measurement_source="loaded",
            measurement_count=len(self.loaded_measurements),
            status="not_run",
        )
        return self.loaded_measurements

    def load_crisp_measurements(
        self, path: str | Path, h5_shot_index: int | None = None
    ) -> tuple[LoadedMeasurement, ...]:
        loaded = self.measurement_loader.load_crisp(
            path,
            h5_shot_index=h5_shot_index,
        )
        self.loaded_measurements = loaded
        self.phase_last = None
        self.reconstruction_summary = ReconstructionSummary(
            measurement_source="loaded",
            measurement_count=len(loaded),
            status="not_run",
        )
        return loaded

    def load_ir_measurements(
        self, path: str | Path, h5_shot_index: int | None = None
    ) -> tuple[LoadedMeasurement, ...]:
        loaded = self.measurement_loader.load_ir(path, h5_shot_index=h5_shot_index)
        replacement = self.measurement_loader.replace_ocean(
            self.loaded_measurements,
            loaded,
            kind=lambda item: item.kind,
        )
        self.loaded_measurements = replacement
        self.phase_last = None
        self.reconstruction_summary = ReconstructionSummary(
            measurement_source="loaded",
            measurement_count=len(replacement),
            status="not_run",
        )
        return replacement

    def replace_loaded_ocean_measurements(
        self, path: str | Path
    ) -> tuple[LoadedMeasurement, ...]:
        return self.load_ir_measurements(path)

    def compute_initial(
        self, controls_state: ControlsState
    ) -> tuple[
        Profile,
        FormFactor,
        tuple[Measurement, ...],
    ]:
        prof_input = self.compute_input_profile(controls_state)
        ff_input = self.compute_input_formfactor(prof_input)
        measurements, _source = self.active_measurements(
            ff_input,
            controls_state.measurement,
            input_profile=prof_input,
        )
        return prof_input, ff_input, measurements

    def compute_reconstruction(
        self,
        grid: Grid,
        measurements: tuple[Measurement, ...],
        controls_state: ControlsState,
        ff_input: FormFactor | None,
        measurement_source: str = "simulated",
        input_profile: Profile | None = None,
    ) -> tuple[Profile, FormFactor, ReconstructionSummary]:
        request = ReconstructionRequest(
            grid=grid,
            measurements=measurements,
            controls_state=controls_state,
            ff_input=ff_input,
            measurement_source=measurement_source,
            input_profile=input_profile,
        )
        return self._reconstruction_pipeline.run(
            controls_state.reconstruction.algorithm,
            request,
        )

    def _run_gs_pipeline(
        self, request: ReconstructionRequest
    ) -> tuple[Profile, FormFactor, ReconstructionSummary]:
        state = replace(
            request.controls_state,
            reconstruction=replace(
                request.controls_state.reconstruction,
                algorithm=RECONSTRUCTION_ALGORITHM.GERCHBERG_SAXTON,
            ),
        )
        return self._compute_reconstruction_legacy(
            request.grid,
            request.measurements,
            state,
            request.ff_input,
            request.measurement_source,
            request.input_profile,
        )

    def _run_crisp_pipeline(
        self, request: ReconstructionRequest
    ) -> tuple[Profile, FormFactor, ReconstructionSummary]:
        state = replace(
            request.controls_state,
            reconstruction=replace(
                request.controls_state.reconstruction,
                algorithm=RECONSTRUCTION_ALGORITHM.CRISP,
            ),
        )
        return self._compute_reconstruction_legacy(
            request.grid,
            request.measurements,
            state,
            request.ff_input,
            request.measurement_source,
            request.input_profile,
        )

    def _compute_reconstruction_legacy(
        self,
        grid: Grid,
        measurements: tuple[Measurement, ...],
        controls_state: ControlsState,
        ff_input: FormFactor | None,
        measurement_source: str = "simulated",
        input_profile: Profile | None = None,
    ) -> tuple[Profile, FormFactor, ReconstructionSummary]:
        algorithm = controls_state.reconstruction.algorithm
        if algorithm == RECONSTRUCTION_ALGORITHM.CRISP:
            crisp_input = self._active_crisp_input(
                measurements,
                controls_state,
                measurement_source=measurement_source,
                input_profile=input_profile,
            )
            reconstruction = CrispReconstruction(crisp_input)
            result = reconstruction.run()
            relative_measurements = self._available_relative_ir_measurements(
                measurement_state=controls_state.measurement,
                form_factor=ff_input,
            )
            has_absolute_ir_measurement = len(measurements) > 1
            if has_absolute_ir_measurement or relative_measurements:
                extension = CrispThenIrSeed(
                    crisp=lambda _request: result,
                    extension=lambda _request, crisp_result: self._run_crisp_ir_extension(
                        crisp_result,
                        measurements=measurements,
                        controls_state=controls_state,
                        measurement_source=measurement_source,
                        relative_measurements=relative_measurements,
                    ),
                )
                profile, form_factor, summary = extension.run(None)
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

        relative_measurements = self.relative_measurements_for_reconstruction(
            controls_state.reconstruction,
            measurement_state=controls_state.measurement,
            form_factor=ff_input,
        )
        reconstruction = GerchbergSaxton(
            grid=grid,
            measurements=measurements,
            reconstruction_state=controls_state.reconstruction,
            formfactor_input=ff_input,
            phase_last=self.phase_last,
            relative_measurements=relative_measurements,
            relative_anchor_formfactor=self.relative_anchor_formfactor(
                controls_state.reconstruction,
                measurement_source=measurement_source,
            ),
            use_formfactor_input_magnitude=not measurements,
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

    def _run_crisp_ir_extension(
        self,
        result,
        *,
        measurements: tuple[Measurement, ...],
        controls_state: ControlsState,
        measurement_source: str,
        relative_measurements: tuple[Measurement, ...],
    ) -> tuple[Profile, FormFactor, ReconstructionSummary]:
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
            algorithm=f"{RECONSTRUCTION_ALGORITHM.CRISP.value} + IR",
            measurement_source=measurement_source,
            measurement_count=len(measurements),
            iterations=result.diagnostics.num_iterations + extension.last_iterations,
            stop_reason=(
                f"crisp:{result.stop_reason}+ir:{extension.last_stop_reason}"
            ),
            measurement_error=extension.last_measurement_error,
            status="finished",
            history=extension.history,
            crisp_diagnostics=result.diagnostics,
            ir_relative_constraint_used=bool(relative_measurements),
            relative_measurement_count=len(relative_measurements),
        )
        return profile, form_factor, summary

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
        self,
        reconstruction_state: ReconstructionState,
        *,
        measurement_state: MeasurementState | None = None,
        form_factor: FormFactor | None = None,
    ) -> tuple[Measurement, ...]:
        if not reconstruction_state.use_ir_relative_constraint:
            return ()

        relative = self._available_relative_ir_measurements(
            measurement_state=measurement_state,
            form_factor=form_factor,
        )
        if relative:
            return relative

        raise ValueError(
            "No relative IR measurement is available. Enable IR with Ocean detector "
            "simulation or load an Ocean/NIR measurement."
        )

    def _available_relative_ir_measurements(
        self,
        *,
        measurement_state: MeasurementState | None = None,
        form_factor: FormFactor | None = None,
    ) -> tuple[Measurement, ...]:
        relative = tuple(
            item.measured
            for item in self.loaded_measurements
            if item.kind == "ocean_nir"
        )
        if relative:
            return relative

        if (
            measurement_state is not None
            and measurement_state.infrared
            and measurement_state.infrared_simulation_mode
            == IR_SIMULATION_MODE.OCEAN
        ):
            if form_factor is None:
                raise ValueError("Ocean detector simulation requires the input form factor")
            return (
                self._simulated_ocean_measurement(form_factor, measurement_state),
            )

        return ()

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
        measurements: tuple[Measurement, ...],
        controls_state: ControlsState,
        *,
        measurement_source: str,
        input_profile: Profile | None,
    ) -> SquaredMagnitudeMeasurement:
        if measurement_source == "loaded":
            for item in self.loaded_measurements:
                if item.crisp_input is not None:
                    return item.crisp_input

        if not measurements:
            raise ValueError("CRISP reconstruction requires at least one measurement")

        if (
            controls_state.measurement.crisp
            and
            controls_state.measurement.crisp_simulation_mode
            == CRISP_SIMULATION_MODE.DETECTOR
        ):
            if input_profile is None:
                raise ValueError("CRISP detector simulation requires the input profile")
            return self._simulated_crisp_input(
                input_profile,
                controls_state.measurement,
            )

        measured = measurements[0]
        positive = measured.freq > 0.0
        # Convert explicitly (never silently): the magnitude measurement is
        # squared here with uncertainty propagated via .to_squared().
        squared = measured.to_squared()
        mag_std = (
            squared.mag_std
            if squared.mag_std is not None
            else np.zeros_like(squared.mag)
        )
        detection_limit = (
            squared.detection_limit
            if squared.detection_limit is not None
            else np.zeros_like(squared.mag)
        )
        return SquaredMagnitudeMeasurement(
            freq=squared.freq[positive],
            mag=squared.mag[positive],
            mag_std=mag_std[positive],
            detection_limit=detection_limit[positive],
            kind=measured.kind,
            calibration=measured.calibration,
            label=measured.label,
            source=f"{measured.source}; crisp_ideal",
            charge_c=controls_state.scenario.charge,
            use_in_absolute_constraint=measured.use_in_absolute_constraint,
            use_in_relative_constraint=measured.use_in_relative_constraint,
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

    def _simulate_crisp_detector_raw(
        self,
        input_profile: Profile,
        measurement_state: MeasurementState,
    ) -> SquaredMagnitudeMeasurement:
        if input_profile.charge is None:
            raise ValueError("CRISP detector simulation requires profile charge")
        return simulate_crisp_measurement(
            input_profile.grid.t,
            input_profile.values * input_profile.charge,
            input_profile.charge,
            CrispSimulationConfig(
                n_shots=measurement_state.crisp_n_shots,
                seed=measurement_state.crisp_noise_seed,
            ),
        )

    def _simulated_crisp_input(
        self,
        input_profile: Profile,
        measurement_state: MeasurementState,
    ) -> SquaredMagnitudeMeasurement:
        simulation = self.simulation_service.simulate_crisp(
            input_profile, measurement_state
        )
        assert input_profile.charge is not None
        scale_sq = measurement_state.crisp_scale**2
        ffsq_std = (
            simulation.mag_std
            if simulation.mag_std is not None
            else np.zeros_like(simulation.mag)
        )
        detection_limit = (
            simulation.detection_limit
            if simulation.detection_limit is not None
            else np.zeros_like(simulation.mag)
        )
        return SquaredMagnitudeMeasurement(
            freq=simulation.freq,
            mag=simulation.mag * scale_sq,
            mag_std=ffsq_std * scale_sq,
            detection_limit=detection_limit * scale_sq,
            kind=simulation.kind,
            calibration=simulation.calibration,
            label=simulation.label,
            source=simulation.source,
            charge_c=input_profile.charge,
        )

    def _simulate_ocean_detector_raw(
        self,
        form_factor: FormFactor,
        measurement_state: MeasurementState,
    ) -> Measurement:
        return simulate_ocean_measurement(
            form_factor.grid.f_pos,
            form_factor.mag,
            OceanSimulationConfig(
                n_shots=measurement_state.infrared_n_shots,
                seed=measurement_state.infrared_noise_seed,
            ),
        )

    def _simulated_ocean_measurement(
        self,
        form_factor: FormFactor,
        measurement_state: MeasurementState,
    ) -> Measurement:
        simulation = self.simulation_service.simulate_ocean(
            form_factor, measurement_state
        )
        band = (
            (simulation.freq >= IR_MIN_HZ)
            & (simulation.freq <= IR_MAX_HZ)
        )
        return Measurement(
            freq=simulation.freq[band],
            mag=(
                simulation.mag[band]
                * measurement_state.infrared_scale
            ),
            mag_std=(
                simulation.mag_std[band]
                * measurement_state.infrared_scale
                if simulation.mag_std is not None
                else None
            ),
            detection_limit=(
                simulation.detection_limit[band]
                * measurement_state.infrared_scale
                if simulation.detection_limit is not None
                else None
            ),
        )

    def compute_measured_formfactor(
        self,
        form_factor: FormFactor,
        measurement_state: MeasurementState,
        *,
        input_profile: Profile | None = None,
    ) -> tuple[Measurement, ...]:
        freq = form_factor.grid.f_pos
        mag = form_factor.mag
        if not measurement_state.crisp and not measurement_state.infrared:
            return (Measurement(freq=freq, mag=mag),)

        measured: list[Measurement] = []

        if measurement_state.crisp:
            if measurement_state.crisp_simulation_mode == CRISP_SIMULATION_MODE.DETECTOR:
                if input_profile is None:
                    raise ValueError("CRISP detector simulation requires the input profile")
                simulation = self.simulation_service.simulate_crisp(
                    input_profile,
                    measurement_state,
                )
                freq_crisp = simulation.freq
                mag_crisp = simulation.as_magnitude().mag * measurement_state.crisp_scale
            else:
                mask_crisp = (freq >= CRISP_MIN_HZ) & (freq <= CRISP_MAX_HZ)
                freq_crisp = freq[mask_crisp]
                mag_crisp = mag[mask_crisp] * measurement_state.crisp_scale
            meas_crisp = Measurement(freq=freq_crisp, mag=mag_crisp)

            measured.append(meas_crisp)

        if measurement_state.infrared:
            if measurement_state.infrared_simulation_mode == IR_SIMULATION_MODE.OCEAN:
                meas_ir = self._simulated_ocean_measurement(
                    form_factor,
                    measurement_state,
                )
            else:
                mask_ir = (freq >= IR_MIN_HZ) & (freq <= IR_MAX_HZ)
                freq_ir = freq[mask_ir]
                mag_ir = mag[mask_ir] * measurement_state.infrared_scale
                meas_ir = Measurement(freq=freq_ir, mag=mag_ir)

            measured.append(meas_ir)

        return tuple(measured)

    def active_measurements(
        self,
        form_factor: FormFactor,
        measurement_state: MeasurementState,
        *,
        input_profile: Profile | None = None,
    ) -> tuple[tuple[Measurement, ...], str]:
        if self.loaded_measurements:
            return (
                tuple(
                    item.measured
                    for item in self.loaded_measurements
                    if item.use_in_reconstruction
                ),
                "loaded",
            )
        measurements = self.compute_measured_formfactor(
            form_factor,
            measurement_state,
            input_profile=input_profile,
        )
        if (
            measurement_state.infrared
            and measurement_state.infrared_simulation_mode
            == IR_SIMULATION_MODE.OCEAN
        ):
            # The Ocean result is an uncalibrated relative measurement.  Keep
            # it out of the absolute magnitude constraint.
            measurements = measurements[:-1]
        return measurements, "simulated"

    def visible_measurements(
        self,
        form_factor: FormFactor,
        measurement_state: MeasurementState,
        *,
        input_profile: Profile | None = None,
    ) -> tuple[LoadedMeasurement, ...]:
        if self.loaded_measurements:
            return self.loaded_measurements

        visible: list[LoadedMeasurement] = []
        freq = form_factor.grid.f_pos
        mag = form_factor.mag

        if measurement_state.crisp:
            if measurement_state.crisp_simulation_mode == CRISP_SIMULATION_MODE.DETECTOR:
                if input_profile is None:
                    raise ValueError("CRISP detector simulation requires the input profile")
                simulation = self.simulation_service.simulate_crisp(
                    input_profile,
                    measurement_state,
                )
                crisp_measurement = Measurement(
                    freq=simulation.freq,
                    mag=simulation.as_magnitude().mag * measurement_state.crisp_scale,
                )
                crisp_label = "CRISP detector simulation"
            else:
                mask_crisp = (freq >= CRISP_MIN_HZ) & (freq <= CRISP_MAX_HZ)
                crisp_measurement = Measurement(
                    freq=freq[mask_crisp],
                    mag=mag[mask_crisp] * measurement_state.crisp_scale,
                )
                crisp_label = "CRISP"
            visible.append(
                LoadedMeasurement(
                    label=crisp_label,
                    measured=crisp_measurement,
                    kind="crisp",
                    calibration="simulated_calibrated"
                    if measurement_state.crisp_simulation_mode
                    == CRISP_SIMULATION_MODE.DETECTOR
                    else "simulated_ideal",
                )
            )

        if measurement_state.infrared:
            if measurement_state.infrared_simulation_mode == IR_SIMULATION_MODE.OCEAN:
                ir_measurement = self._simulated_ocean_measurement(
                    form_factor,
                    measurement_state,
                )
                ir_label = "Ocean NIR detector simulation"
                ir_calibration = "simulated_relative_shape"
                ir_kind = "ocean_nir"
                ir_use_in_reconstruction = False
            else:
                mask_ir = (freq >= IR_MIN_HZ) & (freq <= IR_MAX_HZ)
                ir_measurement = Measurement(
                    freq=freq[mask_ir],
                    mag=mag[mask_ir] * measurement_state.infrared_scale,
                )
                ir_label = "IR"
                ir_calibration = "simulated_ideal"
                ir_kind = "infrared"
                ir_use_in_reconstruction = True
            visible.append(
                LoadedMeasurement(
                    label=ir_label,
                    measured=ir_measurement,
                    kind=ir_kind,
                    calibration=ir_calibration,
                    use_in_reconstruction=ir_use_in_reconstruction,
                )
            )

        return tuple(visible)

    def _build_reconstruction(
        self,
        grid: Grid,
        measurements: tuple[Measurement, ...],
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
        self.npz_exporter.export_npz(
            path,
            time_model,
            spectrum_model,
            summary=summary,
            controls_state=controls_state,
            loaded_measurements=self.loaded_measurements,
            crisp_diagnostics_payload=self._crisp_diagnostics_payload,
            state_payload=self._state_payload,
        )

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
            "crisp_intermediate_ffsq_std": diagnostics.intermediate_ffsq_std,
            "crisp_interpolated_frequencies_thz": (
                diagnostics.interpolated_frequencies_thz
            ),
            "crisp_interpolated_ffabs": diagnostics.interpolated_ffabs,
            "crisp_interpolated_ffabs_error": (
                diagnostics.interpolated_ffabs_error
            ),
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
        measurement = controls_state.measurement
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
            "measurement_crisp_enabled": np.array(measurement.crisp),
            "measurement_infrared_enabled": np.array(measurement.infrared),
            "measurement_crisp_simulation_mode": np.array(
                measurement.crisp_simulation_mode.value
            ),
            "measurement_crisp_n_shots": np.array(measurement.crisp_n_shots),
            "measurement_crisp_noise_seed": np.array(
                measurement.crisp_noise_seed
            ),
            "measurement_infrared_simulation_mode": np.array(
                measurement.infrared_simulation_mode.value
            ),
            "measurement_infrared_n_shots": np.array(
                measurement.infrared_n_shots
            ),
            "measurement_infrared_noise_seed": np.array(
                measurement.infrared_noise_seed
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
