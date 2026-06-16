import numpy as np
import pytest
from numpy.testing import assert_allclose

from phase_weaver.app.config import PHASE_INIT_MODE
from phase_weaver.app.logic import (
    AppLogic,
    ReconstructionSummary,
    load_measurements_npz,
)
from phase_weaver.app.plot_model import SpectrumPlotModel, TimePlotModel
from phase_weaver.app.state import (
    ControlsState,
    MeasurementState,
    ProfileModel,
    ProfileModelState,
    ReconstructionState,
)
from phase_weaver.core.reconstruction import ReconstructionHistory


def test_load_measurements_npz_single_pair(tmp_path):
    path = tmp_path / "measurement.npz"
    np.savez(path, freq_hz=np.array([3.0, 1.0]), mag=np.array([30.0, 10.0]))

    measurements = load_measurements_npz(path)

    assert len(measurements) == 1
    assert measurements[0].label == "measurement 1"
    assert_allclose(measurements[0].measured.freq, [1.0, 3.0])
    assert_allclose(measurements[0].measured.mag, [10.0, 30.0])


def test_load_measurements_npz_multiple_pairs_with_labels(tmp_path):
    path = tmp_path / "measurements.npz"
    np.savez(
        path,
        freq_hz_0=np.array([1.0, 2.0]),
        mag_0=np.array([0.8, 0.6]),
        label_0=np.array("CRISP"),
        freq_hz_1=np.array([3.0, 4.0]),
        mag_1=np.array([0.4, 0.2]),
        label_1=np.array("IR"),
    )

    measurements = load_measurements_npz(path)

    assert [item.label for item in measurements] == ["CRISP", "IR"]
    assert_allclose(measurements[1].measured.freq, [3.0, 4.0])
    assert_allclose(measurements[1].measured.mag, [0.4, 0.2])


def test_load_measurements_npz_rejects_missing_pair(tmp_path):
    path = tmp_path / "bad_measurements.npz"
    np.savez(path, freq_hz_0=np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="both 'freq_hz_0' and 'mag_0'"):
        load_measurements_npz(path)


def test_load_measurements_npz_rejects_invalid_arrays(tmp_path):
    path = tmp_path / "bad_measurements.npz"
    np.savez(path, freq_hz=np.array([1.0, 2.0]), mag=np.array([np.nan, 1.0]))

    with pytest.raises(ValueError, match="finite"):
        load_measurements_npz(path)


def test_app_logic_uses_loaded_measurements_over_simulated(tmp_path):
    path = tmp_path / "measurement.npz"
    np.savez(path, freq_hz=np.array([10.0, 20.0]), mag=np.array([0.9, 0.7]))

    logic = AppLogic()
    logic.load_measurements(path)

    state = ControlsState(
        scenario=ProfileModelState(),
        measurement=MeasurementState(crisp=True, infrared=True),
        reconstruction=ReconstructionState(phase_init_mode=PHASE_INIT_MODE.ZERO),
    )
    prof = ProfileModel(state.scenario).compute_profile()
    ff = prof.to_form_factor()

    measurements, source = logic.active_measurements(ff, state.measurement)

    assert source == "loaded"
    assert len(measurements) == 1
    assert_allclose(measurements[0].freq, [10.0, 20.0])


def test_app_logic_visible_measurements_follow_active_bands():
    logic = AppLogic()
    state = ControlsState(
        scenario=ProfileModelState(),
        measurement=MeasurementState(crisp=True, infrared=True),
        reconstruction=ReconstructionState(phase_init_mode=PHASE_INIT_MODE.ZERO),
    )
    prof = ProfileModel(state.scenario).compute_profile()
    ff = prof.to_form_factor()

    visible = logic.visible_measurements(ff, state.measurement)

    assert [item.label for item in visible] == ["CRISP", "IR"]
    assert np.all(visible[0].measured.freq <= 60e12)
    assert np.all(visible[1].measured.freq >= 120e12)


def test_app_logic_visible_measurements_prefer_loaded(tmp_path):
    path = tmp_path / "measurement.npz"
    np.savez(
        path,
        freq_hz_0=np.array([10.0, 20.0]),
        mag_0=np.array([0.9, 0.7]),
        label_0=np.array("Loaded CRISP"),
        freq_hz_1=np.array([130.0, 150.0]),
        mag_1=np.array([0.6, 0.5]),
        label_1=np.array("Loaded IR"),
    )

    logic = AppLogic()
    logic.load_measurements(path)
    state = ControlsState(
        scenario=ProfileModelState(),
        measurement=MeasurementState(crisp=True, infrared=True),
        reconstruction=ReconstructionState(phase_init_mode=PHASE_INIT_MODE.ZERO),
    )
    prof = ProfileModel(state.scenario).compute_profile()
    ff = prof.to_form_factor()

    visible = logic.visible_measurements(ff, state.measurement)

    assert [item.label for item in visible] == ["Loaded CRISP", "Loaded IR"]
    assert_allclose(visible[0].measured.freq, [10.0, 20.0])


def test_app_logic_export_writes_measurements_and_summary(tmp_path):
    logic = AppLogic()
    measurement_path = tmp_path / "measurement.npz"
    np.savez(
        measurement_path,
        freq_hz=np.array([10.0, 20.0]),
        mag=np.array([0.9, 0.7]),
    )
    logic.load_measurements(measurement_path)

    state = ControlsState(
        scenario=ProfileModelState(),
        measurement=MeasurementState(),
        reconstruction=ReconstructionState(phase_init_mode=PHASE_INIT_MODE.ZERO),
    )
    prof = ProfileModel(state.scenario).compute_profile()
    ff = prof.to_form_factor()
    time_model = TimePlotModel(profile_input=prof)
    spectrum_model = SpectrumPlotModel(formfactor_input=ff)
    summary = ReconstructionSummary(
        measurement_source="loaded",
        measurement_count=1,
        iterations=12,
        stop_reason="max_iter",
        measurement_error=0.123,
        status="finished",
        history=ReconstructionHistory(
            iterations=[1, 2],
            measurement_error=[0.3, 0.123],
            phase_delta_mse=[1e-2, 1e-4],
            profile_delta_mse=[2e-2, 2e-4],
        ),
    )

    export_path = tmp_path / "export.npz"
    logic.export_npz(
        export_path,
        time_model,
        spectrum_model,
        controls_state=state,
        summary=summary,
    )

    with np.load(export_path, allow_pickle=False) as data:
        assert data["measurement_source"].item() == "loaded"
        assert data["measurement_count"].item() == 1
        assert data["reconstruction_iterations"].item() == 12
        assert data["measurement_label_0"].item() == "measurement 1"
        assert_allclose(data["measurement_freq_hz_0"], [10.0, 20.0])
        assert_allclose(data["reconstruction_history_iteration"], [1, 2])
        assert_allclose(
            data["reconstruction_history_measurement_error"], [0.3, 0.123]
        )
        assert_allclose(data["reconstruction_history_phase_delta_mse"], [1e-2, 1e-4])
        assert_allclose(
            data["reconstruction_history_profile_delta_mse"], [2e-2, 2e-4]
        )
        assert data["phase_init_mode"].item() == "zero"
        assert "Normalize area" in set(data["reconstruction_time_constraints"])
        assert "Blend measured" in set(data["reconstruction_frequency_constraints"])
        assert "Measurement error" in set(data["reconstruction_stop_conditions"])
