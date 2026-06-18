import numpy as np
import pytest
from numpy.testing import assert_allclose

from phase_weaver.app.config import PHASE_INIT_MODE, RECONSTRUCTION_ALGORITHM
from phase_weaver.app.logic import (
    AppLogic,
    CRISP_CHARGE_KEY,
    CRISP_FORMFACTOR_XY_KEY,
    ReconstructionSummary,
    TIMESTAMP_KEY,
    list_h5_measurement_shots,
    load_measurements_file,
    load_measurements_h5,
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


def test_load_measurements_h5_uses_latest_timestamped_crisp_row(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "measurement.h5"
    with h5py.File(path, "w") as data:
        data.create_dataset(TIMESTAMP_KEY, data=np.array([100.0, 300.0, 200.0]))
        data.create_dataset(
            CRISP_FORMFACTOR_XY_KEY,
            data=np.array(
                [
                    [[0.1, 0.01], [0.2, 0.04]],
                    [[0.3, 0.09], [0.4, 0.16]],
                    [[0.5, 0.25], [0.6, 0.36]],
                ],
                dtype=np.float32,
            ),
        )

    measurements = load_measurements_h5(path)

    assert len(measurements) == 1
    assert measurements[0].label == "CRISP latest 1970-01-01T00:05:00+00:00"
    assert_allclose(measurements[0].measured.freq, [0.3e12, 0.4e12])
    assert_allclose(measurements[0].measured.mag, [0.3, 0.4])


def test_load_measurements_h5_builds_crisp_input_with_charge(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "measurement.h5"
    with h5py.File(path, "w") as data:
        data.create_dataset(TIMESTAMP_KEY, data=np.array([100.0, 300.0]))
        data.create_dataset(CRISP_CHARGE_KEY, data=np.array([0.1, 0.25]))
        data.create_dataset(
            CRISP_FORMFACTOR_XY_KEY,
            data=np.array(
                [
                    [[0.1, 0.01], [0.2, 0.04]],
                    [[0.3, 0.09], [0.4, -0.16]],
                ],
                dtype=np.float32,
            ),
        )

    measurements = load_measurements_h5(path)

    crisp_input = measurements[0].crisp_input
    assert crisp_input is not None
    assert crisp_input.shot_index == 1
    assert crisp_input.charge_c == pytest.approx(0.25e-9)
    assert_allclose(crisp_input.freq_hz, [0.3e12, 0.4e12])
    assert_allclose(crisp_input.ffsq, [0.09, -0.16])


def test_load_measurements_h5_skips_negative_crisp_points(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "measurement.h5"
    with h5py.File(path, "w") as data:
        data.create_dataset(TIMESTAMP_KEY, data=np.array([100.0]))
        data.create_dataset(
            CRISP_FORMFACTOR_XY_KEY,
            data=np.array(
                [
                    [[0.1, 0.01], [0.2, -1.0], [0.3, 0.09]],
                ],
                dtype=np.float32,
            ),
        )

    measurements = load_measurements_h5(path)

    assert_allclose(measurements[0].measured.freq, [0.1e12, 0.3e12])
    assert_allclose(measurements[0].measured.mag, [0.1, 0.3])


def test_list_h5_measurement_shots_returns_available_timestamps(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "measurement.h5"
    with h5py.File(path, "w") as data:
        data.create_dataset(TIMESTAMP_KEY, data=np.array([100.0, 300.0]))
        data.create_dataset(
            CRISP_FORMFACTOR_XY_KEY,
            data=np.zeros((2, 2, 2), dtype=np.float32),
        )

    shots = list_h5_measurement_shots(path)

    assert [shot.index for shot in shots] == [0, 1]
    assert [shot.measured_at for shot in shots] == [
        "1970-01-01T00:01:40+00:00",
        "1970-01-01T00:05:00+00:00",
    ]


def test_load_measurements_h5_uses_selected_timestamped_crisp_row(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "measurement.h5"
    with h5py.File(path, "w") as data:
        data.create_dataset(TIMESTAMP_KEY, data=np.array([100.0, 300.0, 200.0]))
        data.create_dataset(
            CRISP_FORMFACTOR_XY_KEY,
            data=np.array(
                [
                    [[0.1, 0.01], [0.2, 0.04]],
                    [[0.3, 0.09], [0.4, 0.16]],
                    [[0.5, 0.25], [0.6, 0.36]],
                ],
                dtype=np.float32,
            ),
        )

    measurements = load_measurements_h5(path, shot_index=2)

    assert measurements[0].label == "CRISP 1970-01-01T00:03:20+00:00"
    assert_allclose(measurements[0].measured.freq, [0.5e12, 0.6e12])
    assert_allclose(measurements[0].measured.mag, [0.5, 0.6])


def test_load_measurements_h5_rejects_invalid_shot_index(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "measurement.h5"
    with h5py.File(path, "w") as data:
        data.create_dataset(TIMESTAMP_KEY, data=np.array([100.0]))
        data.create_dataset(
            CRISP_FORMFACTOR_XY_KEY,
            data=np.zeros((1, 2, 2), dtype=np.float32),
        )

    with pytest.raises(ValueError, match="shot index"):
        load_measurements_h5(path, shot_index=3)


def test_load_measurements_h5_rejects_missing_crisp_dataset(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "bad_measurement.h5"
    with h5py.File(path, "w") as data:
        data.create_dataset(TIMESTAMP_KEY, data=np.array([100.0]))

    with pytest.raises(ValueError, match="CRISP dataset"):
        load_measurements_h5(path)


def test_load_measurements_file_dispatches_by_suffix(tmp_path):
    path = tmp_path / "measurement.npz"
    np.savez(path, freq_hz=np.array([1.0]), mag=np.array([0.5]))

    measurements = load_measurements_file(path)

    assert_allclose(measurements[0].measured.freq, [1.0])


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


def test_app_logic_runs_crisp_algorithm_from_simulated_crisp_band():
    logic = AppLogic()
    state = ControlsState(
        scenario=ProfileModelState(),
        measurement=MeasurementState(crisp=True),
        reconstruction=ReconstructionState(
            algorithm=RECONSTRUCTION_ALGORITHM.CRISP,
        ),
    )

    prof_input, ff_input, measurements = logic.compute_initial(state)
    prof_recon, ff_recon, summary = logic.compute_reconstruction(
        grid=prof_input.grid,
        measurements=measurements,
        controls_state=state,
        ff_input=ff_input,
    )

    assert summary.algorithm == "CRISP"
    assert summary.status == "finished"
    assert summary.crisp_diagnostics is not None
    assert prof_recon.grid.N == 1024
    assert ff_recon.mag.shape == (513,)
    assert summary.iterations <= 20


def test_export_npz_includes_crisp_diagnostics(tmp_path):
    logic = AppLogic()
    state = ControlsState(
        scenario=ProfileModelState(),
        measurement=MeasurementState(crisp=True),
        reconstruction=ReconstructionState(
            algorithm=RECONSTRUCTION_ALGORITHM.CRISP,
        ),
    )
    prof_input, ff_input, measurements = logic.compute_initial(state)
    prof_recon, ff_recon, summary = logic.compute_reconstruction(
        grid=prof_input.grid,
        measurements=measurements,
        controls_state=state,
        ff_input=ff_input,
    )
    time_model = TimePlotModel(profile_input=prof_input, profile_recon=prof_recon)
    spectrum_model = SpectrumPlotModel(
        formfactor_input=ff_input,
        formfactor_recon=ff_recon,
    )
    path = tmp_path / "export.npz"

    logic.export_npz(path, time_model, spectrum_model, controls_state=state, summary=summary)

    with np.load(path) as data:
        assert data["reconstruction_algorithm"] == "CRISP"
        assert data["crisp_num_iterations"] == summary.iterations
        assert data["crisp_interpolated_ffabs"].shape == (512,)


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
