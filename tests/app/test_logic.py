import numpy as np
import pytest
from numpy.testing import assert_allclose

from phase_weaver.app.config import (
    CRISP_SIMULATION_MODE,
    IR_SIMULATION_MODE,
    IR_MAX_HZ,
    IR_MIN_HZ,
    PHASE_INIT_MODE,
    RECONSTRUCTION_ALGORITHM,
)
from phase_weaver.app.logic import (
    AppLogic,
    CRISP_CHARGE_KEY,
    CRISP_FORMFACTOR_XY_KEY,
    CRISP_INPUT_FFSQ_DETECTION_LIMIT_KEY,
    CRISP_INPUT_FFSQ_KEY,
    CRISP_INPUT_FFSQ_STD_KEY,
    CRISP_SA1_CURRENT_PROFILE_KEY,
    LoadedMeasurement,
    OCEAN_SPECTRUM_KEY,
    ReferenceCurrentProfile,
    ReconstructionSummary,
    TIMESTAMP_KEY,
    list_h5_measurement_shots,
    load_measurements_file,
    load_measurements_h5,
    load_measurements_npz,
    load_crisp_measurements_file,
    load_ir_measurements_file,
)
from phase_weaver.core.crisp_reconstruction import (
    CrispReconstruction,
)
from phase_weaver.core.measurement import Measurement, SquaredMagnitudeMeasurement
from phase_weaver.app.plot_model import SpectrumPlotModel, TimePlotModel
from phase_weaver.app.state import (
    ControlsState,
    MeasurementState,
    ProfileModel,
    ProfileModelState,
    ReconstructionState,
)
from phase_weaver.core.reconstruction import ReconstructionHistory
from phase_weaver.core.pipeline import ReconstructionPipeline
from phase_weaver.app.loading import MeasurementLoader
from phase_weaver.app.export import NpzExporter
from phase_weaver.app.simulation import SimulationService


def test_load_measurements_npz_single_pair(tmp_path):
    path = tmp_path / "measurement.npz"
    np.savez(path, freq_hz=np.array([3.0, 1.0]), mag=np.array([30.0, 10.0]))

    measurements = load_measurements_npz(path)

    assert len(measurements) == 1
    assert measurements[0].label == "measurement 1"
    assert_allclose(measurements[0].measured.freq, [1.0, 3.0])
    assert_allclose(measurements[0].measured.mag, [10.0, 30.0])


def test_measurement_loader_delegates_loading_and_ocean_replacement():
    calls: list[tuple[str, int | None]] = []

    def load(path, shot_index):
        calls.append((str(path), shot_index))
        return ("crisp", "ocean_nir")

    loader = MeasurementLoader(load, lambda path: ("shot",))

    assert loader.load("recording.h5", h5_shot_index=2) == ("crisp", "ocean_nir")
    assert loader.list_shots("recording.h5") == ("shot",)
    assert loader.replace_ocean(
        ("old_crisp", "old_ocean_nir"),
        ("new_ocean_nir",),
        kind=lambda item: item.split("_", 1)[-1],
    ) == ("old_crisp", "new_ocean_nir")
    assert calls == [("recording.h5", 2)]


def test_measurement_loader_exposes_type_specific_contracts():
    loader = MeasurementLoader(
        lambda path, shot_index: {
            "crisp.npz": (LoadedMeasurement("CRISP", Measurement([1.0], [1.0]), kind="crisp"),),
            "ir.npz": (LoadedMeasurement("IR", Measurement([1.0], [1.0]), kind="ocean_nir"),),
        }[str(path)],
        lambda path: (),
    )

    assert loader.load_crisp("crisp.npz")[0].kind == "crisp"
    assert loader.load_ir("ir.npz")[0].kind == "ocean_nir"

    with pytest.raises(ValueError, match="CRISP"):
        loader.load_crisp("ir.npz")
    with pytest.raises(ValueError, match="IR"):
        loader.load_ir("crisp.npz")


def test_npz_exporter_writes_plot_arrays_and_summary(tmp_path):
    exporter = NpzExporter()
    path = tmp_path / "export.npz"
    summary = ReconstructionSummary(algorithm="CRISP", status="finished")

    exporter.export_npz(path, None, None, summary=summary)

    with np.load(path) as data:
        assert data["reconstruction_algorithm"].item() == "CRISP"
        assert data["reconstruction_status"].item() == "finished"
        assert data["t"].size == 0


def test_simulation_service_delegates_crisp_and_ocean_forward_models():
    calls: list[str] = []
    service = SimulationService(
        crisp_simulator=lambda *args: calls.append("crisp") or "crisp-result",
        ocean_simulator=lambda *args: calls.append("ocean") or "ocean-result",
    )

    assert service.simulate_crisp("profile", "state") == "crisp-result"
    assert service.simulate_ocean("form-factor", "state") == "ocean-result"
    assert calls == ["crisp", "ocean"]


def test_app_logic_reconstruction_pipeline_dispatches_by_selected_algorithm():
    logic = AppLogic()
    calls: list[str] = []
    logic._reconstruction_pipeline = ReconstructionPipeline(
        {
            RECONSTRUCTION_ALGORITHM.GERCHBERG_SAXTON: lambda request: (
                calls.append("gs") or ("profile", "form_factor", "summary")
            ),
            RECONSTRUCTION_ALGORITHM.CRISP: lambda request: (
                calls.append("crisp") or ("profile", "form_factor", "summary")
            ),
        }
    )
    state = ControlsState(
        scenario=ProfileModelState(),
        measurement=MeasurementState(),
        reconstruction=ReconstructionState(
            algorithm=RECONSTRUCTION_ALGORITHM.CRISP,
        ),
    )

    result = logic.compute_reconstruction(
        grid=ProfileModel(state.scenario).grid,
        measurements=(),
        controls_state=state,
        ff_input=None,
    )

    assert result == ("profile", "form_factor", "summary")
    assert calls == ["crisp"]


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


def test_load_measurements_npz_waterflow_average_as_ocean_relative_signal(tmp_path):
    path = tmp_path / "waterflow.npz"
    wavelength_nm = np.array([950.0, 1000.0, 1500.0, 2000.0, 2400.0])
    average = np.array([100.0, 400.0, 900.0, 1600.0, 2500.0])
    background = np.full_like(average, 3000.0)
    np.savez(
        path,
        dumpversion=np.array(2),
        e_axis=wavelength_nm,
        average=average,
        background=background,
    )

    measurements = load_measurements_npz(path)

    assert len(measurements) == 1
    ocean = measurements[0]
    assert ocean.label == "Ocean NIR relative |F|"
    assert ocean.kind == "ocean_nir"
    assert ocean.calibration == "relative_shape"
    assert ocean.use_in_reconstruction is False
    assert np.all(np.diff(ocean.measured.freq) > 0.0)
    assert np.all(ocean.measured.freq >= IR_MIN_HZ)
    assert np.all(ocean.measured.freq <= IR_MAX_HZ)
    assert ocean.measured.mag[0] == pytest.approx(1.0)
    assert ocean.measured.mag[-1] == pytest.approx(np.sqrt(100.0) / 48.0)


def test_load_measurements_npz_waterflow_rejects_empty_signal(tmp_path):
    path = tmp_path / "waterflow.npz"
    np.savez(
        path,
        e_axis=np.array([1000.0, 1500.0]),
        average=np.zeros(2),
        background=np.ones(2),
    )

    with pytest.raises(ValueError, match="usable Ocean/NIR signal"):
        load_measurements_npz(path)


def test_load_measurements_npz_cor2d_spec_hist_mean_as_ocean_relative_signal(tmp_path):
    path = tmp_path / "cor2d.npz"
    wavelength_nm = np.array([950.0, 1000.0, 1500.0, 2000.0, 2400.0])
    spec_hist = np.array(
        [
            [100.0, 400.0, 900.0, 1600.0, 2500.0, 9999.0],
            [100.0, 400.0, 900.0, 1600.0, 2500.0, 9999.0],
        ]
    )
    np.savez(
        path,
        dumpversion=np.array(1),
        phen_scale=wavelength_nm,
        spec_hist=spec_hist,
    )

    measurements = load_measurements_npz(path)

    assert len(measurements) == 1
    ocean = measurements[0]
    assert ocean.label == "Ocean NIR relative |F|"
    assert ocean.kind == "ocean_nir"
    assert ocean.use_in_reconstruction is False
    assert np.all(ocean.measured.freq >= IR_MIN_HZ)
    assert np.all(ocean.measured.freq <= IR_MAX_HZ)
    assert ocean.measured.mag[0] == pytest.approx(1.0)
    assert ocean.measured.mag[-1] == pytest.approx(np.sqrt(100.0) / 48.0)


def test_load_measurements_npz_cor2d_clips_negative_background_corrected_mean(
    tmp_path,
):
    path = tmp_path / "cor2d_noise.npz"
    wavelength_nm = np.array([950.0, 1000.0, 1500.0, 2000.0, 2400.0])
    spec_hist = np.array(
        [
            [100.0, 400.0, -10.0, 1600.0, 2500.0],
            [100.0, 400.0, -4.0, 1600.0, 2500.0],
        ]
    )
    np.savez(
        path,
        dumpversion=np.array(1),
        phen_scale=wavelength_nm,
        spec_hist=spec_hist,
    )

    measurements = load_measurements_npz(path)

    assert len(measurements) == 1
    ocean = measurements[0]
    assert ocean.kind == "ocean_nir"
    assert np.all(np.isfinite(ocean.measured.mag))
    assert np.all(ocean.measured.mag >= 0.0)


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
                    [[0.3, 0.09], [0.4, 0.16]],
                ],
                dtype=np.float32,
            ),
        )

    measurements = load_measurements_h5(path)

    crisp_input = measurements[0].crisp_input
    assert crisp_input is not None
    assert "shot=1" in crisp_input.source
    assert crisp_input.charge_c == pytest.approx(0.25e-9)
    assert_allclose(crisp_input.freq, [0.3e12, 0.4e12])
    assert_allclose(crisp_input.mag, [0.09, 0.16])
    assert_allclose(crisp_input.mag_std, [0.0, 0.0])
    assert_allclose(crisp_input.detection_limit, [0.0, 0.0])


def test_load_measurements_h5_uses_reconstruction_input_columns(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "measurement.h5"
    with h5py.File(path, "w") as data:
        data.create_dataset(TIMESTAMP_KEY, data=np.array([100.0, 300.0]))
        data.create_dataset(
            CRISP_FORMFACTOR_XY_KEY,
            data=np.array(
                [
                    [[0.1, 0.01], [0.2, 0.04]],
                    [[0.3, 0.09], [0.4, 0.16]],
                ],
                dtype=np.float32,
            ),
        )
        data.create_dataset(
            CRISP_INPUT_FFSQ_KEY,
            data=np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32),
        )
        data.create_dataset(
            CRISP_INPUT_FFSQ_STD_KEY,
            data=np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32),
        )
        data.create_dataset(
            CRISP_INPUT_FFSQ_DETECTION_LIMIT_KEY,
            data=np.array([[0.001, 0.002], [0.003, 0.004]], dtype=np.float32),
        )

    measurements = load_measurements_h5(path)

    assert_allclose(measurements[0].measured.mag, [0.3, 0.4])
    crisp_input = measurements[0].crisp_input
    assert crisp_input is not None
    assert_allclose(crisp_input.mag, [0.33, 0.44])
    assert_allclose(crisp_input.mag_std, [0.03, 0.04])
    assert_allclose(crisp_input.detection_limit, [0.003, 0.004])


def test_load_measurements_h5_loads_sa1_reference_current_profile(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "measurement.h5"
    charge_c = 0.25e-9
    current = np.linspace(400.0, 600.0, 1024)
    with h5py.File(path, "w") as data:
        data.create_dataset(TIMESTAMP_KEY, data=np.array([100.0, 300.0]))
        data.create_dataset(CRISP_CHARGE_KEY, data=np.array([0.1, charge_c * 1e9]))
        data.create_dataset(
            CRISP_FORMFACTOR_XY_KEY,
            data=np.array(
                [
                    [[0.1, 0.01], [0.2, 0.04]],
                    [[0.3, 0.09], [0.4, 0.16]],
                ],
                dtype=np.float32,
            ),
        )
        data.create_dataset(
            CRISP_SA1_CURRENT_PROFILE_KEY,
            data=np.vstack(
                [
                    np.linspace(0.0, 1.0, 1024),
                    current,
                ]
            ),
        )

    measurements = load_measurements_h5(path)

    reference = measurements[0].reference_current
    assert reference is not None
    assert reference.label == "CRISP SA1"
    assert reference.time_s.shape == (1024,)
    assert reference.current_a.shape == (1024,)
    assert_allclose(reference.current_a[:3], current[:3])
    dt_s = reference.time_s[1] - reference.time_s[0]
    assert_allclose(float(np.sum(reference.current_a)) * dt_s, charge_c)
    assert measurements[0].crisp_input is not None
    assert measurements[0].crisp_input.max_frequency_thz == pytest.approx(
        reference.inferred_max_frequency_thz
    )


def test_load_measurements_h5_loads_ocean_nir_relative_measurement(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "measurement.h5"
    wavelengths_nm = np.array([950.0, 1000.0, 1500.0, 2000.0, 2400.0])
    intensities = np.array([100.0, 400.0, 900.0, 1600.0, 2500.0])
    with h5py.File(path, "w") as data:
        data.create_dataset(TIMESTAMP_KEY, data=np.array([100.0, 300.0]))
        data.create_dataset(
            CRISP_FORMFACTOR_XY_KEY,
            data=np.array(
                [
                    [[0.1, 0.01], [0.2, 0.04]],
                    [[0.3, 0.09], [0.4, 0.16]],
                ],
                dtype=np.float32,
            ),
        )
        data.create_dataset(
            OCEAN_SPECTRUM_KEY,
            data=np.array(
                [
                    np.column_stack((wavelengths_nm, np.zeros_like(intensities))),
                    np.column_stack((wavelengths_nm, intensities)),
                ],
                dtype=np.float32,
            ),
        )

    measurements = load_measurements_h5(path)

    assert [item.label for item in measurements] == [
        "CRISP latest 1970-01-01T00:05:00+00:00",
        "Ocean NIR relative |F|",
    ]
    ocean = measurements[1]
    assert ocean.kind == "ocean_nir"
    assert ocean.calibration == "relative_shape"
    assert ocean.use_in_reconstruction is False
    assert np.all(np.diff(ocean.measured.freq) > 0.0)
    assert np.all(ocean.measured.freq >= IR_MIN_HZ)
    assert np.all(ocean.measured.freq <= IR_MAX_HZ)
    assert np.all(np.isfinite(ocean.measured.mag))
    assert np.all(ocean.measured.mag >= 0.0)
    assert np.all(ocean.measured.mag <= 1.0)
    assert ocean.measured.mag[-1] == pytest.approx(np.sqrt(100.0) / 48.0)
    assert ocean.measured.mag[0] == pytest.approx(1.0)


def test_load_ir_measurements_file_loads_ocean_from_h5_shot(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "measurement.h5"
    wavelengths_nm = np.array([950.0, 1000.0, 1500.0, 2000.0, 2400.0])
    first_intensities = np.array([100.0, 400.0, 900.0, 1600.0, 2500.0])
    second_intensities = first_intensities * 2.0
    with h5py.File(path, "w") as data:
        data.create_dataset(TIMESTAMP_KEY, data=np.array([100.0, 300.0]))
        data.create_dataset(
            CRISP_FORMFACTOR_XY_KEY,
            data=np.array(
                [
                    [[0.1, 0.01], [0.2, 0.04]],
                    [[0.3, 0.09], [0.4, 0.16]],
                ],
                dtype=np.float32,
            ),
        )
        data.create_dataset(
            OCEAN_SPECTRUM_KEY,
            data=np.array(
                [
                    np.column_stack((wavelengths_nm, first_intensities)),
                    np.column_stack((wavelengths_nm, second_intensities)),
                ],
                dtype=np.float32,
            ),
        )

    loaded = load_ir_measurements_file(path, h5_shot_index=0)

    assert len(loaded) == 1
    assert loaded[0].kind == "ocean_nir"
    assert loaded[0].label == "Ocean NIR relative |F|"
    assert loaded[0].measured.mag[0] == pytest.approx(1.0)


def test_app_logic_load_ir_from_h5_replaces_ocean_for_selected_shot(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "measurement.h5"
    wavelengths_nm = np.array([1000.0, 1500.0, 2000.0])
    with h5py.File(path, "w") as data:
        data.create_dataset(TIMESTAMP_KEY, data=np.array([100.0, 300.0]))
        data.create_dataset(
            CRISP_FORMFACTOR_XY_KEY,
            data=np.array(
                [
                    [[0.1, 0.01], [0.2, 0.04]],
                    [[0.3, 0.09], [0.4, 0.16]],
                ],
                dtype=np.float32,
            ),
        )
        data.create_dataset(
            OCEAN_SPECTRUM_KEY,
            data=np.array(
                [
                    np.column_stack((wavelengths_nm, [1.0, 4.0, 9.0])),
                    np.column_stack((wavelengths_nm, [2.0, 8.0, 18.0])),
                ],
                dtype=np.float32,
            ),
        )

    logic = AppLogic()
    crisp = LoadedMeasurement(
        "CRISP", Measurement(np.array([1e12]), np.array([0.5])), kind="crisp"
    )
    old_ir = LoadedMeasurement(
        "old IR", Measurement(np.array([130e12]), np.array([0.5])), kind="ocean_nir"
    )
    logic.loaded_measurements = (crisp, old_ir)

    loaded = logic.load_ir_measurements(path, h5_shot_index=0)

    assert loaded[0] is crisp
    assert loaded[1].kind == "ocean_nir"
    assert loaded[1] is not old_ir


def test_load_measurements_h5_skips_zero_ocean_signal(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "measurement.h5"
    with h5py.File(path, "w") as data:
        data.create_dataset(TIMESTAMP_KEY, data=np.array([100.0]))
        data.create_dataset(
            CRISP_FORMFACTOR_XY_KEY,
            data=np.array([[[0.1, 0.01], [0.2, 0.04]]], dtype=np.float32),
        )
        data.create_dataset(
            OCEAN_SPECTRUM_KEY,
            data=np.array(
                [np.column_stack((np.array([1000.0, 1500.0]), np.zeros(2)))],
                dtype=np.float32,
            ),
        )

    measurements = load_measurements_h5(path)

    assert len(measurements) == 1
    assert measurements[0].kind == "crisp"


def test_load_measurements_h5_rejects_mismatched_ocean_spectrum_shape(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "measurement.h5"
    with h5py.File(path, "w") as data:
        data.create_dataset(TIMESTAMP_KEY, data=np.array([100.0]))
        data.create_dataset(
            CRISP_FORMFACTOR_XY_KEY,
            data=np.array([[[0.1, 0.01], [0.2, 0.04]]], dtype=np.float32),
        )
        data.create_dataset(
            OCEAN_SPECTRUM_KEY,
            data=np.zeros((2, 2, 2), dtype=np.float32),
        )

    with pytest.raises(ValueError, match="Ocean spectrum"):
        load_measurements_h5(path)


def test_load_measurements_h5_rejects_mismatched_reconstruction_input_shape(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "measurement.h5"
    with h5py.File(path, "w") as data:
        data.create_dataset(TIMESTAMP_KEY, data=np.array([100.0]))
        data.create_dataset(
            CRISP_FORMFACTOR_XY_KEY,
            data=np.array([[[0.1, 0.01], [0.2, 0.04]]], dtype=np.float32),
        )
        data.create_dataset(
            CRISP_INPUT_FFSQ_KEY,
            data=np.array([[0.11, 0.22, 0.33]], dtype=np.float32),
        )

    with pytest.raises(ValueError, match="must have shape"):
        load_measurements_h5(path)


def test_load_measurements_h5_rejects_invalid_crisp_reconstruction_values(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "bad_reconstruction_input.h5"
    with h5py.File(path, "w") as data:
        data.create_dataset(TIMESTAMP_KEY, data=np.array([100.0]))
        data.create_dataset(
            CRISP_FORMFACTOR_XY_KEY,
            data=np.array([[[0.1, 0.01], [0.2, 0.04]]], dtype=np.float32),
        )
        data.create_dataset(
            CRISP_INPUT_FFSQ_KEY,
            data=np.array([[-0.1, 0.04]], dtype=np.float32),
        )

    measurements = load_crisp_measurements_file(path)

    crisp_input = measurements[0].crisp_input
    assert crisp_input is not None
    assert crisp_input.mag[0] == pytest.approx(-0.1)


def test_load_measurements_h5_rejects_mismatched_reference_current_shape(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "measurement.h5"
    with h5py.File(path, "w") as data:
        data.create_dataset(TIMESTAMP_KEY, data=np.array([100.0]))
        data.create_dataset(
            CRISP_FORMFACTOR_XY_KEY,
            data=np.array([[[0.1, 0.01], [0.2, 0.04]]], dtype=np.float32),
        )
        data.create_dataset(
            CRISP_SA1_CURRENT_PROFILE_KEY,
            data=np.zeros((2, 1024), dtype=float),
        )

    with pytest.raises(ValueError, match="CURRENT_PROFILE"):
        load_measurements_h5(path)


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


def test_load_measurements_h5_crisp_loader_ignores_optional_ocean_dataset(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "measurement.h5"
    with h5py.File(path, "w") as data:
        data.create_dataset(TIMESTAMP_KEY, data=np.array([100.0]))
        data.create_dataset(
            CRISP_FORMFACTOR_XY_KEY,
            data=np.array([[[0.1, 0.01], [0.2, 0.04]]], dtype=np.float32),
        )
        data.create_dataset(OCEAN_SPECTRUM_KEY, data=np.zeros((2, 2, 2)))

    measurements = load_crisp_measurements_file(path)

    assert [item.kind for item in measurements] == ["crisp"]


def test_app_logic_load_crisp_rejects_generic_npz_and_preserves_state(tmp_path):
    path = tmp_path / "generic.npz"
    np.savez(path, freq_hz=np.array([1.0]), mag=np.array([0.5]))
    logic = AppLogic()
    previous = LoadedMeasurement(
        "old CRISP", Measurement([1.0], [0.5]), kind="crisp"
    )
    logic.loaded_measurements = (previous,)
    logic.phase_last = np.array([0.1, 0.2])
    logic.reconstruction_summary.status = "finished"
    summary_before = logic.reconstruction_summary

    with pytest.raises(ValueError, match="CRISP"):
        logic.load_crisp_measurements(path)

    assert logic.loaded_measurements == (previous,)
    assert np.array_equal(logic.phase_last, [0.1, 0.2])
    assert logic.reconstruction_summary is summary_before
    assert logic.reconstruction_summary.status == "finished"


def test_app_logic_load_ir_rejects_generic_npz_and_preserves_state(tmp_path):
    path = tmp_path / "generic.npz"
    np.savez(path, freq_hz=np.array([1.0]), mag=np.array([0.5]))
    logic = AppLogic()
    previous = LoadedMeasurement(
        "old IR", Measurement([130e12], [0.5]), kind="ocean_nir"
    )
    logic.loaded_measurements = (previous,)
    logic.phase_last = np.array([0.3, 0.4])
    logic.reconstruction_summary.status = "finished"
    summary_before = logic.reconstruction_summary

    with pytest.raises(ValueError, match="IR"):
        logic.load_ir_measurements(path)

    assert logic.loaded_measurements == (previous,)
    assert np.array_equal(logic.phase_last, [0.3, 0.4])
    assert logic.reconstruction_summary is summary_before
    assert logic.reconstruction_summary.status == "finished"


def test_load_ir_measurements_file_rejects_negative_ocean_signal(tmp_path):
    path = tmp_path / "bad_ir.npz"
    np.savez(path, e_axis=np.array([1000.0, 1500.0]), average=np.array([1.0, -1.0]))

    with pytest.raises(ValueError, match="non-negative"):
        load_ir_measurements_file(path)


def test_app_logic_load_ir_replaces_only_existing_ocean_measurement(tmp_path):
    path = tmp_path / "ir.npz"
    np.savez(
        path,
        e_axis=np.array([1000.0, 1500.0, 2000.0]),
        average=np.array([100.0, 400.0, 900.0]),
    )
    logic = AppLogic()
    crisp = LoadedMeasurement(
        "CRISP", Measurement(np.array([1e12]), np.array([0.5])), kind="crisp"
    )
    old_ir = LoadedMeasurement(
        "old IR", Measurement(np.array([130e12]), np.array([0.5])), kind="ocean_nir"
    )
    logic.loaded_measurements = (crisp, old_ir)

    loaded = logic.load_ir_measurements(path)

    assert loaded[0] is crisp
    assert len(loaded) == 2
    assert loaded[1].kind == "ocean_nir"
    assert loaded[1] is not old_ir


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


def test_app_logic_excludes_loaded_ocean_nir_from_active_reconstruction():
    logic = AppLogic()
    logic.loaded_measurements = (
        LoadedMeasurement(
            label="Loaded CRISP",
            measured=Measurement(
                freq=np.array([10e12, 20e12]),
                mag=np.array([0.9, 0.7]),
            ),
            kind="crisp",
            calibration="absolute_or_calibrated",
            use_in_reconstruction=True,
        ),
        LoadedMeasurement(
            label="Ocean NIR relative |F|",
            measured=Measurement(
                freq=np.array([130e12, 150e12]),
                mag=np.array([1.0, 0.8]),
            ),
            kind="ocean_nir",
            calibration="relative_shape",
            use_in_reconstruction=False,
        ),
    )
    state = ControlsState(
        scenario=ProfileModelState(),
        measurement=MeasurementState(crisp=True, infrared=True),
        reconstruction=ReconstructionState(phase_init_mode=PHASE_INIT_MODE.ZERO),
    )
    prof = ProfileModel(state.scenario).compute_profile()
    ff = prof.to_form_factor()

    active, source = logic.active_measurements(ff, state.measurement)
    visible = logic.visible_measurements(ff, state.measurement)

    assert source == "loaded"
    assert len(active) == 1
    assert_allclose(active[0].freq, [10e12, 20e12])
    assert [item.label for item in visible] == [
        "Loaded CRISP",
        "Ocean NIR relative |F|",
    ]


def test_app_logic_uses_ocean_nir_only_as_relative_constraint():
    logic = AppLogic()
    logic.loaded_measurements = (
        LoadedMeasurement(
            label="Loaded CRISP",
            measured=Measurement(
                freq=np.array([10e12, 20e12]),
                mag=np.array([0.9, 0.7]),
            ),
            kind="crisp",
            calibration="absolute_or_calibrated",
            use_in_reconstruction=True,
        ),
        LoadedMeasurement(
            label="Ocean NIR relative |F|",
            measured=Measurement(
                freq=np.array([130e12, 150e12]),
                mag=np.array([1.0, 0.8]),
            ),
            kind="ocean_nir",
            calibration="relative_shape",
            use_in_reconstruction=False,
        ),
    )
    reconstruction_state = ReconstructionState(
        phase_init_mode=PHASE_INIT_MODE.ZERO,
        use_ir_relative_constraint=True,
    )

    relative = logic.relative_measurements_for_reconstruction(reconstruction_state)

    assert len(relative) == 1
    assert_allclose(relative[0].freq, [130e12, 150e12])


def test_app_logic_missing_ocean_nir_relative_constraint_fails_clearly():
    logic = AppLogic()
    logic.loaded_measurements = (
        LoadedMeasurement(
            label="Loaded CRISP",
            measured=Measurement(
                freq=np.array([10e12, 20e12]),
                mag=np.array([0.9, 0.7]),
            ),
            kind="crisp",
            calibration="absolute_or_calibrated",
            use_in_reconstruction=True,
        ),
    )
    reconstruction_state = ReconstructionState(
        phase_init_mode=PHASE_INIT_MODE.ZERO,
        use_ir_relative_constraint=True,
    )

    with pytest.raises(
        ValueError,
        match="No relative IR measurement is available",
    ):
        logic.relative_measurements_for_reconstruction(reconstruction_state)


def test_app_logic_uses_ocean_detector_simulation_with_loaded_crisp_measurement():
    logic = AppLogic()
    logic.loaded_measurements = (
        LoadedMeasurement(
            label="Loaded CRISP",
            measured=Measurement(
                freq=np.array([10e12, 20e12]),
                mag=np.array([0.9, 0.7]),
            ),
            kind="crisp",
            calibration="absolute_or_calibrated",
        ),
    )
    profile = ProfileModel(ProfileModelState()).compute_profile()
    form_factor = profile.to_form_factor()

    relative = logic.relative_measurements_for_reconstruction(
        ReconstructionState(use_ir_relative_constraint=True),
        measurement_state=MeasurementState(
            infrared=True,
            infrared_simulation_mode=IR_SIMULATION_MODE.OCEAN,
        ),
        form_factor=form_factor,
    )

    assert len(relative) == 1
    assert np.all(relative[0].freq >= IR_MIN_HZ)
    assert np.all(relative[0].freq <= IR_MAX_HZ)


def test_app_logic_crisp_algorithm_can_extend_with_ocean_nir():
    freq_hz = np.linspace(0.5e12, 60.0e12, 160)
    crisp_input = LoadedMeasurement(
        label="Loaded CRISP",
        measured=Measurement(
            freq=freq_hz,
            mag=np.sqrt(np.exp(-0.5 * ((freq_hz / 1e12) / 18.0) ** 2)),
        ),
        crisp_input=SquaredMagnitudeMeasurement(
            freq=freq_hz,
            mag=np.exp(-0.5 * ((freq_hz / 1e12) / 18.0) ** 2),
            mag_std=np.zeros_like(freq_hz),
            detection_limit=np.zeros_like(freq_hz),
            charge_c=250e-12,
        ),
        kind="crisp",
        calibration="absolute_or_calibrated",
        use_in_reconstruction=True,
    )
    ocean = LoadedMeasurement(
        label="Ocean NIR relative |F|",
        measured=Measurement(
            freq=np.linspace(120e12, 333e12, 40),
            mag=np.linspace(1.0, 0.05, 40),
        ),
        kind="ocean_nir",
        calibration="relative_shape",
        use_in_reconstruction=False,
    )
    logic = AppLogic()
    logic.loaded_measurements = (crisp_input, ocean)
    state = ControlsState(
        scenario=ProfileModelState(),
        measurement=MeasurementState(),
        reconstruction=ReconstructionState(
            algorithm=RECONSTRUCTION_ALGORITHM.CRISP,
            phase_init_mode=PHASE_INIT_MODE.ZERO,
        ),
    )
    prof = ProfileModel(state.scenario).compute_profile()
    ff = prof.to_form_factor()
    measurements, source = logic.active_measurements(ff, state.measurement)

    profile, form_factor, summary = logic.compute_reconstruction(
        grid=ff.grid,
        measurements=measurements,
        controls_state=state,
        ff_input=ff,
        measurement_source=source,
    )

    assert summary.algorithm == "CRISP + IR"
    assert summary.ir_relative_constraint_used is True
    assert summary.relative_measurement_count == 1
    assert summary.crisp_diagnostics is not None
    assert summary.measurement_error is not None
    assert profile.charge == pytest.approx(250e-12)
    assert form_factor.mag.shape == profile.to_form_factor().mag.shape


def test_app_logic_relative_anchor_uses_crisp_reconstructed_formfactor():
    freq_hz = np.linspace(0.5e12, 60.0e12, 160)
    ffsq = np.exp(-0.5 * ((freq_hz / 1e12) / 18.0) ** 2)
    crisp_input = LoadedMeasurement(
        label="Loaded CRISP",
        measured=Measurement(freq=freq_hz, mag=np.sqrt(ffsq)),
        crisp_input=SquaredMagnitudeMeasurement(
            freq=freq_hz,
            mag=ffsq,
            mag_std=np.zeros_like(freq_hz),
            detection_limit=np.zeros_like(freq_hz),
            charge_c=250e-12,
        ),
        kind="crisp",
        calibration="absolute_or_calibrated",
        use_in_reconstruction=True,
    )
    logic = AppLogic()
    logic.loaded_measurements = (crisp_input,)
    state = ReconstructionState(use_ir_relative_constraint=True)

    anchor = logic.relative_anchor_formfactor(state, measurement_source="loaded")
    expected = CrispReconstruction(crisp_input.crisp_input).run().form_factor

    assert anchor is not None
    assert_allclose(anchor.mag, expected.mag)
    assert_allclose(anchor.phase, expected.phase)


def test_app_logic_replace_loaded_ocean_measurements_preserves_crisp(tmp_path):
    logic = AppLogic()
    crisp = LoadedMeasurement(
        label="Loaded CRISP",
        measured=Measurement(
            freq=np.array([10e12, 20e12]),
            mag=np.array([0.9, 0.7]),
        ),
        kind="crisp",
        calibration="absolute_or_calibrated",
        use_in_reconstruction=True,
    )
    stale_ocean = LoadedMeasurement(
        label="Stale Ocean",
        measured=Measurement(
            freq=np.array([130e12, 150e12]),
            mag=np.array([0.2, 0.1]),
        ),
        kind="ocean_nir",
        calibration="relative_shape",
        use_in_reconstruction=False,
    )
    logic.loaded_measurements = (crisp, stale_ocean)
    logic.phase_last = np.array([1.0])
    path = tmp_path / "waterflow.npz"
    np.savez(
        path,
        e_axis=np.array([950.0, 1000.0, 1500.0, 2000.0, 2400.0]),
        average=np.array([100.0, 400.0, 900.0, 1600.0, 2500.0]),
    )

    loaded = logic.replace_loaded_ocean_measurements(path)

    assert loaded[0] is crisp
    assert [item.kind for item in loaded] == ["crisp", "ocean_nir"]
    assert loaded[1].label == "Ocean NIR relative |F|"
    assert loaded[1] is not stale_ocean
    assert logic.phase_last is None
    assert logic.reconstruction_summary.status == "not_run"

    state = ControlsState(
        scenario=ProfileModelState(),
        measurement=MeasurementState(crisp=True, infrared=True),
        reconstruction=ReconstructionState(phase_init_mode=PHASE_INIT_MODE.ZERO),
    )
    ff = ProfileModel(state.scenario).compute_profile().to_form_factor()
    active, source = logic.active_measurements(ff, state.measurement)
    visible = logic.visible_measurements(ff, state.measurement)

    assert source == "loaded"
    assert len(active) == 1
    assert_allclose(active[0].freq, [10e12, 20e12])
    assert [item.kind for item in visible] == ["crisp", "ocean_nir"]


def test_app_logic_replace_loaded_ocean_measurements_does_not_accumulate(tmp_path):
    logic = AppLogic()
    first = tmp_path / "first_waterflow.npz"
    second = tmp_path / "second_waterflow.npz"
    wavelengths = np.array([950.0, 1000.0, 1500.0, 2000.0, 2400.0])
    np.savez(first, e_axis=wavelengths, average=np.array([100, 200, 300, 400, 500]))
    np.savez(second, e_axis=wavelengths, average=np.array([500, 400, 300, 200, 100]))

    logic.replace_loaded_ocean_measurements(first)
    first_measurement = logic.loaded_measurements[0]
    logic.replace_loaded_ocean_measurements(second)

    assert len(logic.loaded_measurements) == 1
    assert logic.loaded_measurements[0].kind == "ocean_nir"
    assert logic.loaded_measurements[0] is not first_measurement


def test_app_logic_replace_loaded_ocean_measurements_error_preserves_state(tmp_path):
    logic = AppLogic()
    crisp = LoadedMeasurement(
        label="Loaded CRISP",
        measured=Measurement(freq=np.array([10e12]), mag=np.array([0.9])),
        kind="crisp",
        calibration="absolute_or_calibrated",
    )
    logic.loaded_measurements = (crisp,)
    path = tmp_path / "not_ocean.npz"
    np.savez(path, freq_hz=np.array([1.0]), mag=np.array([0.5]))

    with pytest.raises(ValueError, match="Ocean/NIR"):
        logic.replace_loaded_ocean_measurements(path)

    assert logic.loaded_measurements == (crisp,)


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


def test_crisp_algorithm_automatically_extends_with_enabled_simulated_ir_band():
    logic = AppLogic()
    state = ControlsState(
        scenario=ProfileModelState(),
        measurement=MeasurementState(crisp=True, infrared=True),
        reconstruction=ReconstructionState(
            algorithm=RECONSTRUCTION_ALGORITHM.CRISP,
        ),
    )

    profile, form_factor, measurements = logic.compute_initial(state)
    _, _, summary = logic.compute_reconstruction(
        grid=profile.grid,
        measurements=measurements,
        controls_state=state,
        ff_input=form_factor,
    )

    assert summary.algorithm == "CRISP + IR"
    assert summary.ir_relative_constraint_used is False
    assert summary.relative_measurement_count == 0


def test_app_logic_uses_detector_simulation_for_visible_and_crisp_input():
    logic = AppLogic()
    state = ControlsState(
        scenario=ProfileModelState(),
        measurement=MeasurementState(
            crisp=True,
            crisp_simulation_mode=CRISP_SIMULATION_MODE.DETECTOR,
            crisp_n_shots=4,
            crisp_noise_seed=17,
        ),
        reconstruction=ReconstructionState(
            algorithm=RECONSTRUCTION_ALGORITHM.CRISP,
        ),
    )
    prof_input, ff_input, measurements = logic.compute_initial(state)
    visible = logic.visible_measurements(
        ff_input,
        state.measurement,
        input_profile=prof_input,
    )
    active, source = logic.active_measurements(
        ff_input,
        state.measurement,
        input_profile=prof_input,
    )
    crisp_input = logic._active_crisp_input(
        active,
        state,
        measurement_source=source,
        input_profile=prof_input,
    )

    assert len(measurements) == len(active) == 1
    assert visible[0].label == "CRISP detector simulation"
    assert visible[0].measured.freq.shape == (240,)
    assert_allclose(active[0].mag, visible[0].measured.mag)
    assert_allclose(crisp_input.mag, active[0].mag**2)
    assert np.all(crisp_input.mag_std > 0.0)
    assert np.all(crisp_input.detection_limit > 0.0)


def test_detector_simulation_crisp_reconstruction_runs_with_simulated_input():
    logic = AppLogic()
    state = ControlsState(
        scenario=ProfileModelState(),
        measurement=MeasurementState(
            crisp=True,
            crisp_simulation_mode=CRISP_SIMULATION_MODE.DETECTOR,
            crisp_noise_seed=3,
        ),
        reconstruction=ReconstructionState(
            algorithm=RECONSTRUCTION_ALGORITHM.CRISP,
        ),
    )
    prof_input, ff_input, measurements = logic.compute_initial(state)

    profile, form_factor, summary = logic.compute_reconstruction(
        grid=prof_input.grid,
        measurements=measurements,
        controls_state=state,
        ff_input=ff_input,
        input_profile=prof_input,
    )

    assert summary.algorithm == "CRISP"
    assert summary.status == "finished"
    assert summary.crisp_diagnostics is not None
    assert profile.grid.N == 1024
    assert form_factor.mag.shape == (513,)


def test_detector_simulation_is_accepted_by_gerchberg_saxton():
    logic = AppLogic()
    state = ControlsState(
        scenario=ProfileModelState(),
        measurement=MeasurementState(
            crisp=True,
            crisp_simulation_mode=CRISP_SIMULATION_MODE.DETECTOR,
            crisp_noise_seed=9,
        ),
        reconstruction=ReconstructionState(),
    )
    prof_input, ff_input, measurements = logic.compute_initial(state)

    profile, form_factor, summary = logic.compute_reconstruction(
        grid=prof_input.grid,
        measurements=measurements,
        controls_state=state,
        ff_input=ff_input,
        input_profile=prof_input,
    )

    assert summary.algorithm == "Gerchberg-Saxton"
    assert summary.status == "finished"
    assert profile.grid == prof_input.grid
    assert form_factor.mag.shape == ff_input.mag.shape


def test_ocean_detector_simulation_is_relative_and_carries_uncertainty():
    logic = AppLogic()
    state = ControlsState(
        scenario=ProfileModelState(),
        measurement=MeasurementState(
            crisp=True,
            infrared=True,
            infrared_simulation_mode=IR_SIMULATION_MODE.OCEAN,
            infrared_n_shots=9,
            infrared_noise_seed=12,
        ),
        reconstruction=ReconstructionState(use_ir_relative_constraint=True),
    )
    profile = ProfileModel(state.scenario).compute_profile()
    form_factor = profile.to_form_factor()

    visible = logic.visible_measurements(
        form_factor,
        state.measurement,
        input_profile=profile,
    )
    active, source = logic.active_measurements(
        form_factor,
        state.measurement,
        input_profile=profile,
    )
    relative = logic.relative_measurements_for_reconstruction(
        state.reconstruction,
        measurement_state=state.measurement,
        form_factor=form_factor,
    )

    assert source == "simulated"
    assert [item.label for item in visible] == [
        "CRISP",
        "Ocean NIR detector simulation",
    ]
    assert visible[1].kind == "ocean_nir"
    assert visible[1].calibration == "simulated_relative_shape"
    assert visible[1].use_in_reconstruction is False
    assert len(active) == 1
    assert len(relative) == 1
    assert relative[0].freq.size < 512
    assert np.all(relative[0].freq >= IR_MIN_HZ)
    assert np.all(relative[0].freq <= IR_MAX_HZ)
    assert relative[0].mag_std is not None
    assert relative[0].detection_limit is not None
    assert np.all(relative[0].mag_std > 0.0)


def test_ocean_detector_relative_constraint_runs_with_crisp_reconstruction():
    logic = AppLogic()
    state = ControlsState(
        scenario=ProfileModelState(),
        measurement=MeasurementState(
            crisp=True,
            infrared=True,
            infrared_simulation_mode=IR_SIMULATION_MODE.OCEAN,
            infrared_n_shots=4,
            infrared_noise_seed=5,
        ),
        reconstruction=ReconstructionState(
            algorithm=RECONSTRUCTION_ALGORITHM.CRISP,
        ),
    )
    profile, form_factor, _ = logic.compute_initial(state)
    active, source = logic.active_measurements(
        form_factor,
        state.measurement,
        input_profile=profile,
    )

    _, _, summary = logic.compute_reconstruction(
        grid=profile.grid,
        measurements=active,
        controls_state=state,
        ff_input=form_factor,
        measurement_source=source,
        input_profile=profile,
    )

    assert summary.algorithm == "CRISP + IR"
    assert summary.ir_relative_constraint_used is True
    assert summary.relative_measurement_count == 1


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
        assert data["crisp_intermediate_ffsq_std"].shape == data[
            "crisp_intermediate_ffsq"
        ].shape
        assert data["crisp_interpolated_ffabs"].shape == (513,)
        assert data["crisp_interpolated_ffabs_error"].shape == (513,)
        assert data["measurement_crisp_simulation_mode"] == "Ideal samples"
        assert data["measurement_crisp_n_shots"] == 1
        assert data["measurement_crisp_noise_seed"] == 0
        assert data["measurement_infrared_simulation_mode"] == "Ideal samples"
        assert data["measurement_infrared_n_shots"] == 1
        assert data["measurement_infrared_noise_seed"] == 0


def test_app_logic_export_writes_measurements_and_summary(tmp_path):
    logic = AppLogic()
    measurement_path = tmp_path / "measurement.npz"
    np.savez(
        measurement_path,
        freq_hz=np.array([10.0, 20.0]),
        mag=np.array([0.9, 0.7]),
    )
    logic.load_measurements(measurement_path)
    logic.loaded_measurements = (
        LoadedMeasurement(
            label=logic.loaded_measurements[0].label,
            measured=logic.loaded_measurements[0].measured,
            reference_current=ReferenceCurrentProfile(
                label="CRISP SA1",
                time_s=np.array([-1e-15, 0.0, 1e-15]),
                current_a=np.array([0.0, 10.0, 0.0]),
                inferred_max_frequency_thz=250.0,
            ),
            kind="crisp",
            calibration="absolute_or_calibrated",
            use_in_reconstruction=True,
        ),
        LoadedMeasurement(
            label="Ocean NIR relative |F|",
            measured=Measurement(
                freq=np.array([130e12, 150e12]),
                mag=np.array([1.0, 0.8]),
            ),
            kind="ocean_nir",
            calibration="relative_shape",
            use_in_reconstruction=False,
        ),
    )

    state = ControlsState(
        scenario=ProfileModelState(),
        measurement=MeasurementState(),
        reconstruction=ReconstructionState(
            phase_init_mode=PHASE_INIT_MODE.ZERO,
            use_fixed_ir_scale=True,
            fixed_ir_scale=2.5,
        ),
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
        assert not data["reconstruction_ir_relative_constraint_used"].item()
        assert data["reconstruction_relative_measurement_count"].item() == 0
        assert data["reconstruction_iterations"].item() == 12
        assert data["measurement_label_0"].item() == "measurement 1"
        assert_allclose(data["measurement_freq_hz_0"], [10.0, 20.0])
        assert data["measurement_kind_0"].item() == "crisp"
        assert data["measurement_calibration_0"].item() == "absolute_or_calibrated"
        assert data["measurement_use_in_reconstruction_0"].item()
        assert data["measurement_reference_current_label_0"].item() == "CRISP SA1"
        assert_allclose(data["measurement_reference_current_time_s_0"], [-1e-15, 0.0, 1e-15])
        assert_allclose(data["measurement_reference_current_a_0"], [0.0, 10.0, 0.0])
        assert data["measurement_reference_current_max_frequency_thz_0"] == 250.0
        assert data["measurement_label_1"].item() == "Ocean NIR relative |F|"
        assert data["measurement_kind_1"].item() == "ocean_nir"
        assert data["measurement_calibration_1"].item() == "relative_shape"
        assert not data["measurement_use_in_reconstruction_1"].item()
        assert_allclose(data["measurement_freq_hz_1"], [130e12, 150e12])
        assert_allclose(data["measurement_mag_1"], [1.0, 0.8])
        assert_allclose(data["reconstruction_history_iteration"], [1, 2])
        assert_allclose(
            data["reconstruction_history_measurement_error"], [0.3, 0.123]
        )
        assert_allclose(data["reconstruction_history_phase_delta_mse"], [1e-2, 1e-4])
        assert_allclose(
            data["reconstruction_history_profile_delta_mse"], [2e-2, 2e-4]
        )
        assert data["phase_init_mode"].item() == "zero"
        assert not data["reconstruction_use_ir_relative_constraint"].item()
        assert data["reconstruction_use_fixed_ir_scale"].item()
        assert data["reconstruction_fixed_ir_scale"] == 2.5
        assert "Normalize area" in set(data["reconstruction_time_constraints"])
        assert "Blend measured" in set(data["reconstruction_frequency_constraints"])
        assert "Measurement error" in set(data["reconstruction_stop_conditions"])
