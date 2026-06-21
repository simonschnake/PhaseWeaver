import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import phase_weaver.app.config as cfg
from phase_weaver.app.config import MEASUREMENT_MODE
from phase_weaver.app.logic import H5MeasurementShot
from phase_weaver.app.state import ProfileModelState, ReconstructionState
from phase_weaver.app.ui.main_window import MainWindow
from phase_weaver.app.ui.measurement_time_dialog import MeasurementTimeDialog
from phase_weaver.app.ui.reconstruction_panel import ReconstructionPanel
from phase_weaver.app.ui.toy_model_panel import ToyModelPanel
from phase_weaver.model.profiles import AsymSuperGaussParams


def _app():
    return QApplication.instance() or QApplication([])


def _settings(path):
    settings = QSettings(str(path), QSettings.Format.IniFormat)
    settings.clear()
    return settings


def _assert_params_close(actual, expected):
    assert actual.center == pytest.approx(expected.center)
    assert actual.width == pytest.approx(expected.width)
    assert actual.skew == pytest.approx(expected.skew)
    assert actual.order == pytest.approx(expected.order)
    assert actual.amplitude == pytest.approx(expected.amplitude)


def test_reconstruction_panel_auto_reconstruction_toggle_updates_state_and_label():
    _app()
    panel = ReconstructionPanel()

    assert panel.is_auto_reconstruction_enabled() is True
    assert panel.auto_reconstruct_button.text() == "Auto Reconstruction: On"

    panel.auto_reconstruct_button.click()

    assert panel.is_auto_reconstruction_enabled() is False
    assert panel.auto_reconstruct_button.text() == "Auto Reconstruction: Off"

    panel.auto_reconstruct_button.click()

    assert panel.is_auto_reconstruction_enabled() is True
    assert panel.auto_reconstruct_button.text() == "Auto Reconstruction: On"


def test_toy_model_panel_provides_profile_state():
    _app()
    panel = ToyModelPanel()

    state = panel.get_scenario_state()

    assert state.background == panel.background_box.get_params()
    assert state.peak == panel.spike_box.get_params()
    assert state.peak2_enabled is panel.spike2_box.is_enabled_component()
    assert state.peak2 == panel.spike2_box.get_params()


def test_toy_model_panel_restores_profile_state():
    _app()
    panel = ToyModelPanel()
    state = ProfileModelState(
        background=AsymSuperGaussParams(
            center=0.0,
            width=30e-15,
            skew=0.2,
            order=1.5,
            amplitude=9000.0,
        ),
        peak=AsymSuperGaussParams(
            center=-5e-15,
            width=1.2e-15,
            skew=-0.3,
            order=0.8,
            amplitude=15000.0,
        ),
        peak2_enabled=True,
        peak2=AsymSuperGaussParams(
            center=6e-15,
            width=1.4e-15,
            skew=0.4,
            order=0.9,
            amplitude=12000.0,
        ),
    )

    panel.set_scenario_state(state)

    restored = panel.get_scenario_state()
    _assert_params_close(restored.background, state.background)
    _assert_params_close(restored.peak, state.peak)
    assert restored.peak2_enabled == state.peak2_enabled
    _assert_params_close(restored.peak2, state.peak2)


def test_reconstruction_panel_provides_measurement_and_reconstruction_state():
    _app()
    panel = ReconstructionPanel()

    measurement_state = panel.get_measurement_state()
    reconstruction_state = panel.get_reconstruction_state()

    assert measurement_state.crisp is False
    assert measurement_state.infrared is False
    assert reconstruction_state.use_ir_relative_constraint is False
    assert reconstruction_state.use_fixed_ir_scale is False
    assert reconstruction_state.fixed_ir_scale == 1.0
    assert reconstruction_state.phase_init_mode == panel.phase_init_box.mode
    assert reconstruction_state.time_constraints == set(
        panel.time_constraint_box.selected_modes
    )
    assert reconstruction_state.frequency_constraints == set(
        panel.frequency_constraint_box.selected_modes
    )
    assert reconstruction_state.stop_conditions == set(
        panel.stop_condition_box.selected_modes
    )


def test_reconstruction_panel_restores_reconstruction_state():
    _app()
    panel = ReconstructionPanel()
    reconstruction_state = ReconstructionState(
        phase_init_mode=cfg.PHASE_INIT_MODE.LAST,
        use_ir_relative_constraint=True,
        use_fixed_ir_scale=True,
        fixed_ir_scale=2.5,
        time_constraints={cfg.RECON_TIME_CONSTRAINT.CENTER},
        frequency_constraints={cfg.RECON_FREQUENCY_CONSTRAINT.ENFORCE_DC},
        stop_conditions={cfg.RECON_STOP_CONDITION.MAX_ITER},
    )

    panel.set_measurement_state(
        panel.get_measurement_state().__class__(crisp=True, infrared=False)
    )
    panel.set_reconstruction_state(reconstruction_state)
    panel.set_auto_reconstruction_enabled(False)

    assert panel.get_measurement_state().crisp is True
    assert panel.get_measurement_state().infrared is False
    assert panel.get_reconstruction_state() == reconstruction_state
    assert panel.ir_relative_constraint_box.text() == "Use IR as relative constraint"
    assert panel.fixed_ir_scale_box.text() == "Use fixed IR scale"
    assert panel.is_auto_reconstruction_enabled() is False


def test_reconstruction_panel_ir_scale_slider_and_spinbox_stay_synced():
    _app()
    panel = ReconstructionPanel()

    panel.ir_scale_slider.setValue(1000)

    assert panel.ir_scale_spin.value() == pytest.approx(10.0)
    assert panel.get_reconstruction_state().fixed_ir_scale == pytest.approx(10.0)

    panel.ir_scale_spin.setValue(0.1)

    assert panel.ir_scale_slider.value() == -1000
    assert panel.get_reconstruction_state().fixed_ir_scale == pytest.approx(0.1)


def test_main_window_exposes_file_actions_in_file_menu():
    _app()
    window = MainWindow()

    assert window.file_menu.title() == "File"
    assert [action.text() for action in window.file_menu.actions()] == [
        "Load Measurements...",
        "Load IR Measurement...",
        "Export Data...",
    ]


def test_main_window_remembers_last_measurement_file(tmp_path):
    _app()
    measurement_path = tmp_path / "measurement.h5"
    measurement_path.write_bytes(b"placeholder")
    window = MainWindow(settings=_settings(tmp_path / "settings.ini"))

    window._remember_path(str(measurement_path), "paths/last_measurement_path")

    assert window._dialog_start_path("paths/last_measurement_path") == str(
        measurement_path
    )


def test_main_window_falls_back_to_last_measurement_directory(tmp_path):
    _app()
    missing_path = tmp_path / "deleted.h5"
    window = MainWindow(settings=_settings(tmp_path / "settings.ini"))

    window._remember_path(str(missing_path), "paths/last_measurement_path")

    assert window._dialog_start_path("paths/last_measurement_path") == str(tmp_path)


def test_main_window_uses_last_directory_for_export_default(tmp_path):
    _app()
    measurement_path = tmp_path / "measurement.h5"
    window = MainWindow(settings=_settings(tmp_path / "settings.ini"))

    window._remember_path(str(measurement_path), "paths/last_measurement_path")

    assert window._export_start_path() == str(tmp_path / "phase_weaver_export.npz")


def test_main_window_persists_controls_settings(tmp_path):
    _app()
    settings_path = tmp_path / "settings.ini"
    settings = _settings(settings_path)
    window = MainWindow(settings=settings)
    window.toy_model_panel.spike_box.set_params(
        AsymSuperGaussParams(
            center=-4e-15,
            width=2e-15,
            skew=0.5,
            order=1.2,
            amplitude=22000.0,
        )
    )
    window.toy_model_panel.spike2_box.set_enabled_component(True)
    window.reconstruction_panel.set_measurement_state(
        window.reconstruction_panel.get_measurement_state().__class__(
            crisp=True, infrared=True
        )
    )
    window.reconstruction_panel.set_reconstruction_state(
        ReconstructionState(
            phase_init_mode=cfg.PHASE_INIT_MODE.LAST,
            time_constraints={cfg.RECON_TIME_CONSTRAINT.CENTER},
            frequency_constraints={cfg.RECON_FREQUENCY_CONSTRAINT.ENFORCE_DC},
            stop_conditions={cfg.RECON_STOP_CONDITION.MAX_ITER},
        )
    )
    window.reconstruction_panel.set_auto_reconstruction_enabled(False)

    window._save_controls_settings()
    settings.setValue("controls/auto_reconstruction", False)
    settings.sync()

    restored = MainWindow(settings=QSettings(str(settings_path), QSettings.Format.IniFormat))

    assert restored.toy_model_panel.spike_box.get_params() == (
        window.toy_model_panel.spike_box.get_params()
    )
    assert restored.toy_model_panel.spike2_box.is_enabled_component() is True
    assert restored.reconstruction_panel.get_measurement_state() == (
        window.reconstruction_panel.get_measurement_state()
    )
    assert restored.reconstruction_panel.get_reconstruction_state() == (
        window.reconstruction_panel.get_reconstruction_state()
    )
    assert restored.reconstruction_panel.is_auto_reconstruction_enabled() is False


def test_measurement_time_dialog_selects_latest_shot_by_default():
    _app()
    dialog = MeasurementTimeDialog(
        (
            H5MeasurementShot(
                index=0,
                timestamp=100.0,
                measured_at="1970-01-01T00:01:40+00:00",
            ),
            H5MeasurementShot(
                index=1,
                timestamp=300.0,
                measured_at="1970-01-01T00:05:00+00:00",
            ),
            H5MeasurementShot(
                index=2,
                timestamp=200.0,
                measured_at="1970-01-01T00:03:20+00:00",
            ),
        )
    )

    assert dialog.selected_shot_index() == 1

    dialog.time_combo.setCurrentIndex(2)

    assert dialog.selected_shot_index() == 2


def test_main_window_hosts_setup_panels_in_external_windows():
    _app()
    window = MainWindow()

    assert window.toy_model_window.isVisible()
    assert window.toy_model_window.isWindow()
    assert window.toy_model_action.isChecked()
    assert window.toy_model_action.text() == "Toy Model"
    assert window.reconstruction_window.isVisible()
    assert window.reconstruction_window.isWindow()
    assert window.reconstruction_action.isChecked()
    assert window.reconstruction_action.text() == "Reconstruction Setup"

    window.toy_model_action.setChecked(False)
    window.reconstruction_action.setChecked(False)

    assert window.toy_model_window.isHidden()
    assert window.reconstruction_window.isHidden()
    assert window.get_controls_state().scenario == (
        window.toy_model_panel.get_scenario_state()
    )
    assert window.get_controls_state().measurement == (
        window.reconstruction_panel.get_measurement_state()
    )
    assert window.get_controls_state().reconstruction == (
        window.reconstruction_panel.get_reconstruction_state()
    )


def test_main_window_renders_active_measurement_points():
    _app()
    window = MainWindow()
    window.reconstruction_panel.measurement_box.selected_modes = {
        MEASUREMENT_MODE.CRISP,
        MEASUREMENT_MODE.INFRARED,
    }

    window.redraw_input()

    assert window.plot_panel.measurement_labels == ["CRISP", "IR"]
    assert len(window.plot_panel.measurement_lines) == 2
