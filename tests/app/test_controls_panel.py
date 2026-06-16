from PySide6.QtWidgets import QApplication

from phase_weaver.app.ui.main_window import MainWindow
from phase_weaver.app.ui.reconstruction_panel import ReconstructionPanel
from phase_weaver.app.ui.toy_model_panel import ToyModelPanel


def _app():
    return QApplication.instance() or QApplication([])


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


def test_reconstruction_panel_provides_measurement_and_reconstruction_state():
    _app()
    panel = ReconstructionPanel()

    measurement_state = panel.get_measurement_state()
    reconstruction_state = panel.get_reconstruction_state()

    assert measurement_state.crisp is False
    assert measurement_state.infrared is False
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


def test_main_window_exposes_file_actions_in_file_menu():
    _app()
    window = MainWindow()

    assert window.file_menu.title() == "File"
    assert [action.text() for action in window.file_menu.actions()] == [
        "Load Measurements...",
        "Export Data...",
    ]


def test_main_window_hosts_setup_panels_in_external_windows():
    _app()
    window = MainWindow()

    assert window.toy_model_window.isHidden()
    assert window.toy_model_window.isWindow()
    assert window.toy_model_action.text() == "Toy Model"
    assert window.reconstruction_window.isHidden()
    assert window.reconstruction_window.isWindow()
    assert window.reconstruction_action.text() == "Reconstruction Setup"

    window.toy_model_action.setChecked(True)
    window.reconstruction_action.setChecked(True)

    assert window.toy_model_window.isVisible()
    assert window.reconstruction_window.isVisible()

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
