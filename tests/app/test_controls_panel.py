from PySide6.QtWidgets import QApplication

from phase_weaver.app.ui.controls_panel import ControlsPanel
from phase_weaver.app.ui.main_window import MainWindow
from phase_weaver.app.ui.toy_model_panel import ToyModelPanel


def _app():
    return QApplication.instance() or QApplication([])


def test_controls_panel_auto_reconstruction_toggle_updates_state_and_label():
    _app()
    panel = ControlsPanel()

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


def test_main_window_hosts_toy_model_in_hidden_view_dock():
    _app()
    window = MainWindow()

    assert window.toy_model_dock.isHidden()
    assert window.toy_model_dock.toggleViewAction().text() == "Toy Model"
    assert window.get_controls_state().scenario == (
        window.toy_model_panel.get_scenario_state()
    )
