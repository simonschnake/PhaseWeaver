from PySide6.QtWidgets import QApplication

from phase_weaver.app.ui.controls_panel import ControlsPanel


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
