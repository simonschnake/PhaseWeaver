from PySide6.QtWidgets import QApplication

from phase_weaver.app.ui.plot_panel import MplCanvas


def test_mpl_canvas_uses_screen_friendly_dpi():
    app = QApplication.instance() or QApplication([])

    canvas = MplCanvas()

    assert canvas.figure.get_dpi() == 100
    assert app is not None
