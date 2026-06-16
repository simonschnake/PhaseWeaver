import sys

from PySide6.QtWidgets import QApplication

from phase_weaver.qt_theme import APP_THEME, set_app_theme
from phase_weaver.mpl_style import apply_mpl_style, sync_mpl_to_qt
from phase_weaver.app.ui.main_window import MainWindow

from .utils import load_app_icon

def main() -> int:
    app = QApplication(sys.argv)
    app.setWindowIcon(load_app_icon())
    app.setApplicationName("Phase Weaver")
    app.setApplicationDisplayName("Phase Weaver")

    set_app_theme(app, APP_THEME.DARK)
    apply_mpl_style(APP_THEME.DARK)
    sync_mpl_to_qt(app)

    w = MainWindow(theme=APP_THEME.DARK)
    w.setWindowTitle("Phase Weaver")
    w.setWindowIcon(load_app_icon())
    w.show()
    return app.exec()
