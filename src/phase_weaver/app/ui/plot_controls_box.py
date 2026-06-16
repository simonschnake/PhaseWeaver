from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QWidget

import phase_weaver.app.config as cfg


class PlotControlsBox(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._line_actions: dict[cfg.PLOT_LINE_MODE, QAction] = {}
        self._fwhm_action = QAction("FWHM", self)
        self._fwhm_action.setCheckable(True)
        self._fwhm_action.setChecked(False)
        self._fwhm_action.toggled.connect(lambda _checked: self._line_visibility_changed())

        for mode in cfg.PLOT_LINE_MODE:
            action = QAction(str(mode.value), self)
            action.setCheckable(True)
            action.setChecked(mode in cfg.PLOT_LINE_MODE_DEFAULT)
            action.toggled.connect(lambda _checked: self._line_visibility_changed())
            self._line_actions[mode] = action

    @property
    def visible_lines(self) -> set[cfg.PLOT_LINE_MODE]:
        return {
            mode
            for mode, action in self._line_actions.items()
            if action.isChecked()
        }

    @property
    def line_actions(self) -> dict[cfg.PLOT_LINE_MODE, QAction]:
        return self._line_actions

    @property
    def fwhm_action(self) -> QAction:
        return self._fwhm_action

    @property
    def show_fwhm(self) -> bool:
        return self._fwhm_action.isChecked()

    def show_all_lines(self) -> None:
        for action in self._line_actions.values():
            action.blockSignals(True)
            action.setChecked(True)
            action.blockSignals(False)
        self.changed.emit()

    def _line_visibility_changed(self) -> None:
        self.changed.emit()
