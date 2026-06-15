from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu, QPushButton, QVBoxLayout, QWidget

import phase_weaver.app.config as cfg


class PlotControlsBox(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumWidth(100)

        self.lines_menu = QMenu(self)
        self._line_actions: dict[cfg.PLOT_LINE_MODE, QAction] = {}

        for mode in cfg.PLOT_LINE_MODE:
            action = QAction(str(mode.value), self)
            action.setCheckable(True)
            action.setChecked(mode in cfg.PLOT_LINE_MODE_DEFAULT)
            action.toggled.connect(lambda _checked: self._line_visibility_changed())
            self.lines_menu.addAction(action)
            self._line_actions[mode] = action

        self.lines_button = QPushButton(self)
        self.lines_button.setMenu(self.lines_menu)

        self.show_all_button = QPushButton("All")
        self.show_all_button.clicked.connect(self.show_all_lines)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.lines_button)
        layout.addWidget(self.show_all_button)
        layout.addStretch(1)

        self._update_button_text()

    @property
    def visible_lines(self) -> set[cfg.PLOT_LINE_MODE]:
        return {
            mode
            for mode, action in self._line_actions.items()
            if action.isChecked()
        }

    def show_all_lines(self) -> None:
        for action in self._line_actions.values():
            action.blockSignals(True)
            action.setChecked(True)
            action.blockSignals(False)
        self._update_button_text()
        self.changed.emit()

    def _line_visibility_changed(self) -> None:
        self._update_button_text()
        self.changed.emit()

    def _update_button_text(self) -> None:
        selected = len(self.visible_lines)
        total = len(self._line_actions)
        self.lines_button.setText(f"Lines {selected}/{total}")
