from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

import phase_weaver.app.config as cfg

from .option_selector_box import MultiOptionSelectorBox


class PlotControlsBox(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.lines_box = MultiOptionSelectorBox(
            title="Visible lines",
            enum_cls=cfg.PLOT_LINE_MODE,
            default=cfg.PLOT_LINE_MODE_DEFAULT,
        )

        self.show_all_button = QPushButton("Show all")
        self.show_all_button.clicked.connect(self.show_all_lines)

        self.lines_box.changed.connect(self.changed.emit)

        layout = QVBoxLayout(self)
        layout.addWidget(self.lines_box)
        layout.addWidget(self.show_all_button)
        layout.addStretch(1)

    @property
    def visible_lines(self) -> set[cfg.PLOT_LINE_MODE]:
        return self.lines_box.selected_modes

    def show_all_lines(self) -> None:
        self.lines_box.selected_modes = cfg.PLOT_LINE_MODE_DEFAULT
        self.changed.emit()
