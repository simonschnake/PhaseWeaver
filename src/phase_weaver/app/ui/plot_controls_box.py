from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

import phase_weaver.app.config as cfg

from .option_selector_box import MultiOptionSelectorBox, OptionSelectorBox


class PlotControlsBox(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.time_mode_box = OptionSelectorBox(
            title="Time plot",
            enum_cls=cfg.TIME_PLOT_MODE,
            default=cfg.TIME_PLOT_MODE_DEFAULT,
        )

        self.lines_box = MultiOptionSelectorBox(
            title="Visible lines",
            enum_cls=cfg.PLOT_LINE_MODE,
            default=cfg.PLOT_LINE_MODE_DEFAULT,
        )

        self.show_all_button = QPushButton("Show all")
        self.show_all_button.clicked.connect(self.show_all_lines)

        self.time_mode_box.changed.connect(self.changed.emit)
        self.lines_box.changed.connect(self.changed.emit)

        layout = QHBoxLayout(self)
        layout.addWidget(self.time_mode_box)
        layout.addWidget(self.lines_box, 1)
        layout.addWidget(self.show_all_button)

    @property
    def time_mode(self) -> cfg.TIME_PLOT_MODE:
        return self.time_mode_box.mode

    @property
    def visible_lines(self) -> set[cfg.PLOT_LINE_MODE]:
        return self.lines_box.selected_modes

    def show_all_lines(self) -> None:
        self.lines_box.selected_modes = cfg.PLOT_LINE_MODE_DEFAULT
        self.changed.emit()
