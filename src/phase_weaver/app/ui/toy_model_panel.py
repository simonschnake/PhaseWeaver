from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

import phase_weaver.app.config as cfg

from ..state import ProfileModelState
from .gaussian_group import GaussianGroup


class ToyModelPanel(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.background_box = GaussianGroup(
            "Background",
            specs=cfg.BACKGROUND_SPEC,
            include_center=False,
        )
        self.spike_box = GaussianGroup(
            "Spike",
            specs=cfg.SPIKE_SPEC,
        )
        self.spike2_box = GaussianGroup(
            "Spike 2",
            specs=cfg.SPIKE_SPEC,
            checkable=True,
            checked=cfg.PEAK2_ENABLED,
        )

        for widget in (
            self.background_box,
            self.spike_box,
            self.spike2_box,
        ):
            widget.changed.connect(self.changed.emit)

        layout = QVBoxLayout(self)
        layout.addWidget(self.background_box)
        layout.addWidget(self.spike_box)
        layout.addWidget(self.spike2_box)
        layout.addStretch(1)

    def get_scenario_state(self) -> ProfileModelState:
        return ProfileModelState(
            background=self.background_box.get_params(),
            peak=self.spike_box.get_params(),
            peak2_enabled=self.spike2_box.is_enabled_component(),
            peak2=self.spike2_box.get_params(),
        )

    def set_scenario_state(self, state: ProfileModelState) -> None:
        self.background_box.set_params(state.background)
        self.spike_box.set_params(state.peak)
        self.spike2_box.set_enabled_component(state.peak2_enabled)
        self.spike2_box.set_params(state.peak2)
