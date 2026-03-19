from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QPushButton,
    QWidget,
)

from .control_box import ControlBox
from ..state import MeasurementState


class MeasurementBox(ControlBox):
    changed = Signal()

    _MODES = ["full", "below_cut", "between"]

    def __init__(self, initial: MeasurementState, parent: QWidget | None = None):
        super().__init__(
            title="Measurement",
            specs={
                "f_min": (
                    0.0,
                    500,
                    0.1,
                    0.1,
                ),
                "f_max": (
                    0.0,
                    500,
                    0.1,
                    250,
                ),
                "overlap_width": (
                    0.0,
                    150,
                    0.1,
                    50,
                ),
            },
            checkable=False,  # important: do NOT disable the whole box
            parent=parent,
        )

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_buttons: dict[str, QPushButton] = {}

        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)

        for i, mode in enumerate(self._MODES):
            btn = QPushButton(mode)
            btn.setCheckable(True)
            if mode == initial.mode:
                btn.setChecked(True)
            mode_layout.addWidget(btn)
            self._mode_group.addButton(btn, i)
            self._mode_buttons[mode] = btn

        self.form.insertRow(0, "Mode", mode_row)

        self._f_min_slider, self._f_min_spin = self._controls["f_min"]
        self._f_max_slider, self._f_max_spin = self._controls["f_max"]
        self._overlap_slider, self._overlap_spin = self._controls["overlap_width"]

        self._f_min_spin.setSuffix(" THz")
        self._f_max_spin.setSuffix(" THz")
        self._overlap_spin.setSuffix(" THz")

        self._mode_group.idClicked.connect(lambda _id: self._on_mode_changed())

        self._on_enabled_changed()

    def _selected_mode(self) -> str:
        idx = self._mode_group.checkedId()
        if idx < 0:
            return self._MODES[0]
        return self._MODES[idx]

    def _set_pair_enabled(self, slider, spin, enabled: bool) -> None:
        slider.setEnabled(enabled)
        spin.setEnabled(enabled)

    def _on_enabled_changed(self, *_args) -> None:
        mode = self._selected_mode()

        is_full = mode == "full"
        is_below_cut = mode == "below_cut"
        is_between = mode == "between"

        self._set_pair_enabled(self._f_min_slider, self._f_min_spin, is_between)
        self._set_pair_enabled(
            self._f_max_slider, self._f_max_spin, is_below_cut or is_between
        )
        self._set_pair_enabled(self._overlap_slider, self._overlap_spin, not is_full)

        self.changed.emit()

    def _on_mode_changed(self) -> None:
        self._on_enabled_changed()

    def get_mode(self) -> str:
        return self._selected_mode()

    def set_mode(self, mode: str) -> None:
        btn = self._mode_buttons.get(mode)
        if btn is not None:
            with QSignalBlocker(self._mode_group):
                btn.setChecked(True)
        self._on_mode_changed()

    def get_state(self) -> MeasurementState:
        f_min_hz = self._f_min_spin.value() * 1e12
        f_max_hz = self._f_max_spin.value() * 1e12

        if f_max_hz < f_min_hz:
            f_max_hz = f_min_hz

        return MeasurementState(
            enabled=self.get_mode() != "full",
            mode=self.get_mode(),
            f_min_hz=f_min_hz,
            f_max_hz=f_max_hz,
            overlap_width_hz=self._overlap_spin.value() * 1e12,
        )
