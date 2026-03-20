from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QPushButton,
    QWidget,
)

from ..state import MeasurementState

# assumes these already exist
from .control_box import ControlBox


class MeasurementBox(ControlBox):
    changed = Signal()

    _MODES = ["full", "below_cut", "between"]

    def __init__(self, initial: MeasurementState, parent: QWidget | None = None):
        super().__init__(
            title="Measurement",
            specs={
                "f_min_hz": (
                    0.0,
                    10_000.0,
                    0.001,
                    initial.f_min_hz / 1e12,
                ),
                "f_max_hz": (
                    0.0,
                    10_000.0,
                    0.001,
                    initial.f_max_hz / 1e12,
                ),
                "overlap_width_hz": (
                    0.0,
                    10_000.0,
                    0.001,
                    initial.overlap_width_hz / 1e12,
                ),
            },
            checkable=False,  # important: do NOT disable the whole box
            parent=parent,
        )

        self.enable_cb = QCheckBox("Use limited measurement range")
        self.enable_cb.setChecked(initial.enabled)

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

        self.form.insertRow(0, self.enable_cb)
        self.form.insertRow(1, "Mode", mode_row)

        self._f_min_slider, self._f_min_spin = self._controls["f_min_hz"]
        self._f_max_slider, self._f_max_spin = self._controls["f_max_hz"]
        self._overlap_slider, self._overlap_spin = self._controls["overlap_width_hz"]

        self._f_min_spin.setSuffix(" THz")
        self._f_max_spin.setSuffix(" THz")
        self._overlap_spin.setSuffix(" THz")

        self.enable_cb.toggled.connect(self._on_enabled_changed)
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
        enabled = self.enable_cb.isChecked()
        mode = self._selected_mode()

        if not enabled:
            self._set_pair_enabled(self._f_min_slider, self._f_min_spin, False)
            self._set_pair_enabled(self._f_max_slider, self._f_max_spin, False)
            self._set_pair_enabled(self._overlap_slider, self._overlap_spin, False)
            self.changed.emit()
            return

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
            enabled=self.enable_cb.isChecked(),
            mode=self.get_mode(),
            f_min_hz=f_min_hz,
            f_max_hz=f_max_hz,
            overlap_width_hz=self._overlap_spin.value() * 1e12,
        )
