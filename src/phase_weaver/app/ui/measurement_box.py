from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QWidget,
)

from ..state import MeasurementState

# assumes these already exist
from .control_box import ControlBox

import phase_weaver.app.config as cfg


class MeasurementBox(ControlBox):
    changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(
            title="Measurement",
            specs={
                "f_min_thz": (
                    cfg.MEASUREMENT_RANGE_MIN_THZ,
                    cfg.MEASUREMENT_RANGE_MAX_THZ,
                    cfg.MEASUREMENT_RANGE_STEP_THZ,
                    cfg.MEASUREMENT_F_MIN_DEFAULT_THZ,
                ),
                "f_max_thz": (
                    cfg.MEASUREMENT_RANGE_MIN_THZ,
                    cfg.MEASUREMENT_RANGE_MAX_THZ,
                    cfg.MEASUREMENT_RANGE_STEP_THZ,
                    cfg.MEASUREMENT_F_MAX_DEFAULT_THZ,
                ),
                "overlap_width_thz": (
                    cfg.MEASUREMENT_RANGE_MIN_THZ,
                    cfg.MEASUREMENT_RANGE_MAX_THZ,
                    cfg.MEASUREMENT_RANGE_STEP_THZ,
                    cfg.MEASUREMENT_OVERLAP_WIDTH_DEFAULT_THZ,
                ),
            },
            checkable=True,
            checked=cfg.MEASUREMENT_ENABLED,
            parent=parent,
        )

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_buttons: dict[str, QPushButton] = {}

        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)

        self._modes = cfg.MEASUREMENT_MODES

        for i, mode in enumerate(self._modes):
            btn = QPushButton(mode)
            btn.setCheckable(True)
            if mode == cfg.MEASUREMENT_MODE_DEFAULT:
                btn.setChecked(True)
            mode_layout.addWidget(btn)
            self._mode_group.addButton(btn, i)
            self._mode_buttons[mode] = btn

        self.form.insertRow(0, "Mode", mode_row)

        self._f_min_slider, self._f_min_spin = self._controls["f_min_thz"]
        self._f_max_slider, self._f_max_spin = self._controls["f_max_thz"]
        self._overlap_slider, self._overlap_spin = self._controls["overlap_width_thz"]

        self._f_min_spin.setSuffix(" THz")
        self._f_max_spin.setSuffix(" THz")
        self._overlap_spin.setSuffix(" THz")

        self._mode_group.idClicked.connect(lambda _value: self.changed.emit())

    @property
    def mode(self) -> str:
        idx = self._mode_group.checkedId()
        if idx < 0:
            return self._modes[0]
        return self._modes[idx]

    def get_state(self) -> MeasurementState:
        f_min_hz = self._f_min_spin.value() * 1e12
        f_max_hz = self._f_max_spin.value() * 1e12
        overlap_width_hz = self._overlap_spin.value() * 1e12

        if f_max_hz <= f_min_hz:
            self._f_max_spin.setValue(
                self._f_min_spin.value() + cfg.MEASUREMENT_RANGE_STEP_THZ
            )
            self.setChecked(False)

        return MeasurementState(
            enabled=self.isChecked(),
            mode=self.mode,
            f_min_hz=f_min_hz,
            f_max_hz=f_max_hz,
            overlap_width_hz=overlap_width_hz,
        )
