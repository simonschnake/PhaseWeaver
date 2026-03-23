from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QWidget,
)

import phase_weaver.app.config as cfg

from ..state import MeasurementState

# assumes these already exist
from .control_box import ControlBox


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
                "scale": (
                    cfg.MEASUREMENT_SCALE_MIN,
                    cfg.MEASUREMENT_SCALE_MAX,
                    cfg.MEASUREMENT_SCALE_STEP,
                    cfg.MEASUREMENT_SCALE_DEFAULT,
                ),
            },
            parent=parent,
        )

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_buttons: dict[str, QPushButton] = {}

        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)

        self._enum_cls = cfg.MEASUREMENT_MODE
        self._id_to_enum: dict[int, cfg.MEASUREMENT_MODE] = {}

        for i, mode in enumerate(self._enum_cls):
            text = str(mode.value)
            btn = QPushButton(text)
            btn.setCheckable(True)
            if mode == cfg.MEASUREMENT_MODE_DEFAULT:
                btn.setChecked(True)
            mode_layout.addWidget(btn)
            self._mode_group.addButton(btn, i)
            self._id_to_enum[i] = mode

        self.form.insertRow(0, "Mode", mode_row)

        self._f_min_slider, self._f_min_spin = self._controls["f_min_thz"]
        self._f_max_slider, self._f_max_spin = self._controls["f_max_thz"]
        self._overlap_slider, self._overlap_spin = self._controls["overlap_width_thz"]
        self._scale_slider, self._scale_spin = self._controls["scale"]

        self._f_min_spin.setSuffix(" THz")
        self._f_max_spin.setSuffix(" THz")
        self._overlap_spin.setSuffix(" THz")

        self._mode_group.idClicked.connect(lambda _value: self.changed.emit())

    @property
    def mode(self) -> cfg.MEASUREMENT_MODE:
        button_id = self._mode_group.checkedId()
        if button_id < 0:
            raise RuntimeError("no mode is currently selected")
        return self._id_to_enum[button_id]

    def get_state(self) -> MeasurementState:
        f_min_hz = self._f_min_spin.value() * 1e12
        f_max_hz = self._f_max_spin.value() * 1e12
        overlap_width_hz = self._overlap_spin.value() * 1e12

        if (
            self.mode == cfg.MEASUREMENT_MODE.BETWEEN
            and f_max_hz <= f_min_hz + cfg.MEASUREMENT_RANGE_STEP_THZ * 1e12
        ):
            self._mode_group.button(0).setChecked(True)

        return MeasurementState(
            mode=self.mode,
            f_min_hz=f_min_hz,
            f_max_hz=f_max_hz,
            overlap_width_hz=overlap_width_hz,
            scale=self._scale_spin.value(),
        )
