from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSlider,
    QHBoxLayout,
    QWidget,
)


class ControlBox(QGroupBox):
    changed = Signal()

    def __init__(
        self,
        title: str,
        specs: dict[
            str,
            tuple[
                float,
                float,
                float,
                float,
            ],
        ],
        checkable: bool = False,
        checked: bool = True,
        parent=None,
    ):
        super().__init__(title, parent)
        self._specs = specs
        self._controls: dict[str, tuple[QSlider, QDoubleSpinBox]] = {}

        self.setCheckable(checkable)
        if checkable:
            self.setChecked(checked)
            self.toggled.connect(lambda _checked: self.changed.emit())

        self.form = QFormLayout(self)

        for spec_name, spec in specs.items():
            min_v, max_v, step_v, init_v = spec
            spec_title = spec_name.replace("_", " ")
            slider, spin = self._make_slider_spin_row(min_v, max_v, step_v, init_v)
            self.form.addRow(QLabel(spec_title), self._row(slider, spin))
            self._controls[spec_name] = (slider, spin)
            slider.valueChanged.connect(lambda _value: self.changed.emit())
            spin.valueChanged.connect(lambda _value: self.changed.emit())

    def _make_slider_spin_row(
        self, min_v: float, max_v: float, step: float, init: float
    ):
        n_ticks = int(round((max_v - min_v) / step))
        n_ticks = max(n_ticks, 1)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, n_ticks)
        slider.setProperty("_min_v", float(min_v))
        slider.setProperty("_step", float(step))

        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setSingleStep(step)

        if step < 1:
            dec = int(np.ceil(-np.log10(step)))
            spin.setDecimals(min(max(dec, 2), 10))
        else:
            spin.setDecimals(0)

        def tick_to_value(tick: int) -> float:
            return min_v + tick * step

        def value_to_tick(val: float) -> int:
            return int(round((val - min_v) / step))

        def slider_to_spin(tick: int):
            with QSignalBlocker(spin):
                spin.setValue(tick_to_value(tick))

        def spin_to_slider(val: float):
            tick = value_to_tick(val)
            tick = max(0, min(n_ticks, tick))
            with QSignalBlocker(slider):
                slider.setValue(tick)

        slider.valueChanged.connect(slider_to_spin)
        spin.valueChanged.connect(spin_to_slider)

        init = float(np.clip(init, min_v, max_v))
        with QSignalBlocker(slider), QSignalBlocker(spin):
            spin.setValue(init)
            slider.setValue(value_to_tick(init))

        return slider, spin

    def is_enabled_component(self) -> bool:
        return True if not self.isCheckable() else self.isChecked()

    def set_enabled_component(self, enabled: bool) -> None:
        if self.isCheckable():
            with QSignalBlocker(self):
                self.setChecked(enabled)

    def _row(self, slider: QSlider, spin: QDoubleSpinBox) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider, 1)
        h.addWidget(spin)
        return w
