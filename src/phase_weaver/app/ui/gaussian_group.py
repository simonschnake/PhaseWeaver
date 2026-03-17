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

from phase_weaver.model.profiles import AsymSuperGaussParams


class GaussianGroup(QGroupBox):
    changed = Signal()

    def __init__(
        self,
        title: str,
        specs: dict,
        *,
        include_center: bool = True,
        time_unit: str = "fs",
        t_to_s: float = 1e-15,
        checkable: bool = False,
        checked: bool = True,
        parent=None,
    ):
        super().__init__(title, parent)
        self._specs = specs
        self._include_center = include_center
        self._time_unit = time_unit
        self._t_to_s = t_to_s
        self._controls: dict[str, tuple[QSlider, QDoubleSpinBox]] = {}

        self.setCheckable(checkable)
        if checkable:
            self.setChecked(checked)
            self.toggled.connect(lambda _checked: self.changed.emit())

        form = QFormLayout(self)

        for spec_name, spec in specs.items():
            if spec_name == "center_fs" and not include_center:
                continue

            min_v, max_v, step_v, init_v = spec
            spec_title = spec_name.replace("_", " ")
            slider, spin = self._make_slider_spin_row(min_v, max_v, step_v, init_v)
            form.addRow(QLabel(spec_title), self._row(slider, spin))
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

    def get_params(self) -> AsymSuperGaussParams:
        center = 0.0
        if self._include_center:
            center = self._controls["center_fs"][1].value() * self._t_to_s

        width = self._controls["width_fs"][1].value() * self._t_to_s
        skew = self._controls["skew"][1].value()
        order = self._controls["order"][1].value()
        amplitude = self._controls["amplitude"][1].value()

        return AsymSuperGaussParams(
            center=center,
            width=width,
            skew=skew,
            order=order,
            amplitude=amplitude,
        )

    def set_params(self, params: AsymSuperGaussParams) -> None:
        values = {
            "center_fs": params.center / self._t_to_s,
            "width_fs": params.width / self._t_to_s,
            "skew": params.skew,
            "order": params.order,
            "amplitude": params.amplitude,
        }
        if self._include_center:
            values["center_fs"] = params.center / self._t_to_s

        for key, value in values.items():
            slider, spin = self._controls[key]
            min_v = float(slider.property("_min_v"))
            step = float(slider.property("_step"))
            tick = int(round((value - min_v) / step))
            tick = max(slider.minimum(), min(slider.maximum(), tick))
            with QSignalBlocker(slider), QSignalBlocker(spin):
                slider.setValue(tick)
                spin.setValue(value)

    def _row(self, slider: QSlider, spin: QDoubleSpinBox) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider, 1)
        h.addWidget(spin)
        return w
