import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QWidget,
)


class BandLimitBox(QGroupBox):
    changed = Signal()

    def __init__(
        self,
        *,
        enabled: bool,
        f_cut_hz: float,
        f_nyq_hz: float,
        parent=None,
    ):
        super().__init__("Band-limit Transform", parent)

        self.setCheckable(True)
        self.setChecked(enabled)
        self.toggled.connect(lambda _checked: self.changed.emit())

        form = QFormLayout(self)
        fmax_thz = max(min(333.0, f_nyq_hz * 1e-12 - 0.1), 0.2)
        init_thz = float(np.clip(f_cut_hz * 1e-12, 0.1, 10 * fmax_thz))

        self._slider, self._spin = self._make_slider_spin(
            0.1, 10 * fmax_thz, 1.0, init_thz
        )
        form.addRow(QLabel("f_cut (THz)"), self._row(self._slider, self._spin))

        self._slider.valueChanged.connect(lambda _checked: self.changed.emit())
        self._spin.valueChanged.connect(lambda _checked: self.changed.emit())

    def _row(self, slider: QSlider, spin: QDoubleSpinBox) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider, 1)
        h.addWidget(spin)
        return w

    def _make_slider_spin(self, min_v: float, max_v: float, step: float, init: float):
        n_ticks = max(int(round((max_v - min_v) / step)), 1)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, n_ticks)
        slider.setProperty("_min_v", float(min_v))
        slider.setProperty("_step", float(step))

        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setSingleStep(step)
        spin.setDecimals(0)

        def tick_to_value(tick: int) -> float:
            return min_v + tick * step

        def value_to_tick(val: float) -> int:
            return int(round((val - min_v) / step))

        slider.valueChanged.connect(lambda tick: spin.setValue(tick_to_value(tick)))
        spin.valueChanged.connect(
            lambda val: slider.setValue(max(0, min(n_ticks, value_to_tick(val))))
        )

        with QSignalBlocker(slider), QSignalBlocker(spin):
            spin.setValue(init)
            slider.setValue(value_to_tick(init))

        return slider, spin

    def get_values(self) -> tuple[bool, float]:
        return self.isChecked(), max(self._spin.value(), 0.1) * 1e12

    def set_values(self, enabled: bool, f_cut_hz: float) -> None:
        with QSignalBlocker(self):
            self.setChecked(enabled)
        with QSignalBlocker(self._spin):
            self._spin.setValue(f_cut_hz * 1e-12)
