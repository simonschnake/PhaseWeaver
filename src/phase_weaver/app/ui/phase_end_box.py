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


class PhaseEndBox(QGroupBox):
    changed = Signal()

    def __init__(
        self,
        *,
        enabled: bool,
        start_freq_hz: float,
        phase_at_300_thz: float,
        ref_freq_hz: float = 300e12,
        parent=None,
    ):
        super().__init__("Linear Phase End", parent)
        self._ref_freq_hz = ref_freq_hz

        self.setCheckable(True)
        self.setChecked(enabled)
        self.toggled.connect(self.changed.emit)

        form = QFormLayout(self)

        f_max_thz = ref_freq_hz * 1e-12 - 0.1
        self._start_slider, self._start_spin = self._make_slider_spin(
            0.0, f_max_thz, 0.1, start_freq_hz * 1e-12
        )
        form.addRow(
            QLabel("start freq (THz)"),
            self._row(self._start_slider, self._start_spin),
        )

        self._phase_slider, self._phase_spin = self._make_slider_spin(
            -300.0, 300.0, 1.0, phase_at_300_thz
        )
        form.addRow(
            QLabel("phase @ 300 THz"),
            self._row(self._phase_slider, self._phase_spin),
        )

        self._start_slider.valueChanged.connect(self.changed.emit)
        self._start_spin.valueChanged.connect(self.changed.emit)
        self._phase_slider.valueChanged.connect(self.changed.emit)
        self._phase_spin.valueChanged.connect(self.changed.emit)

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
        spin.setDecimals(2 if step < 1 else 0)

        def tick_to_value(tick: int) -> float:
            return min_v + tick * step

        def value_to_tick(val: float) -> int:
            return int(round((val - min_v) / step))

        slider.valueChanged.connect(lambda tick: spin.setValue(tick_to_value(tick)))
        spin.valueChanged.connect(
            lambda val: slider.setValue(max(0, min(n_ticks, value_to_tick(val))))
        )

        with QSignalBlocker(slider), QSignalBlocker(spin):
            spin.setValue(float(np.clip(init, min_v, max_v)))
            slider.setValue(value_to_tick(spin.value()))

        return slider, spin

    def get_values(self) -> tuple[bool, float, float]:
        return (
            self.isChecked(),
            self._start_spin.value() * 1e12,
            self._phase_spin.value(),
        )

    def set_values(self, enabled: bool, start_freq_hz: float, phase_at_300_thz: float) -> None:
        with QSignalBlocker(self):
            self.setChecked(enabled)
        with QSignalBlocker(self._start_spin), QSignalBlocker(self._phase_spin):
            self._start_spin.setValue(start_freq_hz * 1e-12)
            self._phase_spin.setValue(phase_at_300_thz)