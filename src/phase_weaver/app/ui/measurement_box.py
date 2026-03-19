from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

from .control_box import ControlBox
from ..state import MeasurementState


class MeasurementBox(ControlBox):
    def __init__(self, initial: MeasurementState, parent: QWidget | None = None):

        specs = {}

        super().__init__(
            title="Measurement Range",
            specs=specs,
            checkable=True,
            checked=initial.enabled,
            parent=parent,
        )

        self.form.insertRow(
            0,
        )


class MeasurementBox(QGroupBox):
    changed = Signal()

    def __init__(self, initial: MeasurementState, parent: QWidget | None = None):
        super().__init__("Measurement", parent)

        self.enable_cb = QCheckBox("Use limited measurement range")
        self.enable_cb.setChecked(initial.enabled)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["full", "below_cut", "between"])
        self.mode_combo.setCurrentText(initial.mode)

        self.f_min_spin = QDoubleSpinBox()
        self.f_min_spin.setRange(0.0, 10_000.0)
        self.f_min_spin.setDecimals(3)
        self.f_min_spin.setSuffix(" THz")
        self.f_min_spin.setValue(initial.f_min_hz / 1e12)

        self.f_max_spin = QDoubleSpinBox()
        self.f_max_spin.setRange(0.0, 10_000.0)
        self.f_max_spin.setDecimals(3)
        self.f_max_spin.setSuffix(" THz")
        self.f_max_spin.setValue(initial.f_max_hz / 1e12)

        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setRange(0.0, 10_000.0)
        self.overlap_spin.setDecimals(3)
        self.overlap_spin.setSuffix(" THz")
        self.overlap_spin.setValue(initial.overlap_width_hz / 1e12)

        form = QFormLayout()
        form.addRow(self.enable_cb)
        form.addRow("Mode", self.mode_combo)
        form.addRow("Min frequency", self.f_min_spin)
        form.addRow("Max / cutoff frequency", self.f_max_spin)
        form.addRow("Overlap width", self.overlap_spin)

        layout = QVBoxLayout(self)
        layout.addLayout(form)

        self.enable_cb.toggled.connect(self._emit_changed)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.f_min_spin.valueChanged.connect(self._emit_changed)
        self.f_max_spin.valueChanged.connect(self._emit_changed)
        self.overlap_spin.valueChanged.connect(self._emit_changed)

        self._on_mode_changed(self.mode_combo.currentText())

    def _emit_changed(self, *args) -> None:
        self.changed.emit()

    def _on_mode_changed(self, mode: str) -> None:
        is_full = mode == "full"
        is_below_cut = mode == "below_cut"
        is_between = mode == "between"

        self.f_min_spin.setEnabled(is_between)
        self.f_max_spin.setEnabled(is_below_cut or is_between)
        self.overlap_spin.setEnabled(not is_full)

        self.changed.emit()

    def get_state(self) -> MeasurementState:
        f_min_hz = self.f_min_spin.value() * 1e12
        f_max_hz = self.f_max_spin.value() * 1e12

        if f_max_hz < f_min_hz:
            f_max_hz = f_min_hz

        return MeasurementState(
            enabled=self.enable_cb.isChecked(),
            mode=self.mode_combo.currentText(),
            f_min_hz=f_min_hz,
            f_max_hz=f_max_hz,
            overlap_width_hz=self.overlap_spin.value() * 1e12,
        )

