from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from phase_weaver.app.logic import H5MeasurementShot


class MeasurementTimeDialog(QDialog):
    def __init__(
        self,
        shots: tuple[H5MeasurementShot, ...],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Select Measurement Time")
        self._shots = shots

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Zeitpunkt aus den Messdaten"))

        self.time_combo = QComboBox()
        for shot in shots:
            self.time_combo.addItem(
                f"{shot.measured_at}  (shot {shot.index})",
                shot.index,
            )

        if shots:
            latest_combo_index = max(
                range(len(shots)), key=lambda index: shots[index].timestamp
            )
            self.time_combo.setCurrentIndex(latest_combo_index)

        form = QFormLayout()
        form.addRow("Zeitpunkt", self.time_combo)
        layout.addLayout(form)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def selected_shot_index(self) -> int:
        return int(self.time_combo.currentData())
