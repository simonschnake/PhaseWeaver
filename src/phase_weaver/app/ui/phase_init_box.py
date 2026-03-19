from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QGroupBox, QHBoxLayout, QPushButton


class PhaseInitBox(QGroupBox):
    changed = Signal()

    def __init__(self, labels: list[str], default: str, parent=None):
        super().__init__("Phase Init", parent)
        self._labels = labels
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        layout = QHBoxLayout(self)
        for i, label in enumerate(labels):
            btn = QPushButton(label)
            btn.setCheckable(True)
            if label == default:
                btn.setChecked(True)
            layout.addWidget(btn)
            self._group.addButton(btn, i)

        self._group.idClicked.connect(lambda _checked: self.changed.emit)

    def get_mode(self) -> str:
        idx = self._group.checkedId()
        return self._labels[idx]

    def set_mode(self, mode: str) -> None:
        idx = self._labels.index(mode)
        btn = self._group.button(idx)
        if btn is not None:
            btn.setChecked(True)
