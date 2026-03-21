from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QGroupBox, QHBoxLayout, QPushButton


class OptionSelectorBox(QGroupBox):
    changed = Signal()

    def __init__(self, title: str, labels: list[str], default: str, parent=None):
        if not labels:
            raise ValueError("labels must not be empty")
        if default not in labels:
            raise ValueError(f"default {default!r} is not in labels")
        if len(set(labels)) != len(labels):
            raise ValueError("labels must be unique")

        super().__init__(title, parent)

        self._labels = list(labels)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        layout = QHBoxLayout(self)
        for i, label in enumerate(self._labels):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(label == default)
            layout.addWidget(btn)
            self._group.addButton(btn, i)

        self._group.idClicked.connect(lambda _id: self.changed.emit())

    @property
    def mode(self) -> str:
        idx = self._group.checkedId()
        if idx < 0:
            raise RuntimeError("no option is currently selected")
        return self._labels[idx]

    @mode.setter
    def mode(self, option: str) -> None:
        try:
            idx = self._labels.index(option)
        except ValueError as e:
            raise ValueError(
                f"unknown option {option!r}; expected one of {self._labels!r}"
            ) from e

        btn = self._group.button(idx)
        if btn is None:
            raise RuntimeError(f"no button found for option index {idx}")

        btn.setChecked(True)
