from enum import Enum
from typing import Type

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QGroupBox, QHBoxLayout, QPushButton


class OptionSelectorBox(QGroupBox):
    changed = Signal()

    def __init__(
        self, title: str, enum_cls: Type[Enum], default: Enum | None = None, parent=None
    ):
        super().__init__(title, parent)

        self._enum_cls = enum_cls
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._id_to_enum: dict[int, Enum] = {}
        self._enum_to_id: dict[Enum, int] = {}

        layout = QHBoxLayout(self)

        for i, member in enumerate(self._enum_cls):
            text = str(member.value)
            btn = QPushButton(text)
            btn.setCheckable(True)

            if default is not None and default == member:
                btn.setChecked(True)

            layout.addWidget(btn)
            self._group.addButton(btn, i)
            self._id_to_enum[i] = member
            self._enum_to_id[member] = i

        self._group.idClicked.connect(lambda _id: self.changed.emit())

    @property
    def mode(self) -> Enum:
        button_id = self._group.checkedId()
        if button_id < 0:
            raise RuntimeError("no option is currently selected")
        return self._id_to_enum[button_id]

    @mode.setter
    def mode(self, mode: Enum) -> None:
        button_id = self._enum_to_id.get(mode, None)
        if button_id is None:
            raise ValueError(
                f"unknown option {mode!r}; expected one of {self._enum_cls!r}"
            )

        btn = self._group.button(button_id)
        btn.setChecked(True)
