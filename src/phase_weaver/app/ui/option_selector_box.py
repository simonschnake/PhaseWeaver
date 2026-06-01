from enum import Enum
from typing import Type

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QGroupBox, QHBoxLayout, QPushButton


class _BaseOptionSelectorBox(QGroupBox):
    changed = Signal()

    def __init__(
        self,
        title: str,
        enum_cls: Type[Enum],
        parent=None,
    ):
        super().__init__(title, parent)

        self._enum_cls = enum_cls
        self._group = QButtonGroup(self)
        self._id_to_enum: dict[int, Enum] = {}
        self._enum_to_id: dict[Enum, int] = {}

        self._layout = QHBoxLayout(self)

        for i, member in enumerate(self._enum_cls):
            text = str(member.value)

            btn = QPushButton(text)
            btn.setCheckable(True)

            self._layout.addWidget(btn)
            self._group.addButton(btn, i)

            self._id_to_enum[i] = member
            self._enum_to_id[member] = i

    def _button_for_enum(self, option: Enum) -> QPushButton:
        button_id = self._enum_to_id.get(option)

        if button_id is None:
            raise ValueError(
                f"unknown option {option!r}; expected one of {list(self._enum_cls)!r}"
            )

        return self._group.button(button_id)


class OptionSelectorBox(_BaseOptionSelectorBox):
    """
    Selector where exactly one option can be active.
    """

    def __init__(
        self,
        title: str,
        enum_cls: Type[Enum],
        default: Enum | None = None,
        parent=None,
    ):
        super().__init__(title, enum_cls, parent)

        self._group.setExclusive(True)

        if default is not None:
            self.mode = default

        self._group.idClicked.connect(lambda _id: self.changed.emit())

    @property
    def mode(self) -> Enum:
        button_id = self._group.checkedId()

        if button_id < 0:
            raise RuntimeError("no option is currently selected")

        return self._id_to_enum[button_id]

    @mode.setter
    def mode(self, mode: Enum) -> None:
        old_mode = None

        try:
            old_mode = self.mode
        except RuntimeError:
            pass

        btn = self._button_for_enum(mode)
        btn.setChecked(True)

        if old_mode != mode:
            self.changed.emit()


class MultiOptionSelectorBox(_BaseOptionSelectorBox):
    """
    Selector where multiple options can be active.
    """

    def __init__(
        self,
        title: str,
        enum_cls: Type[Enum],
        default: set[Enum] | None = None,
        parent=None,
    ):
        super().__init__(title, enum_cls, parent)

        self._group.setExclusive(False)

        if default is not None:
            self.selected_modes = default

        self._group.buttonToggled.connect(lambda _button, _checked: self.changed.emit())

    @property
    def selected_modes(self) -> set[Enum]:
        selected: set[Enum] = set()

        for button in self._group.buttons():
            if button.isChecked():
                button_id = self._group.id(button)
                selected.add(self._id_to_enum[button_id])

        return selected

    @selected_modes.setter
    def selected_modes(self, modes: set[Enum]) -> None:
        unknown = set(modes) - set(self._enum_to_id)

        if unknown:
            raise ValueError(
                f"unknown options {unknown!r}; expected values from {list(self._enum_cls)!r}"
            )

        old_modes = self.selected_modes

        self._group.blockSignals(True)

        for member in self._enum_cls:
            btn = self._button_for_enum(member)
            btn.setChecked(member in modes)

        self._group.blockSignals(False)

        if old_modes != modes:
            self.changed.emit()
