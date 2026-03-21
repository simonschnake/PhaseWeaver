from __future__ import annotations

from phase_weaver.model.profiles import AsymSuperGaussParams
from .control_box import ControlBox
from PySide6.QtCore import QSignalBlocker


from ..config import T_TO_S, S_TO_T


class GaussianGroup(ControlBox):
    def __init__(
        self,
        title: str,
        specs: dict[str, tuple[float, float, float, float]],
        include_center: bool = True,
        checkable: bool = False,
        checked: bool = True,
        parent=None,
    ) -> None:
        self._include_center = include_center

        if not include_center:
            specs.pop("center_fs", None)

        super().__init__(title, specs, checkable, checked, parent)

    def get_params(self) -> AsymSuperGaussParams:
        center = 0.0
        if self._include_center:
            center = self._controls["center_fs"][1].value() * T_TO_S

        width = self._controls["width_fs"][1].value() * T_TO_S
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
            "center_fs": params.center * S_TO_T,
            "width_fs": params.width * S_TO_T,
            "skew": params.skew,
            "order": params.order,
            "amplitude": params.amplitude,
        }
        if self._include_center:
            values["center_fs"] = params.center * S_TO_T

        for key, value in values.items():
            slider, spin = self._controls[key]
            min_v = float(slider.property("_min_v"))
            step = float(slider.property("_step"))
            tick = int(round((value - min_v) / step))
            tick = max(slider.minimum(), min(slider.maximum(), tick))
            with QSignalBlocker(slider), QSignalBlocker(spin):
                slider.setValue(tick)
                spin.setValue(value)
