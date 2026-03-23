from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

import phase_weaver.app.config as cfg

from ..state import (
    MeasurementState,
    ProfileModelState,
    ReconstructionState,
    ControlsState,
)
from .band_limit_box import BandLimitBox
from .gaussian_group import GaussianGroup
from .measurement_box import MeasurementBox
from .option_selector_box import OptionSelectorBox


class ControlsPanel(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.background_box = GaussianGroup(
            "Background",
            specs=cfg.BACKGROUND_SPEC,
            include_center=False,
        )
        self.spike_box = GaussianGroup(
            "Spike",
            specs=cfg.SPIKE_SPEC,
        )
        self.spike2_box = GaussianGroup(
            "Spike 2",
            specs=cfg.SPIKE_SPEC,
            checkable=True,
            checked=cfg.PEAK2_ENABLED,
        )

        self.mag_init_box = OptionSelectorBox(
            "Magnitude Init", enum_cls=cfg.MAG_INIT_MODE, default=cfg.MAG_INIT_DEFAULT
        )
        self.phase_init_box = OptionSelectorBox(
            "Phase Init", enum_cls=cfg.PHASE_INIT_MODE, default=cfg.PHASE_INIT_DEFAULT
        )

        self.measurement_box = MeasurementBox()

        self.band_limit_box = BandLimitBox()

        for widget in (
            self.background_box,
            self.spike_box,
            self.spike2_box,
            self.measurement_box,
            self.mag_init_box,
            self.phase_init_box,
            self.band_limit_box,
        ):
            widget.changed.connect(self.changed.emit)

        peak_col = QVBoxLayout()
        peak_col.addWidget(self.background_box)
        peak_col.addWidget(self.spike_box)
        peak_col.addWidget(self.spike2_box)
        peak_col.addStretch(1)

        recon_col = QVBoxLayout()
        recon_col.addWidget(self.mag_init_box)
        recon_col.addWidget(self.phase_init_box)
        recon_col.addWidget(self.measurement_box)
        recon_col.addWidget(self.band_limit_box)
        recon_col.addStretch(1)

        layout = QHBoxLayout(self)
        layout.addLayout(peak_col, 1)
        layout.addLayout(recon_col, 1)

    def get_scenario_state(self) -> ProfileModelState:
        state = ProfileModelState(
            background=self.background_box.get_params(),
            peak=self.spike_box.get_params(),
            peak2_enabled=self.spike2_box.is_enabled_component(),
            peak2=self.spike2_box.get_params(),
        )
        return state

    def get_reconstruction_state(self) -> ReconstructionState:
        # phase_end_enabled, start_freq_hz, phase_at_300_thz = (
        # self.phase_end_box.get_values()
        #  )
        band_limit_enabled, f_cut_hz = self.band_limit_box.get_values()

        return ReconstructionState(
            mag_init_mode=self.mag_init_box.mode,
            phase_init_mode=self.phase_init_box.mode,
            band_limit=band_limit_enabled,
            band_limit_f_cut_hz=f_cut_hz,
        )

    def get_measurement_state(self) -> MeasurementState:
        return self.measurement_box.get_state()

    def get_state(self) -> ControlsState:
        return ControlsState(
            scenario=self.get_scenario_state(),
            measurement=self.get_measurement_state(),
            reconstruction=self.get_reconstruction_state(),
        )
