from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ..state import ProfileModelState

from ..state import ReconstructionState, AppState
from .band_limit_box import BandLimitBox
from .gaussian_group import GaussianGroup
from .phase_end_box import PhaseEndBox
from .phase_init_box import PhaseInitBox
from .measurement_box import MeasurementBox

import phase_weaver.app.config as cfg


class ControlsPanel(QWidget):
    changed = Signal()

    def __init__(
        self,
        *,
        background_spec: dict,
        spike_spec: dict,
        spike2_spec: dict,
        phase_init_labels: list[str],
        phase_init_default: str,
        phase_end_ref_freq_hz: float,
        f_nyq_hz: float,
        time_unit: str = "fs",
        t_to_s: float = 1e-15,
        parent=None,
    ):
        super().__init__(parent)

        self.background_box = GaussianGroup(
            "Background",
            background_spec,
            include_center=False,
            time_unit=time_unit,
            t_to_s=t_to_s,
        )
        self.spike_box = GaussianGroup(
            "Spike",
            spike_spec,
            include_center=True,
            time_unit=time_unit,
            t_to_s=t_to_s,
        )
        self.spike2_box = GaussianGroup(
            "Spike 2",
            spike2_spec,
            include_center=True,
            time_unit=time_unit,
            t_to_s=t_to_s,
            checkable=True,
            checked=False,
        )

        self.phase_init_box = PhaseInitBox(
            phase_init_labels,
            default=phase_init_default,
        )
        self.phase_end_box = PhaseEndBox(
            enabled=False,
            start_freq_hz=0.5 * phase_end_ref_freq_hz,
            phase_at_300_thz=0.0,
            ref_freq_hz=phase_end_ref_freq_hz,
        )
        self.band_limit_box = BandLimitBox(
            enabled=False,
            f_cut_hz=250e12,
            f_nyq_hz=f_nyq_hz,
        )

        default_state = AppState.default()
        self.measurement_box = MeasurementBox(default_state.measurement)

        for widget in (
            self.background_box,
            self.spike_box,
            self.spike2_box,
            self.phase_end_box,
            self.band_limit_box,
            self.measurement_box,
            self.phase_init_box,
        ):
            widget.changed.connect(self.changed.emit)

        peak_col = QVBoxLayout()
        peak_col.addWidget(self.background_box)
        peak_col.addWidget(self.spike_box)
        peak_col.addWidget(self.spike2_box)
        peak_col.addStretch(1)

        recon_col = QVBoxLayout()
        recon_col.addWidget(self.phase_init_box)
        recon_col.addWidget(self.phase_end_box)
        recon_col.addWidget(self.band_limit_box)
        recon_col.addWidget(self.measurement_box)
        recon_col.addStretch(1)

        layout = QHBoxLayout(self)
        layout.addLayout(peak_col, 1)
        layout.addLayout(recon_col, 1)

    def get_scenario_state(self, base: ProfileModelState) -> ProfileModelState:
        state = ProfileModelState(
            dt=base.dt,
            t_max=base.t_max,
            charge=base.charge,
            background=self.background_box.get_params(),
            peak=self.spike_box.get_params(),
            peak2_enabled=self.spike2_box.is_enabled_component(),
            peak2=self.spike2_box.get_params(),
        )
        return state

    def get_reconstruction_state(self) -> ReconstructionState:
        phase_end_enabled, start_freq_hz, phase_at_300_thz = (
            self.phase_end_box.get_values()
        )
        band_limit_enabled, f_cut_hz = self.band_limit_box.get_values()

        return ReconstructionState(
            phase_init_mode=self.phase_init_box.get_mode(),
            linear_phase_end=phase_end_enabled,
            linear_phase_end_start_freq_hz=start_freq_hz,
            linear_phase_end_phase_at_300_thz=phase_at_300_thz,
            band_limit=band_limit_enabled,
            band_limit_f_cut_hz=f_cut_hz,
            initial_gaussian_sigma_s=cfg.INITIAL_GAUSSIAN_SIGMA_S,
        )

    def set_reconstruction_state(self, state: ReconstructionState) -> None:
        self.phase_init_box.set_mode(state.phase_init_mode)
        self.phase_end_box.set_values(
            state.linear_phase_end,
            state.linear_phase_end_start_freq_hz,
            state.linear_phase_end_phase_at_300_thz,
        )
        self.band_limit_box.set_values(
            state.band_limit,
            state.band_limit_f_cut_hz,
        )

    def set_scenario_state(self, state: ProfileModelState) -> None:
        self.background_box.set_params(state.background)
        self.spike_box.set_params(state.peak)
        self.spike2_box.set_enabled_component(state.peak2_enabled)
        self.spike2_box.set_params(state.peak2)
