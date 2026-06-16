from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

import phase_weaver.app.config as cfg

from ..state import (
    MeasurementState,
    ReconstructionState,
)
from .option_selector_box import MultiOptionSelectorBox, OptionSelectorBox


class ControlsPanel(QWidget):
    changed = Signal()
    export_requested = Signal()
    measurements_load_requested = Signal()
    reconstruction_requested = Signal()
    reconstruction_auto_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        # self.mag_init_box = OptionSelectorBox(
        #    "Magnitude Init", enum_cls=cfg.MAG_INIT_MODE, default=cfg.MAG_INIT_DEFAULT
        # )
        self.phase_init_box = OptionSelectorBox(
            "Phase Init", enum_cls=cfg.PHASE_INIT_MODE, default=cfg.PHASE_INIT_DEFAULT
        )

        self.measurement_box = MultiOptionSelectorBox(
            title="Measurements", enum_cls=cfg.MEASUREMENT_MODE
        )
        self.time_constraint_box = MultiOptionSelectorBox(
            title="Time Constraints",
            enum_cls=cfg.RECON_TIME_CONSTRAINT,
            default=set(cfg.RECON_TIME_CONSTRAINT_DEFAULT),
        )
        self.frequency_constraint_box = MultiOptionSelectorBox(
            title="Frequency Constraints",
            enum_cls=cfg.RECON_FREQUENCY_CONSTRAINT,
            default=set(cfg.RECON_FREQUENCY_CONSTRAINT_DEFAULT),
        )
        self.stop_condition_box = MultiOptionSelectorBox(
            title="Stop Conditions",
            enum_cls=cfg.RECON_STOP_CONDITION,
            default=set(cfg.RECON_STOP_CONDITION_DEFAULT),
        )

        # self.band_limit_box = BandLimitBox()

        for widget in (
            self.phase_init_box,
            self.measurement_box,
            self.time_constraint_box,
            self.frequency_constraint_box,
            self.stop_condition_box,
            # self.mag_init_box,
            # self.band_limit_box,
        ):
            widget.changed.connect(self.changed.emit)

        layout = QVBoxLayout(self)
        # layout.addWidget(self.mag_init_box)
        layout.addWidget(self.phase_init_box)
        layout.addWidget(self.measurement_box)
        layout.addWidget(self.time_constraint_box)
        layout.addWidget(self.frequency_constraint_box)
        layout.addWidget(self.stop_condition_box)
        # layout.addWidget(self.band_limit_box)

        self.load_measurements_button = QPushButton("Load Measurements")
        self.load_measurements_button.clicked.connect(
            self.measurements_load_requested.emit
        )
        layout.addWidget(self.load_measurements_button)

        self.auto_reconstruct_button = QPushButton("Auto Reconstruction: On")
        self.auto_reconstruct_button.setCheckable(True)
        self.auto_reconstruct_button.setChecked(True)
        self.auto_reconstruct_button.toggled.connect(self._update_auto_button_text)
        self.auto_reconstruct_button.toggled.connect(
            self.reconstruction_auto_toggled.emit
        )
        layout.addWidget(self.auto_reconstruct_button)

        self.reconstruct_button = QPushButton("Run Reconstruction")
        self.reconstruct_button.clicked.connect(self.reconstruction_requested.emit)
        layout.addWidget(self.reconstruct_button)

        self.export_button = QPushButton("Export Data")
        self.export_button.clicked.connect(self.export_requested.emit)
        layout.addWidget(self.export_button)
        layout.addStretch(1)

    def get_reconstruction_state(self) -> ReconstructionState:
        # phase_end_enabled, start_freq_hz, phase_at_300_thz = (
        # self.phase_end_box.get_values()
        #  )
        # band_limit_enabled, f_cut_hz = self.band_limit_box.get_values()

        return ReconstructionState(
            # mag_init_mode=self.mag_init_box.mode,
            phase_init_mode=self.phase_init_box.mode,
            time_constraints=set(self.time_constraint_box.selected_modes),
            frequency_constraints=set(self.frequency_constraint_box.selected_modes),
            stop_conditions=set(self.stop_condition_box.selected_modes),
            # band_limit=band_limit_enabled,
            # band_limit_f_cut_hz=f_cut_hz,
        )

    def get_measurement_state(self) -> MeasurementState:
        meas_state = MeasurementState()
        for mode in self.measurement_box.selected_modes:
            if mode == cfg.MEASUREMENT_MODE.CRISP:
                meas_state.crisp = True
            if mode == cfg.MEASUREMENT_MODE.INFRARED:
                meas_state.infrared = True

        return meas_state

    def is_auto_reconstruction_enabled(self) -> bool:
        return self.auto_reconstruct_button.isChecked()

    def _update_auto_button_text(self, enabled: bool) -> None:
        self.auto_reconstruct_button.setText(
            "Auto Reconstruction: On" if enabled else "Auto Reconstruction: Off"
        )
