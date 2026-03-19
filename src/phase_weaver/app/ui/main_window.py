from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QWidget
from PySide6.QtCore import QTimer

import phase_weaver.app.config as cfg
from ..state import ProfileModelState

from ..logic import AppLogic
from ..state import AppState
from ..utils import load_app_icon
from .controls_panel import ControlsPanel
from .plot_panel import PlotPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phase Weaver")
        self.setWindowIcon(load_app_icon())

        self.logic = AppLogic()
        self.plot_panel: PlotPanel = PlotPanel()

        self._base_scenario_state: ProfileModelState = ProfileModelState(
            dt=cfg.DT,
            t_max=cfg.T_MAX,
            charge=cfg.CHARGE_C,
        )

        spike2_spec = {
            **cfg.SPIKE_SPEC,
            "center_fs": (-20.0, 20.0, 1.0, 10.0),
        }

        f_nyq_hz = 1.0 / (2.0 * cfg.DT)

        self.controls_panel = ControlsPanel(
            background_spec=cfg.BACKGROUND_SPEC,
            spike_spec=cfg.SPIKE_SPEC,
            spike2_spec=spike2_spec,
            phase_init_labels=cfg.PHASE_INIT_MODES,
            phase_init_default=cfg.PHASE_INIT_DEFAULT,
            phase_end_ref_freq_hz=cfg.PHASE_END_REF_FREQ_HZ,
            f_nyq_hz=f_nyq_hz,
            time_unit=cfg.TIME_UNIT,
            t_to_s=cfg.T_TO_S,
        )

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self.plot_panel, 1)
        layout.addWidget(self.controls_panel)
        self.setCentralWidget(central)
        self.resize(1800, 650)

        self.controls_panel.changed.connect(self.schedule_updates)

        self.redraw_timer = QTimer(self)
        self.redraw_timer.setSingleShot(True)
        self.redraw_timer.timeout.connect(self.redraw_input)

        self.reconstruction_timer = QTimer(self)
        self.reconstruction_timer.setSingleShot(True)
        self.reconstruction_timer.timeout.connect(self.redraw_reconstruction)

        self.redraw_input()
        self.redraw_reconstruction()

    def get_app_state(self) -> AppState:
        default_state = AppState.default()
        return AppState(
            scenario=self.controls_panel.get_scenario_state(self._base_scenario_state),
            measurement=default_state.measurement,  # TODO: add measurement controls
            reconstruction=self.controls_panel.get_reconstruction_state(),
        )

    def schedule_updates(self) -> None:
        self.redraw_timer.start(10)
        self.reconstruction_timer.start(250)

    def redraw_input(self) -> None:
        state = self.get_app_state()
        prof_input, ff_input, measurement = self.logic.compute_initial(app_state=state)
        self.plot_panel.render_input(prof_input, ff_input)

    def redraw_reconstruction(self) -> None:
        state = self.get_app_state()
        prof_input, ff_input, measurement = self.logic.compute_initial(app_state=state)
        prof_recon, ff_recon = self.logic.compute_reconstruction(
            app_state=state,
            measured_ff=measurement,
            grid=ff_input.grid,
        )
        prof_recon.charge = prof_input.charge
        self.plot_panel.render_reconstruction(prof_recon, ff_recon)