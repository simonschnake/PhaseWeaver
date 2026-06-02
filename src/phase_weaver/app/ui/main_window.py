from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QWidget

from ..logic import AppLogic
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

        self.controls_panel = ControlsPanel()

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self.plot_panel, 1)
        layout.addWidget(self.controls_panel)
        self.setCentralWidget(central)
        self.resize(1800, 650)

        self.controls_panel.changed.connect(self.schedule_updates)
        self.controls_panel.export_requested.connect(self.export_requested)

        self.redraw_timer = QTimer(self)
        self.redraw_timer.setSingleShot(True)
        self.redraw_timer.timeout.connect(self.redraw_input)

        self.reconstruction_timer = QTimer(self)
        self.reconstruction_timer.setSingleShot(True)
        self.reconstruction_timer.timeout.connect(self.redraw_reconstruction)

        self.redraw_input()
        self.redraw_reconstruction()

    def schedule_updates(self) -> None:
        self.redraw_timer.start(10)
        self.reconstruction_timer.start(250)

    def redraw_input(self) -> None:
        state = self.controls_panel.get_state()
        prof_input = self.logic.compute_input_profile(state)
        ff_input = self.logic.compute_input_formfactor(prof_input)
        self.plot_panel.render_input(prof_input, ff_input)

    def redraw_reconstruction(self) -> None:
        state = self.controls_panel.get_state()
        prof_input = self.logic.compute_input_profile(state)
        ff_input = self.logic.compute_input_formfactor(prof_input)
        measurements = self.logic.compute_measured_formfactor(
            ff_input, state.measurement
        )
        prof_recon, ff_recon = self.logic.compute_reconstruction(
            grid=ff_input.grid,
            measurements=measurements,
            controls_state=state,
            ff_input=ff_input,
        )
        prof_recon.charge = prof_input.charge
        self.plot_panel.render_reconstruction(prof_recon, ff_recon)

    def export_requested(self) -> None:
        self.logic.export_npz(
            self.plot_panel.time_model,
            self.plot_panel.spectrum_model,
        )
