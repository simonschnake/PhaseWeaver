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
        prof_input, ff_input, _ = self.logic.compute_initial(controls_state=state)
        self.plot_panel.render_input(prof_input, ff_input)

    def redraw_reconstruction(self) -> None:
        state = self.controls_panel.get_state()
        prof_input, ff_input, measurement = self.logic.compute_initial(
            controls_state=state
        )
        prof_recon, ff_recon = self.logic.compute_reconstruction(
            controls_state=state,
            measured_ff=measurement,
            ff_input=ff_input,
        )
        prof_recon.charge = prof_input.charge
        self.plot_panel.render_reconstruction(prof_recon, ff_recon)

    def export_requested(self) -> None:
        self.logic.export_npz(
            self.plot_panel.time_model,
            self.plot_panel.spectrum_model,
        )
