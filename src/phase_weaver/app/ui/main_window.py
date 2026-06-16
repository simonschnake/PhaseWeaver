from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from phase_weaver.mpl_style import apply_mpl_style, sync_mpl_to_qt
from phase_weaver.qt_theme import APP_THEME, set_app_theme

from ..logic import AppLogic, ReconstructionSummary
from ..utils import load_app_icon
from .controls_panel import ControlsPanel
from .plot_panel import PlotPanel


class MainWindow(QMainWindow):
    def __init__(self, theme: APP_THEME = APP_THEME.DARK):
        super().__init__()
        self.setWindowTitle("Phase Weaver")
        self.setWindowIcon(load_app_icon())
        self.theme = theme

        self.logic = AppLogic()
        self.plot_panel: PlotPanel = PlotPanel(theme=theme)

        self.controls_panel = ControlsPanel()

        central = QWidget()
        layout = QVBoxLayout(central)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.addWidget(self.plot_panel, 1)
        body_layout.addWidget(self.controls_panel)

        self.summary_label = QLabel()
        self.summary_label.setTextInteractionFlags(self.summary_label.textInteractionFlags())
        self.summary_label.setWordWrap(True)

        layout.addWidget(body, 1)
        layout.addWidget(self.summary_label)
        self.setCentralWidget(central)
        self.resize(1800, 650)

        self.controls_panel.changed.connect(self.schedule_updates)
        self.controls_panel.export_requested.connect(self.export_requested)
        self.controls_panel.measurements_load_requested.connect(
            self.load_measurements_requested
        )
        self.controls_panel.reconstruction_requested.connect(
            self.run_reconstruction_requested
        )

        self.redraw_timer = QTimer(self)
        self.redraw_timer.setSingleShot(True)
        self.redraw_timer.timeout.connect(self.redraw_input)

        self._create_menus()
        self.redraw_input()
        self.update_summary(self.logic.reconstruction_summary)

    def _create_menus(self) -> None:
        view_menu = self.menuBar().addMenu("View")
        theme_menu = view_menu.addMenu("Theme")
        lines_menu = view_menu.addMenu("Lines")

        self.theme_action_group = QActionGroup(self)
        self.theme_action_group.setExclusive(True)
        self.theme_actions: dict[APP_THEME, QAction] = {}

        for theme in APP_THEME:
            action = QAction(theme.value, self)
            action.setCheckable(True)
            action.setChecked(theme == self.theme)
            action.triggered.connect(
                lambda _checked=False, selected=theme: self.set_theme(selected)
            )
            self.theme_action_group.addAction(action)
            theme_menu.addAction(action)
            self.theme_actions[theme] = action

        self.line_action_group = QActionGroup(self)
        self.line_action_group.setExclusive(False)
        self.line_actions = {}

        for mode, action in self.plot_panel.plot_controls.line_actions.items():
            self.line_action_group.addAction(action)
            lines_menu.addAction(action)
            self.line_actions[mode] = action

        lines_menu.addSeparator()
        show_all_action = QAction("Show all", self)
        show_all_action.triggered.connect(self.plot_panel.plot_controls.show_all_lines)
        lines_menu.addAction(show_all_action)

    def set_theme(self, theme: APP_THEME) -> None:
        if theme == self.theme:
            return

        app = QApplication.instance()
        if not isinstance(app, QApplication):
            return

        self.theme = theme
        set_app_theme(app, theme)
        apply_mpl_style(theme)
        sync_mpl_to_qt(app, self.plot_panel.canvas.figure)
        self.plot_panel.set_theme(theme)
        self.theme_actions[theme].setChecked(True)
        central = self.centralWidget()
        if central is not None:
            central.updateGeometry()
            layout = central.layout()
            if layout is not None:
                layout.invalidate()
                layout.activate()
        self.updateGeometry()
        self.repaint()

    def schedule_updates(self) -> None:
        self.redraw_timer.start(10)
        if self.logic.reconstruction_summary.status == "finished":
            self.logic.reconstruction_summary.status = "stale"
        self.plot_panel.clear_reconstruction()
        self.update_summary(self.logic.reconstruction_summary)

    def redraw_input(self) -> None:
        state = self.controls_panel.get_state()
        prof_input = self.logic.compute_input_profile(state)
        ff_input = self.logic.compute_input_formfactor(prof_input)
        self.plot_panel.render_input(prof_input, ff_input)
        self.plot_panel.render_measurements(self.logic.loaded_measurements)

    def run_reconstruction_requested(self) -> None:
        state = self.controls_panel.get_state()
        prof_input = self.logic.compute_input_profile(state)
        ff_input = self.logic.compute_input_formfactor(prof_input)
        measurements, source = self.logic.active_measurements(ff_input, state.measurement)

        running_summary = ReconstructionSummary(
            measurement_source=source,
            measurement_count=len(measurements),
            status="running",
        )
        self.update_summary(running_summary)

        try:
            prof_recon, ff_recon, summary = self.logic.compute_reconstruction(
                grid=ff_input.grid,
                measurements=measurements,
                controls_state=state,
                ff_input=ff_input,
                measurement_source=source,
            )
        except Exception as exc:
            failed_summary = ReconstructionSummary(
                measurement_source=source,
                measurement_count=len(measurements),
                status="failed",
                stop_reason=str(exc),
            )
            self.logic.reconstruction_summary = failed_summary
            self.update_summary(failed_summary)
            QMessageBox.critical(self, "Reconstruction failed", str(exc))
            return

        prof_recon.charge = prof_input.charge
        self.plot_panel.render_input(prof_input, ff_input)
        self.plot_panel.render_measurements(self.logic.loaded_measurements)
        self.plot_panel.render_reconstruction(prof_recon, ff_recon)
        self.update_summary(summary)

    def load_measurements_requested(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load measurement bundle",
            "",
            "NumPy measurement bundle (*.npz)",
        )
        if not path:
            return

        try:
            measurements = self.logic.load_measurements(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load measurements", str(exc))
            return

        self.plot_panel.clear_reconstruction()
        self.plot_panel.render_measurements(measurements)
        self.update_summary(self.logic.reconstruction_summary)

    def export_requested(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export PhaseWeaver data",
            "phase_weaver_export.npz",
            "NumPy archive (*.npz)",
        )
        if not path:
            return
        if not path.lower().endswith(".npz"):
            path = f"{path}.npz"

        self.logic.export_npz(
            path,
            self.plot_panel.time_model,
            self.plot_panel.spectrum_model,
            controls_state=self.controls_panel.get_state(),
            summary=self.logic.reconstruction_summary,
        )

    def update_summary(self, summary: ReconstructionSummary) -> None:
        error = (
            "n/a"
            if summary.measurement_error is None
            else f"{summary.measurement_error:.3e}"
        )
        self.summary_label.setText(
            " | ".join(
                (
                    f"source: {summary.measurement_source}",
                    f"measurements: {summary.measurement_count}",
                    f"state: {summary.status}",
                    f"iterations: {summary.iterations}",
                    f"stop: {summary.stop_reason}",
                    f"error: {error}",
                )
            )
        )
