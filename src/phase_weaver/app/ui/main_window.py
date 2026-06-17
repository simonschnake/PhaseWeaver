from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from phase_weaver.qt_theme import APP_THEME, set_app_theme

from ..logic import AppLogic, ReconstructionSummary
from ..state import ControlsState
from ..utils import load_app_icon
from .plot_panel import PlotPanel
from .reconstruction_panel import ReconstructionPanel
from .toy_model_panel import ToyModelPanel


class MainWindow(QMainWindow):
    def __init__(self, theme: APP_THEME = APP_THEME.DARK):
        super().__init__()
        self.setWindowTitle("Phase Weaver")
        self.setWindowIcon(load_app_icon())
        self.theme = theme

        self.logic = AppLogic()
        self.plot_panel: PlotPanel = PlotPanel(theme=theme)

        self.reconstruction_panel = ReconstructionPanel()
        self.toy_model_panel = ToyModelPanel()
        self.reconstruct_timer = QTimer(self)
        self.reconstruct_timer.setSingleShot(True)
        self.reconstruct_timer.timeout.connect(self.run_reconstruction_requested)

        central = QWidget()
        layout = QVBoxLayout(central)

        self.summary_label = QLabel()
        self.summary_label.setTextInteractionFlags(self.summary_label.textInteractionFlags())
        self.summary_label.setWordWrap(True)

        layout.addWidget(self.plot_panel, 1)
        layout.addWidget(self.summary_label)
        self.setCentralWidget(central)
        self._create_toy_model_window()
        self._create_reconstruction_window()
        self.resize(1800, 650)

        self.reconstruction_panel.changed.connect(self.schedule_updates)
        self.reconstruction_panel.reconstruction_auto_toggled.connect(
            self.reconstruction_auto_toggled
        )
        self.reconstruction_panel.reconstruction_requested.connect(
            self.run_reconstruction_requested
        )
        self.toy_model_panel.changed.connect(self.schedule_updates)

        self.redraw_timer = QTimer(self)
        self.redraw_timer.setSingleShot(True)
        self.redraw_timer.timeout.connect(self.redraw_input)

        self._create_menus()
        self.schedule_updates()
        self.update_summary(self.logic.reconstruction_summary)

    def _create_toy_model_window(self) -> None:
        self.toy_model_window = QDialog(self)
        self.toy_model_window.setWindowTitle("Toy Model")
        self.toy_model_window.setWindowIcon(self.windowIcon())
        self.toy_model_window.setModal(False)
        self.toy_model_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        layout = QVBoxLayout(self.toy_model_window)
        layout.addWidget(self.toy_model_panel)
        self.toy_model_window.resize(520, 650)

        self.toy_model_action = QAction("Toy Model", self)
        self.toy_model_action.setCheckable(True)
        self.toy_model_action.toggled.connect(self._set_toy_model_window_visible)
        self.toy_model_window.finished.connect(
            lambda _result: self.toy_model_action.setChecked(False)
        )

    def _set_toy_model_window_visible(self, visible: bool) -> None:
        if visible:
            self.toy_model_window.show()
            self.toy_model_window.raise_()
            self.toy_model_window.activateWindow()
        else:
            self.toy_model_window.hide()

    def _create_reconstruction_window(self) -> None:
        self.reconstruction_window = QDialog(self)
        self.reconstruction_window.setWindowTitle("Reconstruction Setup")
        self.reconstruction_window.setWindowIcon(self.windowIcon())
        self.reconstruction_window.setModal(False)
        self.reconstruction_window.setAttribute(
            Qt.WidgetAttribute.WA_DeleteOnClose, False
        )
        layout = QVBoxLayout(self.reconstruction_window)
        layout.addWidget(self.reconstruction_panel)
        self.reconstruction_window.resize(520, 650)

        self.reconstruction_action = QAction("Reconstruction Setup", self)
        self.reconstruction_action.setCheckable(True)
        self.reconstruction_action.toggled.connect(
            self._set_reconstruction_window_visible
        )
        self.reconstruction_window.finished.connect(
            lambda _result: self.reconstruction_action.setChecked(False)
        )

    def _set_reconstruction_window_visible(self, visible: bool) -> None:
        if visible:
            self.reconstruction_window.show()
            self.reconstruction_window.raise_()
            self.reconstruction_window.activateWindow()
        else:
            self.reconstruction_window.hide()

    def _create_menus(self) -> None:
        self.file_menu = QMenu("File", self)
        self.menuBar().addMenu(self.file_menu)
        load_measurements_action = QAction("Load Measurements...", self)
        load_measurements_action.triggered.connect(self.load_measurements_requested)
        self.file_menu.addAction(load_measurements_action)

        export_action = QAction("Export Data...", self)
        export_action.triggered.connect(self.export_requested)
        self.file_menu.addAction(export_action)

        self.view_menu = QMenu("View", self)
        self.menuBar().addMenu(self.view_menu)
        self.view_menu.addAction(self.toy_model_action)
        self.view_menu.addAction(self.reconstruction_action)
        self.view_menu.addSeparator()
        theme_menu = self.view_menu.addMenu("Theme")
        lines_menu = self.view_menu.addMenu("Lines")
        self.fwhm_action = self.plot_panel.plot_controls.fwhm_action
        self.view_menu.addAction(self.fwhm_action)
        self.view_menu.addSeparator()

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

    def get_controls_state(self) -> ControlsState:
        return ControlsState(
            scenario=self.toy_model_panel.get_scenario_state(),
            measurement=self.reconstruction_panel.get_measurement_state(),
            reconstruction=self.reconstruction_panel.get_reconstruction_state(),
        )

    def set_theme(self, theme: APP_THEME) -> None:
        if theme == self.theme:
            return

        app = QApplication.instance()
        if not isinstance(app, QApplication):
            return

        self.theme = theme
        set_app_theme(app, theme)
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
        self.reconstruct_timer.stop()
        if self.logic.reconstruction_summary.status == "finished":
            self.logic.reconstruction_summary.status = "stale"
        self.plot_panel.clear_reconstruction()
        self.update_summary(self.logic.reconstruction_summary)
        if self.reconstruction_panel.is_auto_reconstruction_enabled():
            self.reconstruct_timer.start(200)

    def reconstruction_auto_toggled(self, enabled: bool) -> None:
        if enabled:
            self.schedule_updates()
        else:
            self.reconstruct_timer.stop()
            self.update_summary(self.logic.reconstruction_summary)

    def redraw_input(self) -> None:
        state = self.get_controls_state()
        prof_input = self.logic.compute_input_profile(state)
        ff_input = self.logic.compute_input_formfactor(prof_input)
        self.plot_panel.render_input(prof_input, ff_input)
        self.plot_panel.render_measurements(
            self.logic.visible_measurements(ff_input, state.measurement)
        )

    def run_reconstruction_requested(self) -> None:
        state = self.get_controls_state()
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
        self.plot_panel.render_measurements(
            self.logic.visible_measurements(ff_input, state.measurement)
        )
        self.plot_panel.render_reconstruction(prof_recon, ff_recon)
        self.update_summary(summary)

    def load_measurements_requested(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load measurement bundle",
            "",
            "Measurement files (*.npz *.h5 *.hdf5);;NumPy measurement bundle (*.npz);;HDF5 recording (*.h5 *.hdf5)",
        )
        if not path:
            return

        try:
            self.logic.load_measurements(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load measurements", str(exc))
            return

        self.schedule_updates()

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
            controls_state=self.get_controls_state(),
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
