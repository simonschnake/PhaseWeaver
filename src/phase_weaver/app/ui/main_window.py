import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, Qt, QTimer
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

from .. import config as cfg
from ..logic import AppLogic, ReconstructionSummary, list_h5_measurement_shots
from ..state import ControlsState, MeasurementState, ProfileModelState, ReconstructionState
from ..utils import load_app_icon
from phase_weaver.model.profiles import AsymSuperGaussParams
from .measurement_time_dialog import MeasurementTimeDialog
from .plot_panel import PlotPanel
from .reconstruction_panel import ReconstructionPanel
from .toy_model_panel import ToyModelPanel


LAST_DIRECTORY_KEY = "paths/last_directory"
LAST_MEASUREMENT_PATH_KEY = "paths/last_measurement_path"
LAST_EXPORT_PATH_KEY = "paths/last_export_path"
CONTROLS_STATE_KEY = "controls/state"
AUTO_RECONSTRUCTION_KEY = "controls/auto_reconstruction"


class MainWindow(QMainWindow):
    def __init__(
        self, theme: APP_THEME = APP_THEME.DARK, settings: QSettings | None = None
    ):
        super().__init__()
        self.setWindowTitle("Phase Weaver")
        self.setWindowIcon(load_app_icon())
        self.theme = theme
        self.settings = settings or QSettings("PhaseWeaver", "Phase Weaver")

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
        self._restore_controls_settings()

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
        self._open_setup_windows()
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

    def _open_setup_windows(self) -> None:
        self.toy_model_action.setChecked(True)
        self.reconstruction_action.setChecked(True)

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
        self._save_controls_settings()
        self.redraw_timer.start(10)
        self.reconstruct_timer.stop()
        if self.logic.reconstruction_summary.status == "finished":
            self.logic.reconstruction_summary.status = "stale"
        self.plot_panel.clear_reconstruction()
        self.update_summary(self.logic.reconstruction_summary)
        if self.reconstruction_panel.is_auto_reconstruction_enabled():
            self.reconstruct_timer.start(200)

    def reconstruction_auto_toggled(self, enabled: bool) -> None:
        self.settings.setValue(AUTO_RECONSTRUCTION_KEY, enabled)
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
            self._dialog_start_path(LAST_MEASUREMENT_PATH_KEY),
            "Measurement files (*.npz *.h5 *.hdf5);;NumPy measurement bundle (*.npz);;HDF5 recording (*.h5 *.hdf5)",
        )
        if not path:
            return

        h5_shot_index = None
        try:
            if path.lower().endswith((".h5", ".hdf5")):
                shots = list_h5_measurement_shots(path)
                if len(shots) > 1:
                    dialog = MeasurementTimeDialog(shots, self)
                    if dialog.exec() != QDialog.DialogCode.Accepted:
                        return
                    h5_shot_index = dialog.selected_shot_index()

            self.logic.load_measurements(path, h5_shot_index=h5_shot_index)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load measurements", str(exc))
            return

        self._remember_path(path, LAST_MEASUREMENT_PATH_KEY)
        self.schedule_updates()

    def export_requested(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export PhaseWeaver data",
            self._export_start_path(),
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
        self._remember_path(path, LAST_EXPORT_PATH_KEY)

    def _dialog_start_path(self, key: str, fallback: str = "") -> str:
        value = self._settings_str(key)
        if value:
            path = Path(value)
            if path.exists():
                return str(path)
            if path.parent.exists():
                return str(path.parent)

        directory = self._settings_str(LAST_DIRECTORY_KEY)
        if directory and Path(directory).exists():
            return directory
        return fallback

    def _export_start_path(self) -> str:
        value = self._settings_str(LAST_EXPORT_PATH_KEY)
        if value:
            path = Path(value)
            if path.exists() or path.parent.exists():
                return str(path)

        directory = self._settings_str(LAST_DIRECTORY_KEY)
        if directory and Path(directory).exists():
            return str(Path(directory) / "phase_weaver_export.npz")
        return "phase_weaver_export.npz"

    def _remember_path(self, path: str, key: str) -> None:
        selected = Path(path)
        self.settings.setValue(key, str(selected))
        self.settings.setValue(LAST_DIRECTORY_KEY, str(selected.parent))

    def _restore_controls_settings(self) -> None:
        payload_text = self._settings_str(CONTROLS_STATE_KEY)
        if payload_text:
            try:
                state = self._controls_state_from_payload(json.loads(payload_text))
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                state = None
            if state is not None:
                self.toy_model_panel.set_scenario_state(state.scenario)
                self.reconstruction_panel.set_measurement_state(state.measurement)
                self.reconstruction_panel.set_reconstruction_state(state.reconstruction)

        auto_reconstruction = self.settings.value(AUTO_RECONSTRUCTION_KEY, None)
        if auto_reconstruction is not None:
            self.reconstruction_panel.set_auto_reconstruction_enabled(
                self._settings_bool(auto_reconstruction)
            )

    def _save_controls_settings(self) -> None:
        self.settings.setValue(
            CONTROLS_STATE_KEY,
            json.dumps(self._controls_state_payload(self.get_controls_state())),
        )

    def _controls_state_payload(self, state: ControlsState) -> dict[str, Any]:
        return {
            "scenario": {
                "background": self._gaussian_payload(state.scenario.background),
                "peak": self._gaussian_payload(state.scenario.peak),
                "peak2_enabled": state.scenario.peak2_enabled,
                "peak2": self._gaussian_payload(state.scenario.peak2),
            },
            "measurement": {
                "crisp": state.measurement.crisp,
                "infrared": state.measurement.infrared,
            },
            "reconstruction": {
                "phase_init_mode": state.reconstruction.phase_init_mode.name,
                "time_constraints": sorted(
                    mode.name for mode in state.reconstruction.time_constraints
                ),
                "frequency_constraints": sorted(
                    mode.name for mode in state.reconstruction.frequency_constraints
                ),
                "stop_conditions": sorted(
                    mode.name for mode in state.reconstruction.stop_conditions
                ),
            },
        }

    def _controls_state_from_payload(self, payload: Any) -> ControlsState:
        if not isinstance(payload, dict):
            raise ValueError("controls settings must be a mapping")

        scenario_payload = payload.get("scenario", {})
        measurement_payload = payload.get("measurement", {})
        reconstruction_payload = payload.get("reconstruction", {})

        scenario_default = ProfileModelState()
        scenario = ProfileModelState(
            background=self._gaussian_from_payload(
                scenario_payload.get("background"), scenario_default.background
            ),
            peak=self._gaussian_from_payload(
                scenario_payload.get("peak"), scenario_default.peak
            ),
            peak2_enabled=bool(
                scenario_payload.get("peak2_enabled", scenario_default.peak2_enabled)
            ),
            peak2=self._gaussian_from_payload(
                scenario_payload.get("peak2"), scenario_default.peak2
            ),
        )

        measurement = MeasurementState(
            crisp=bool(measurement_payload.get("crisp", False)),
            infrared=bool(measurement_payload.get("infrared", False)),
        )

        reconstruction = ReconstructionState(
            phase_init_mode=self._enum_from_name(
                cfg.PHASE_INIT_MODE,
                reconstruction_payload.get("phase_init_mode"),
                cfg.PHASE_INIT_DEFAULT,
            ),
            time_constraints=self._enum_set_from_names(
                cfg.RECON_TIME_CONSTRAINT,
                reconstruction_payload.get("time_constraints"),
                set(cfg.RECON_TIME_CONSTRAINT_DEFAULT),
            ),
            frequency_constraints=self._enum_set_from_names(
                cfg.RECON_FREQUENCY_CONSTRAINT,
                reconstruction_payload.get("frequency_constraints"),
                set(cfg.RECON_FREQUENCY_CONSTRAINT_DEFAULT),
            ),
            stop_conditions=self._enum_set_from_names(
                cfg.RECON_STOP_CONDITION,
                reconstruction_payload.get("stop_conditions"),
                set(cfg.RECON_STOP_CONDITION_DEFAULT),
            ),
        )

        return ControlsState(
            scenario=scenario,
            measurement=measurement,
            reconstruction=reconstruction,
        )

    def _gaussian_payload(self, params: AsymSuperGaussParams) -> dict[str, float]:
        return {
            "center": params.center,
            "width": params.width,
            "skew": params.skew,
            "order": params.order,
            "amplitude": params.amplitude,
        }

    def _gaussian_from_payload(
        self, payload: Any, default: AsymSuperGaussParams
    ) -> AsymSuperGaussParams:
        if not isinstance(payload, dict):
            return default
        return AsymSuperGaussParams(
            center=float(payload.get("center", default.center)),
            width=float(payload.get("width", default.width)),
            skew=float(payload.get("skew", default.skew)),
            order=float(payload.get("order", default.order)),
            amplitude=float(payload.get("amplitude", default.amplitude)),
        )

    def _enum_from_name(self, enum_cls, name: Any, default):
        if isinstance(name, str) and name in enum_cls.__members__:
            return enum_cls[name]
        return default

    def _enum_set_from_names(self, enum_cls, names: Any, default):
        if not isinstance(names, list):
            return default
        result = {
            enum_cls[name]
            for name in names
            if isinstance(name, str) and name in enum_cls.__members__
        }
        return result if result else default

    def _settings_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes"}
        return bool(value)

    def _settings_str(self, key: str) -> str:
        value = self.settings.value(key, "")
        return value if isinstance(value, str) else ""

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
