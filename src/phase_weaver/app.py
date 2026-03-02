from dataclasses import dataclass
from importlib.resources import files
import sys

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from phase_weaver.core import DCPhysicalRFFT
from phase_weaver.core.constraints import (
    CenterFirstMoment,
    ClampMagnitude,
    EnforceDCOne,
    NonNegativity,
    NormalizeArea,
    ReplacePhaseEndLinear,  # adjust if your class has a different name
)
from phase_weaver.core.reconstruction import (
    GSMagnitudeOnly,
    MaxIter,
    MinIter,
    MinimumPhaseCepstrum,
    PhaseStoppedChanging,
    PredefinedPhase,
    RandomPhase,
    ZeroPhase,
)
from phase_weaver.model.profile_model import ProfileModel, ProfileModelState
from phase_weaver.mpl_style import apply_mpl_style, sync_mpl_to_qt
from phase_weaver.qt_theme import set_dark_theme

# -----------------------------
# Constants / UI units
# -----------------------------
CHARGE_C = 250e-12  # 250 pC
DT = 1e-16
T_MAX = 1e-13

TIME_UNIT = "fs"
S_TO_T = 1e15
T_TO_S = 1e-15

PHASE_INIT_LABELS = ["zero", "random", "minphase", "last"]
PHASE_INIT_DEFAULT_INDEX = 2  # minphase

SPIKE_SPEC = {
    "center_fs": (-20.0, 20.0, 1.0, -10.0),
    "width_fs": (0.1, 10.0, 0.005, 0.721),
    "skew": (-1.0, 1.0, 0.1, 0.0),
    "order": (0.01, 4.0, 0.01, 0.5),
    "amplitude": (0.0, 1e5, 100.0, 13000.0),
}

BACKGROUND_SPEC = {
    "width_fs": (1.0, 100.0, 0.5, 21.91914604893585),
    "skew": (-1.0, 1.0, 0.1, 0.0),
    "order": (1e-8, 10.0, 0.10, 1.0),
    "amplitude": (0.0, 1e5, 100.0, 5819.0),
}

C_NM_THz = 299792.458
F_EPS_THz = 1e-6
L_EPS_NM = 1e-6
PHASE_END_REF_FREQ_HZ = 300e12


@dataclass
class ReconUiState:
    phase_init_mode: str = "minphase"
    linear_phase_end: bool = False
    linear_phase_end_start_freq_hz: float = 0.0
    linear_phase_end_phase_at_300_thz: float = 0.0


def thz_to_nm(f_thz):
    f = np.asarray(f_thz, dtype=float)
    f = np.maximum(f, F_EPS_THz)
    return C_NM_THz / f


def nm_to_thz(l_nm):
    wavelength = np.asarray(l_nm, dtype=float)
    wavelength = np.maximum(wavelength, L_EPS_NM)
    return C_NM_THz / wavelength


def load_app_icon() -> QIcon:
    try:
        return QIcon(":/icon.png")
    except Exception:
        return QIcon()

class MplCanvas(FigureCanvas):
    def __init__(self):
        fig = Figure(constrained_layout=True)
        self.ax_time = fig.add_subplot(1, 2, 1)
        self.ax_mag = fig.add_subplot(1, 2, 2)
        self.ax_phase = self.ax_mag.twinx()
        super().__init__(fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phase Weaver")
        self.setWindowIcon(load_app_icon())

        # --- Model ---
        self.model = ProfileModel(
            ProfileModelState(
                dt=DT,
                t_max=T_MAX,
                charge=CHARGE_C,
            )
        )
        self.center_prof = CenterFirstMoment()

        # --- Reconstruction core ---
        self._recon_transform = DCPhysicalRFFT(unwrap_phase=True, dc_normalize=False)
        self._recon_time_constraints = (
            NonNegativity() + NormalizeArea() + CenterFirstMoment()
        )
        self._recon_stop_condition = (
            MaxIter(1_000) + PhaseStoppedChanging(tol=1e-8, patience=5) + MinIter(10)
        )

        # --- UI state for reconstruction options ---
        self._ui_state = ReconUiState(
            phase_init_mode=PHASE_INIT_LABELS[PHASE_INIT_DEFAULT_INDEX],
            linear_phase_end=False,
            linear_phase_end_start_freq_hz=0.5 * self.model.grid.f_pos[-1],
            linear_phase_end_phase_at_300_thz=0.0,
        )

        # --- registries ---
        self._controls: dict[tuple[str, str], tuple[QSlider, QDoubleSpinBox]] = {}
        self._selector_groups: dict[str, QButtonGroup] = {}

        # --- reconstruction cache ---
        self._I_input = None
        self._I_recon = None
        self._mag_pos = None
        self._phase_recon = None
        self._phase_in = None
        self._phase_recon_last = None

        # --- Plot ---
        self.canvas = MplCanvas()
        self.canvas.setStyleSheet("background: #1e1e1e;")
        self.toolbar = NavigationToolbar(self.canvas, self)

        (self.line_current,) = self.canvas.ax_time.plot(
            [], [], label="input", linewidth=2.0
        )
        (self.line_recon,) = self.canvas.ax_time.plot(
            [], [], label="reconstructed", linestyle="--", alpha=0.9
        )
        self.canvas.ax_time.legend(loc="upper right", fontsize="small")
        self.canvas.ax_time.set_xlabel(f"t ({TIME_UNIT})")
        self.canvas.ax_time.set_ylabel("Current (kA)")

        (self.line_mag,) = self.canvas.ax_mag.plot([], [], label="|F|", color="C0")
        (self.line_phase_in,) = self.canvas.ax_phase.plot(
            [], [], label="phase (input)", color="C1"
        )
        (self.line_phase_recon,) = self.canvas.ax_phase.plot(
            [], [], label="phase (recon)", color="C2", linestyle="--", alpha=0.9
        )

        self.canvas.ax_mag.set_xlabel("f (THz)")
        self.canvas.ax_mag.set_ylabel("|F(f)|", color="C0")
        self.canvas.ax_phase.set_ylabel("phase (rad)", color="C1")
        self.canvas.ax_mag.set_yscale("log")
        self.canvas.ax_mag.legend(loc="upper left", fontsize="small")
        self.canvas.ax_phase.legend(loc="upper right", fontsize="small")

        self.ax_lambda = self.canvas.ax_mag.secondary_xaxis(
            "top", functions=(thz_to_nm, nm_to_thz)
        )
        self.ax_lambda.set_xlabel("λ (nm)")
        self.ax_lambda.set_xticks([1000, 1500, 2000, 3000, 5000, 10000])

        # --- Controls ---
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)

        gb_bg = self._make_gaussian_group(
            "Background", which="background", spec=BACKGROUND_SPEC, include_center=False
        )
        gb_pk = self._make_gaussian_group(
            "Spike", which="peak", spec=SPIKE_SPEC, include_center=True
        )
        phase_init_box = self._make_button_selector(
            key="phase_init",
            title="Phase Init",
            labels=PHASE_INIT_LABELS,
            default_index=PHASE_INIT_DEFAULT_INDEX,
            on_change=self._on_phase_init_changed,
        )
        phase_end_box = self._make_phase_end_box()

        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_params)

        export_btn = QPushButton("Export PNG…")
        export_btn.clicked.connect(self.export_png)

        btn_layout.addWidget(reset_btn)
        btn_layout.addWidget(export_btn)

        controls_layout.addWidget(gb_bg)
        controls_layout.addWidget(gb_pk)
        controls_layout.addWidget(phase_init_box)
        controls_layout.addWidget(phase_end_box)
        controls_layout.addWidget(btn_row)
        controls_layout.addStretch(1)

        # --- Timers ---
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self.redraw)

        self._recon_timer = QTimer(self)
        self._recon_timer.setSingleShot(True)
        self._recon_timer.timeout.connect(self._reconstruct_now)

        self._recon_in_progress = False
        self._recon_pending = False

        # --- Layout ---
        central = QWidget()
        layout = QHBoxLayout(central)

        left = QVBoxLayout()
        left.addWidget(self.toolbar)
        left.addWidget(self.canvas, 1)

        layout.addLayout(left, 1)
        layout.addWidget(controls)

        self.setStyleSheet(
            """
            QPushButton:checked {
                border: 2px solid #4aa4ff;
                font-weight: bold;
                background-color: #2a2a2a;
            }
            """
        )

        self.setCentralWidget(central)
        self.resize(1800, 650)

        self.redraw()
        self.schedule_reconstruct()

    # -----------------------------
    # Debounced redraw / reconstruction
    # -----------------------------
    def schedule_redraw(self):
        self._update_timer.start(25)

    def schedule_reconstruct(self):
        self._recon_timer.start(250)

    def _invalidate_reconstruction(self, redraw: bool = True):
        self._clear_recon_cache()
        if redraw:
            self.schedule_redraw()
        self.schedule_reconstruct()

    def _reconstruct_now(self):
        if self._recon_in_progress:
            self._recon_pending = True
            return

        self._recon_in_progress = True
        try:
            prof = self.model.compute_profile()
            if self.center_prof is not None:
                self.center_prof.apply(prof)

            formfactor = prof.to_form_factor(transform=DCPhysicalRFFT())
            reconstruction = self._build_reconstruction()

            prof_rec, ff_rec = reconstruction.run(
                formfactor.mag,
                formfactor.grid,
                charge_C=prof.charge,
            )

            self._I_input = prof.current
            self._I_recon = prof_rec.current
            self._mag_pos = formfactor.mag
            self._phase_in = formfactor.phase
            self._phase_recon = ff_rec.phase
            self._phase_recon_last = ff_rec.phase

        finally:
            self._recon_in_progress = False

        self.redraw()

        if self._recon_pending:
            self._recon_pending = False
            self._recon_timer.start(250)

    def _clear_recon_cache(self):
        self._I_input = None
        self._I_recon = None
        self._mag_pos = None
        self._phase_recon = None
        self._phase_in = None

    def _build_reconstruction(self) -> GSMagnitudeOnly:
        return GSMagnitudeOnly(
            transform=self._recon_transform,
            phase_init=self._make_phase_init(),
            time_constraints=self._recon_time_constraints,
            frequency_constraints=self._make_frequency_constraints(),
            stop=self._recon_stop_condition,
        )

    def _make_phase_init(self):
        mode = self._ui_state.phase_init_mode

        if mode == "zero":
            return ZeroPhase()
        if mode == "random":
            return RandomPhase()
        if mode == "minphase":
            return MinimumPhaseCepstrum()
        if mode == "last":
            phase = (
                self._phase_recon_last
                if self._phase_recon_last is not None
                else np.zeros_like(self.model.grid.f_pos)
            )
            return PredefinedPhase(phase=phase)

        return MinimumPhaseCepstrum()

    def _make_frequency_constraints(self):
        constraints = ClampMagnitude(eps=1e-6) + EnforceDCOne()

        if self._ui_state.linear_phase_end:
            alpha = self._ui_state.linear_phase_end_phase_at_300_thz / PHASE_END_REF_FREQ_HZ
            constraints = constraints + ReplacePhaseEndLinear(
                grid=self.model.grid,
                start_freq=self._ui_state.linear_phase_end_start_freq_hz,
                alpha=alpha,
            )

        return constraints

    # -----------------------------
    # UI helpers
    # -----------------------------
    def _row(self, slider: QSlider, spin: QDoubleSpinBox) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider, 1)
        h.addWidget(spin)
        return w

    def _make_slider_spin(self, min_v: float, max_v: float, step: float, init: float):
        n_ticks = int(round((max_v - min_v) / step))
        n_ticks = max(n_ticks, 1)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, n_ticks)
        slider.setProperty("_min_v", float(min_v))
        slider.setProperty("_step", float(step))

        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setSingleStep(step)

        if step < 1:
            dec = int(np.ceil(-np.log10(step)))
            spin.setDecimals(min(max(dec, 2), 10))
        else:
            spin.setDecimals(0)

        def tick_to_value(tick: int) -> float:
            return min_v + tick * step

        def value_to_tick(val: float) -> int:
            return int(round((val - min_v) / step))

        def slider_to_spin(tick: int):
            with QSignalBlocker(spin):
                spin.setValue(tick_to_value(tick))

        def spin_to_slider(val: float):
            tick = value_to_tick(val)
            tick = max(0, min(n_ticks, tick))
            with QSignalBlocker(slider):
                slider.setValue(tick)

        slider.valueChanged.connect(slider_to_spin)
        spin.valueChanged.connect(spin_to_slider)

        init = float(np.clip(init, min_v, max_v))
        with QSignalBlocker(slider), QSignalBlocker(spin):
            spin.setValue(init)
            slider.setValue(value_to_tick(init))

        return slider, spin

    def _register_control(
        self, which: str, field: str, slider: QSlider, spin: QDoubleSpinBox
    ):
        self._controls[(which, field)] = (slider, spin)

    def _set_slider_spin_value(
        self,
        slider: QSlider,
        spin: QDoubleSpinBox,
        value: float,
    ):
        min_v = float(slider.property("_min_v"))
        step = float(slider.property("_step"))
        tick = int(round((value - min_v) / step))
        tick = max(slider.minimum(), min(slider.maximum(), tick))

        with QSignalBlocker(slider), QSignalBlocker(spin):
            spin.setValue(value)
            slider.setValue(tick)

    def _make_button_selector(
        self,
        key: str,
        title: str,
        labels: list[str],
        default_index: int,
        on_change,
    ) -> QGroupBox:
        gb = QGroupBox(title)
        layout = QHBoxLayout(gb)

        group = QButtonGroup(self)
        group.setExclusive(True)

        for i, label in enumerate(labels):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(i == default_index)
            layout.addWidget(btn)
            group.addButton(btn, i)

        group.idClicked.connect(on_change)
        self._selector_groups[key] = group
        return gb

    def _make_phase_end_box(self) -> QGroupBox:
        gb = QGroupBox("Linear Phase End")
        outer = QVBoxLayout(gb)

        self._phase_end_btn = QPushButton("Not applied")
        self._phase_end_btn.setCheckable(True)
        self._phase_end_btn.setChecked(False)
        self._phase_end_btn.toggled.connect(self._on_phase_end_toggled)
        outer.addWidget(self._phase_end_btn)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setContentsMargins(0, 0, 0, 0)

        f_max_thz = 300.0 
        start_init_thz = 200 

        self._phase_end_start_slider, self._phase_end_start_spin = self._make_slider_spin(
            0.0, f_max_thz, 0.1, start_init_thz
        )
        form.addRow(
            QLabel("start freq (THz)"),
            self._row(self._phase_end_start_slider, self._phase_end_start_spin),
        )

        self._phase_end_phase_slider, self._phase_end_phase_spin = self._make_slider_spin(
            -300.0, 300.0, 1.0, self._ui_state.linear_phase_end_phase_at_300_thz
        )
        form.addRow(
            QLabel("phase @ 300 THz"),
            self._row(self._phase_end_phase_slider, self._phase_end_phase_spin),
        )

        outer.addWidget(form_widget)
        self._phase_end_controls_widget = form_widget
        self._phase_end_controls_widget.setEnabled(True)

        self._phase_end_start_slider.valueChanged.connect(self._on_phase_end_params_changed)
        self._phase_end_start_spin.valueChanged.connect(self._on_phase_end_params_changed)
        self._phase_end_phase_slider.valueChanged.connect(self._on_phase_end_params_changed)
        self._phase_end_phase_spin.valueChanged.connect(self._on_phase_end_params_changed)

        return gb

    def _get_selector_state(self) -> dict[str, int]:
        return {key: group.checkedId() for key, group in self._selector_groups.items()}

    def _on_phase_init_changed(self, _id: int):
        idx = self._get_selector_state()["phase_init"]
        self._ui_state.phase_init_mode = PHASE_INIT_LABELS[idx]
        self._invalidate_reconstruction(redraw=False)

    def _on_phase_end_toggled(self, checked: bool):
        self._phase_end_btn.setText("Applied" if checked else "Not applied")
        self._ui_state.linear_phase_end = checked
        self._invalidate_reconstruction(redraw=False)

    def _on_phase_end_params_changed(self, _=None):
        self._ui_state.linear_phase_end_start_freq_hz = self._phase_end_start_spin.value() * 1e12
        self._ui_state.linear_phase_end_phase_at_300_thz = self._phase_end_phase_spin.value()
        self._invalidate_reconstruction(redraw=False)

    def _make_gaussian_group(
        self, title: str, which: str, spec: dict, include_center: bool
    ) -> QGroupBox:
        gb = QGroupBox(title)
        form = QFormLayout(gb)

        controls: dict[str, tuple[QSlider, QDoubleSpinBox]] = {}

        if include_center:
            mn, mx, st, dv = spec["center_fs"]
            c_slider, c_spin = self._make_slider_spin(mn, mx, st, dv)
            form.addRow(QLabel(f"center ({TIME_UNIT})"), self._row(c_slider, c_spin))
            controls["center_fs"] = (c_slider, c_spin)
            self._register_control(which, "center_fs", c_slider, c_spin)

        mn, mx, st, dv = spec["width_fs"]
        w_slider, w_spin = self._make_slider_spin(mn, mx, st, dv)
        form.addRow(QLabel(f"width_scale ({TIME_UNIT})"), self._row(w_slider, w_spin))
        controls["width_fs"] = (w_slider, w_spin)
        self._register_control(which, "width_fs", w_slider, w_spin)

        mn, mx, st, dv = spec["skew"]
        s_slider, s_spin = self._make_slider_spin(mn, mx, st, dv)
        form.addRow(QLabel("skewness"), self._row(s_slider, s_spin))
        controls["skew"] = (s_slider, s_spin)
        self._register_control(which, "skew", s_slider, s_spin)

        mn, mx, st, dv = spec["order"]
        o_slider, o_spin = self._make_slider_spin(mn, mx, st, dv)
        form.addRow(QLabel("shape_order"), self._row(o_slider, o_spin))
        controls["order"] = (o_slider, o_spin)
        self._register_control(which, "order", o_slider, o_spin)

        mn, mx, st, dv = spec["amplitude"]
        a_slider, a_spin = self._make_slider_spin(mn, mx, st, dv)
        form.addRow(QLabel("amplitude"), self._row(a_slider, a_spin))
        controls["amplitude"] = (a_slider, a_spin)
        self._register_control(which, "amplitude", a_slider, a_spin)

        def on_change(_=None):
            target = getattr(self.model.state, which)

            if which == "background":
                target.center = 0.0
            elif include_center:
                target.center = controls["center_fs"][1].value() * T_TO_S

            target.width = controls["width_fs"][1].value() * T_TO_S
            target.skew = controls["skew"][1].value()
            target.order = controls["order"][1].value()
            target.amplitude = controls["amplitude"][1].value()

            self._invalidate_reconstruction(redraw=True)

        for sl, sp in controls.values():
            sl.valueChanged.connect(on_change)
            sp.valueChanged.connect(on_change)

        return gb

    # -----------------------------
    # Plotting
    # -----------------------------
    def redraw(self):
        use_cache = (
            self._I_input is not None
            and self._I_recon is not None
            and self._mag_pos is not None
            and self._phase_in is not None
            and self._phase_recon is not None
        )

        if use_cache:
            I_in = self._I_input
            I_rec = self._I_recon
            mag = self._mag_pos
            phase_in = self._phase_in
            phase_rec = self._phase_recon
        else:
            prof = self.model.compute_profile()
            I_in = prof.current
            I_rec = None
            ff = prof.to_form_factor(transform=DCPhysicalRFFT())
            mag, phase_in = ff.mag, ff.phase
            phase_rec = None

        t_ui = self.model.grid.t * S_TO_T
        self.line_current.set_data(t_ui, I_in * 1e-3)

        if I_rec is not None:
            self.line_recon.set_visible(True)
            self.line_recon.set_data(t_ui, I_rec * 1e-3)
        else:
            self.line_recon.set_visible(False)

        self.canvas.ax_time.set_xlim(t_ui[0], t_ui[-1])
        self.canvas.ax_time.relim()
        self.canvas.ax_time.autoscale_view(scalex=False, scaley=True)

        f_thz = self.model.grid.f_pos * 1e-12
        mag_plot = np.maximum(mag, 1e-30)

        self.line_mag.set_data(f_thz, mag_plot)
        self.line_phase_in.set_data(f_thz, phase_in)

        if phase_rec is not None:
            self.line_phase_recon.set_visible(True)
            self.line_phase_recon.set_data(f_thz, phase_rec)
        else:
            self.line_phase_recon.set_visible(False)

        self.canvas.ax_mag.set_xlim(0.0, 333.0)

        x0, x1 = self.canvas.ax_mag.get_xlim()
        mask = (f_thz >= x0) & (f_thz <= x1)

        if np.any(mask):
            mag_vis = mag_plot[mask]
            mag_vis = mag_vis[np.isfinite(mag_vis) & (mag_vis > 0)]
            if mag_vis.size:
                y0 = mag_vis.min()
                y1 = mag_vis.max()
                if y0 == y1:
                    y0 *= 0.8
                    y1 *= 1.2
                else:
                    y0 /= 1.2
                    y1 *= 1.2
                self.canvas.ax_mag.set_ylim(y0, y1)

            phase_parts = [phase_in[mask]]
            if phase_rec is not None:
                phase_parts.append(phase_rec[mask])

            phase_vis = np.concatenate(
                [p[np.isfinite(p)] for p in phase_parts if np.any(np.isfinite(p))]
            )
            if phase_vis.size:
                y0 = phase_vis.min()
                y1 = phase_vis.max()
                pad = 0.05 * (y1 - y0) if y1 > y0 else 0.5
                self.canvas.ax_phase.set_ylim(y0 - pad, y1 + pad)

        self.canvas.draw_idle()

    # -----------------------------
    # Actions
    # -----------------------------
    def reset_params(self):
        self.model = ProfileModel(
            ProfileModelState(
                dt=DT,
                t_max=T_MAX,
                charge=CHARGE_C,
            )
        )

        for which in ("background", "peak"):
            p = getattr(self.model.state, which)

            defaults = {
                "width_fs": p.width * S_TO_T,
                "skew": p.skew,
                "order": p.order,
                "amplitude": p.amplitude,
            }
            if which != "background":
                defaults["center_fs"] = p.center * S_TO_T

            for field, val in defaults.items():
                pair = self._controls.get((which, field))
                if pair is None:
                    continue
                slider, spin = pair
                self._set_slider_spin_value(slider, spin, val)

        # reset reconstruction UI state
        self._ui_state.phase_init_mode = PHASE_INIT_LABELS[PHASE_INIT_DEFAULT_INDEX]
        self._ui_state.linear_phase_end = False
        self._ui_state.linear_phase_end_start_freq_hz = 0.5 * self.model.grid.f_pos[-1]
        self._ui_state.linear_phase_end_phase_at_300_thz = 0.0

        phase_group = self._selector_groups.get("phase_init")
        if phase_group is not None and phase_group.button(PHASE_INIT_DEFAULT_INDEX):
            with QSignalBlocker(phase_group):
                phase_group.button(PHASE_INIT_DEFAULT_INDEX).setChecked(True)

        with QSignalBlocker(self._phase_end_btn):
            self._phase_end_btn.setChecked(False)
            self._phase_end_btn.setText("Not applied")

        self._set_slider_spin_value(
            self._phase_end_start_slider,
            self._phase_end_start_spin,
            self._ui_state.linear_phase_end_start_freq_hz * 1e-12,
        )
        self._set_slider_spin_value(
            self._phase_end_phase_slider,
            self._phase_end_phase_spin,
            self._ui_state.linear_phase_end_phase_at_300_thz,
        )

        self._phase_recon_last = None
        self._invalidate_reconstruction(redraw=True)

    def export_png(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PNG", "profile.png", "PNG Images (*.png)"
        )
        if path:
            self.canvas.figure.savefig(path, dpi=150)


def main() -> int:
    app = QApplication(sys.argv)
    app.setWindowIcon(load_app_icon())

    set_dark_theme(app)
    apply_mpl_style()
    sync_mpl_to_qt(app)

    w = MainWindow()
    w.setWindowIcon(load_app_icon())
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())