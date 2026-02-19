import sys
import numpy as np

from PySide6.QtCore import Qt, QTimer, QSignalBlocker
from PySide6.QtWidgets import (
    QApplication,
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

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from phase_weaver.qt_theme import set_dark_theme
from phase_weaver.mpl_style import apply_mpl_style, sync_mpl_to_qt
from phase_weaver.model.profile_model import ProfileModel, ProfileModelState
from phase_weaver.core.time_constraints import normalize_center_nonneg_density
from phase_weaver.core.spectrum import mag_phase_from_time, normalize_mag_dc_to_one
from phase_weaver.core.reconstruction import reconstruct_current_from_mag


# -----------------------------
# Constants / UI units
# -----------------------------
CHARGE_C = 250e-12  # 250 pC

TIME_UNIT = "fs"
S_TO_T = 1e15  # seconds -> fs
T_TO_S = 1e-15  # fs -> seconds

# Slider specs: (min, max, step, default) in DISPLAY units (fs for time fields)
SPIKE_SPEC = {
    "center_fs": (-20.0, 20.0, 1.0, -10.0),
    "width_fs": (0.1, 10.0, 0.005, 0.721),
    "skew": (-1.0, 1.0, 0.1, 0.0),
    "order": (0.01, 4.0, 0.01, 0.5),
    "amplitude": (0.0, 1e5, 100.0, 13000.0),
}

BACKGROUND_SPEC = {
    # background has no center control (forced to 0)
    "width_fs": (1.0, 100.0, 0.5, 21.91914604893585),
    "skew": (-1.0, 1.0, 0.1, 0.0),
    "order": (1e-8, 10.0, 0.10, 1.0),
    "amplitude": (0.0, 1e5, 100.0, 5819.0),
}

# Wavelength axis mapping: f(THz) <-> λ(nm), λ = c / f
C_NM_THz = 299792.458  # speed of light in (nm·THz)
F_EPS_THz = 1e-6
L_EPS_NM = 1e-6


def thz_to_nm(f_thz):
    f = np.asarray(f_thz, dtype=float)
    f = np.maximum(f, F_EPS_THz)
    return C_NM_THz / f


def nm_to_thz(l_nm):
    wavelength = np.asarray(l_nm, dtype=float)
    wavelength = np.maximum(wavelength, L_EPS_NM)
    return C_NM_THz / wavelength


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

        # --- Model ---
        self.model = ProfileModel(ProfileModelState())

        # --- internal control registry for reset ---
        # maps (which, field) -> (slider, spin)
        self._controls: dict[tuple[str, str], tuple[QSlider, QDoubleSpinBox]] = {}

        # --- reconstruction cache (cleared on any parameter change) ---
        self._p_norm = None  # density (unit integral)
        self._I_input = None  # charge-scaled input current (A)
        self._I_recon = None  # charge-scaled reconstructed current (A)
        self._mag_pos = None  # normalized mag (mag[0]=1)
        self._phase_pos = None  # reconstructed phase used for plotting (rad)
        self._phase_in = None  # input phase (unwrapped) used for plotting (rad)

        # --- Plot ---
        self.canvas = MplCanvas()
        self.canvas.setStyleSheet("background: #1e1e1e;")

        # Matplotlib toolbar (Option 1: full zoom/pan/etc)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Left: time-domain current
        (self.line_current,) = self.canvas.ax_time.plot([], [], label="input", linewidth=2.0)
        (self.line_recon,) = self.canvas.ax_time.plot([], [], label="reconstructed", linestyle="--", alpha=0.9)
        self.canvas.ax_time.legend(loc="upper right", fontsize="small")
        self.canvas.ax_time.set_xlabel(f"t ({TIME_UNIT})")
        self.canvas.ax_time.set_ylabel("Current (kA)")

        # Right: magnitude + phases
        (self.line_mag,) = self.canvas.ax_mag.plot([], [], label="|F|", color="C0")
        (self.line_phase_in,) = self.canvas.ax_phase.plot([], [], label="phase (input)", color="C1")
        (self.line_phase_recon,) = self.canvas.ax_phase.plot(
            [], [], label="phase (recon)", color="C2", linestyle="--", alpha=0.9
        )

        self.canvas.ax_mag.set_xlabel("f (THz)")
        self.canvas.ax_mag.set_ylabel("|F(f)|", color="C0")
        self.canvas.ax_phase.set_ylabel("phase (rad)", color="C1")
        self.canvas.ax_mag.set_yscale("log")

        self.canvas.ax_mag.legend(loc="upper left", fontsize="small")
        self.canvas.ax_phase.legend(loc="upper right", fontsize="small")

        # Top x-axis: wavelength (nm)
        self.ax_lambda = self.canvas.ax_mag.secondary_xaxis("top", functions=(thz_to_nm, nm_to_thz))
        self.ax_lambda.set_xlabel("λ (nm)")
        self.ax_lambda.set_xticks([1000, 1500, 2000, 3000, 5000, 10000])

        # --- Controls ---
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)

        gb_bg = self._make_gaussian_group(
            "Background", which="background", spec=BACKGROUND_SPEC, include_center=False
        )
        gb_pk = self._make_gaussian_group("Spike", which="peak", spec=SPIKE_SPEC, include_center=True)

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
        controls_layout.addWidget(btn_row)
        controls_layout.addStretch(1)

        # --- Debounce updates ---
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self.redraw)

        # --- Debounce reconstruction ---
        self._recon_timer = QTimer(self)
        self._recon_timer.setSingleShot(True)
        self._recon_timer.timeout.connect(self._reconstruct_now)

        self._recon_in_progress = False
        self._recon_pending = False

        # --- Layout ---
        central = QWidget()
        layout = QHBoxLayout(central)

        left = QVBoxLayout()
        left.addWidget(self.toolbar)  # <-- IMPORTANT: show the Matplotlib buttons
        left.addWidget(self.canvas, 1)

        layout.addLayout(left, 1)
        layout.addWidget(controls)

        self.setCentralWidget(central)
        self.resize(1800, 650)

        # Initial draw + initial reconstruction
        self.redraw()
        self.schedule_reconstruct()

    # -----------------------------
    # Debounced redraw / reconstruction
    # -----------------------------
    def schedule_redraw(self):
        self._update_timer.start(25)

    def schedule_reconstruct(self):
        self._recon_timer.start(250)

    def _reconstruct_now(self):
        # If reconstruction takes long and user changes params again, queue one more run.
        if self._recon_in_progress:
            self._recon_pending = True
            return

        self._recon_in_progress = True
        try:
            g = self.model.grid()
            _, y = self.model.compute_profile()

            # 1) normalize/center/nonneg -> density p(t)
            p = normalize_center_nonneg_density(y, g)

            # 2) scale to current with fixed charge
            I_in = CHARGE_C * p

            # 3) mag + phase (input), normalize mag[0]=1
            mag, phase_in, _ = mag_phase_from_time(p, g)
            mag = normalize_mag_dc_to_one(mag)

            # 4) reconstruct
            _, I_rec, info = reconstruct_current_from_mag(
                g,
                mag,
                charge_C=CHARGE_C,
                n_iter=60,
                init_phase="minphase",
                enforce_dc_one=True,
                enforce_nonneg=True,
                normalize_area_to_one=True,
                center=True,
            )

            # store cache
            self._p_norm = p
            self._I_input = I_in
            self._I_recon = I_rec
            self._mag_pos = mag
            self._phase_in = phase_in
            self._phase_pos = info["phase_pos"]

        finally:
            self._recon_in_progress = False

        self.redraw()

        if self._recon_pending:
            self._recon_pending = False
            self._recon_timer.start(250)

    def _clear_recon_cache(self):
        self._p_norm = None
        self._I_input = None
        self._I_recon = None
        self._mag_pos = None
        self._phase_pos = None
        self._phase_in = None

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
        # slider uses integer ticks: value = min_v + tick * step
        n_ticks = int(round((max_v - min_v) / step))
        n_ticks = max(n_ticks, 1)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, n_ticks)

        # store mapping metadata for reset()
        slider.setProperty("_min_v", float(min_v))
        slider.setProperty("_step", float(step))

        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setSingleStep(step)

        # decimals based on step (cap to avoid silly long floats)
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
            val = tick_to_value(tick)
            with QSignalBlocker(spin):
                spin.setValue(val)

        def spin_to_slider(val: float):
            tick = value_to_tick(val)
            tick = max(0, min(n_ticks, tick))
            with QSignalBlocker(slider):
                slider.setValue(tick)

        slider.valueChanged.connect(slider_to_spin)
        spin.valueChanged.connect(spin_to_slider)

        # set initial values WITHOUT feedback fights
        init = float(np.clip(init, min_v, max_v))
        with QSignalBlocker(slider), QSignalBlocker(spin):
            spin.setValue(init)
            slider.setValue(value_to_tick(init))

        return slider, spin

    def _register_control(self, which: str, field: str, slider: QSlider, spin: QDoubleSpinBox):
        self._controls[(which, field)] = (slider, spin)

    def _make_gaussian_group(self, title: str, which: str, spec: dict, include_center: bool) -> QGroupBox:
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

            # enforce "background center = 0" always
            if which == "background":
                target.center = 0.0
            else:
                if include_center:
                    target.center = controls["center_fs"][1].value() * T_TO_S  # fs -> s

            target.width = controls["width_fs"][1].value() * T_TO_S  # fs -> s
            target.skew = controls["skew"][1].value()
            target.order = controls["order"][1].value()
            target.amplitude = controls["amplitude"][1].value()

            self._clear_recon_cache()
            self.schedule_redraw()
            self.schedule_reconstruct()

        for (sl, sp) in controls.values():
            sl.valueChanged.connect(on_change)
            sp.valueChanged.connect(on_change)

        return gb

    # -----------------------------
    # Plotting
    # -----------------------------
    def redraw(self):
        g = self.model.grid()

        use_cache = (
            self._I_input is not None
            and self._mag_pos is not None
            and self._phase_in is not None
            and self._phase_pos is not None
        )

        if use_cache:
            t = g.t
            I_in = self._I_input
            I_rec = self._I_recon
            mag = self._mag_pos
            phase_in = self._phase_in
            phase_rec = self._phase_pos
        else:
            t, y = self.model.compute_profile()
            p = normalize_center_nonneg_density(y, g)
            I_in = CHARGE_C * p
            I_rec = None
            mag, phase_in, _ = mag_phase_from_time(p, g)
            mag = normalize_mag_dc_to_one(mag)
            phase_rec = None

        # --- left plot (kA) ---
        t_ui = t * S_TO_T  # seconds -> fs for UI display
        self.line_current.set_data(t_ui, I_in * 1e-3)

        if I_rec is not None:
            self.line_recon.set_visible(True)
            self.line_recon.set_data(t_ui, I_rec * 1e-3)
        else:
            self.line_recon.set_visible(False)

        self.canvas.ax_time.set_xlim(t_ui[0], t_ui[-1])
        self.canvas.ax_time.relim()
        self.canvas.ax_time.autoscale_view(scalex=False, scaley=True)

        # --- right plot ---
        f_thz = g.f_pos * 1e-12
        mag_plot = np.maximum(mag, 1e-30)

        self.line_mag.set_data(f_thz, mag_plot)
        self.line_phase_in.set_data(f_thz, phase_in)

        if phase_rec is not None:
            self.line_phase_recon.set_visible(True)
            self.line_phase_recon.set_data(f_thz, phase_rec)
        else:
            self.line_phase_recon.set_visible(False)

        # limit x-axis to 0..333 THz
        self.canvas.ax_mag.set_xlim(0.0, 333.0)

        self.canvas.ax_mag.relim()
        self.canvas.ax_mag.autoscale_view(scalex=False, scaley=True)

        self.canvas.ax_phase.relim()
        self.canvas.ax_phase.autoscale_view(scalex=False, scaley=True)

        self.canvas.draw_idle()

    # -----------------------------
    # Actions
    # -----------------------------
    def reset_params(self):
        self.model = ProfileModel(ProfileModelState())
        self._clear_recon_cache()

        for which in ("background", "peak"):
            p = getattr(self.model.state, which)

            defaults = {
                "width_fs": p.width * S_TO_T,  # s -> fs
                "skew": p.skew,
                "order": p.order,
                "amplitude": p.amplitude,
            }
            if which != "background":
                defaults["center_fs"] = p.center * S_TO_T  # s -> fs

            for field, val in defaults.items():
                pair = self._controls.get((which, field))
                if pair is None:
                    continue

                slider, spin = pair
                min_v = float(slider.property("_min_v"))
                step = float(slider.property("_step"))
                tick = int(round((val - min_v) / step))
                tick = max(slider.minimum(), min(slider.maximum(), tick))

                with QSignalBlocker(slider), QSignalBlocker(spin):
                    spin.setValue(val)
                    slider.setValue(tick)

        self.redraw()
        self.schedule_reconstruct()

    def export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "profile.png", "PNG Images (*.png)")
        if path:
            self.canvas.figure.savefig(path, dpi=150)


def main() -> int:
    app = QApplication(sys.argv)

    set_dark_theme(app)
    apply_mpl_style()  # must be before Figure creation
    sync_mpl_to_qt(app)  # override dpi + font sizes

    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())