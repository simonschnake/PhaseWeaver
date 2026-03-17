from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from phase_weaver.app.plot_model import (
    PlotModels,
    SpectrumPlotModel,
    TimePlotModel,
)
from phase_weaver.app.utils import nm_to_thz, thz_to_nm

from phase_weaver.core import CurrentProfile, FormFactor, Profile


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(constrained_layout=True)
        self.ax_time = fig.add_subplot(1, 2, 1)
        self.ax_mag = fig.add_subplot(1, 2, 2)
        self.ax_phase = self.ax_mag.twinx()
        super().__init__(fig)
        self.setParent(parent)


class PlotPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.canvas = MplCanvas(self)
        self.canvas.setStyleSheet("background: #1e1e1e;")
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        self._create_artists()
        self._style_axes()

        self.canvas.ax_mag.set_xlim(0.0, 333.0)  # up to 333 THz (900 nm) by default

        self.time_model: TimePlotModel | None = None
        self.spectrum_model: SpectrumPlotModel | None = None

    def render_input(self, prof_input: Profile, formfactor_input: FormFactor):
        if self.time_model is None:
            if type(prof_input) is not CurrentProfile:
                prof_input = CurrentProfile.from_profile(prof_input)
            self.time_model = TimePlotModel(profile_input=prof_input)
        else:
            self.time_model.update(profile_input=prof_input)

        if self.spectrum_model is None:
            self.spectrum_model = SpectrumPlotModel(formfactor_input=formfactor_input)
        else:
            self.spectrum_model.update(formfactor_input=formfactor_input)

        self._render_time()
        self._render_spectrum()
        self.canvas.draw_idle()

    def render_reconstruction(
        self, prof_recon: Profile | None, formfactor_recon: FormFactor | None
    ):
        if self.time_model is None:
            raise ValueError(
                "Time model is not initialized. Call render_input() first."
            )
        if self.spectrum_model is None:
            raise ValueError(
                "Spectrum model is not initialized. Call render_input() first."
            )

        self.time_model.update(profile_recon=prof_recon)
        self.spectrum_model.update(formfactor_recon=formfactor_recon)
        self._render_time()
        self._render_spectrum()
        self._apply_time_axes()
        self._apply_spectrum_axes()
        self._render_fwhm()
        self.canvas.draw_idle()

    def _render_time(self) -> None:
        if self.time_model is None:
            raise ValueError(
                "Time model is not initialized. Call render_input() first."
            )
        self.line_current.set_data(
            self.time_model.t_ui, self.time_model.current_input_ui
        )
        if self.time_model.current_recon_ui is not None:
            self.line_recon.set_visible(True)
            self.line_recon.set_data(
                self.time_model.t_ui, self.time_model.current_recon_ui
            )
        else:
            self.line_recon.set_visible(False)
            self.line_recon.set_data([], [])

    def _render_spectrum(self) -> None:
        if self.spectrum_model is None:
            raise ValueError(
                "Spectrum model is not initialized. Call render_input() first."
            )
        self.line_mag.set_data(
            self.spectrum_model.f_ui, self.spectrum_model.mag_input_ui
        )
        if self.spectrum_model.mag_recon_ui is not None:
            self.line_mag_recon.set_visible(True)
            self.line_mag_recon.set_data(
                self.spectrum_model.f_ui, self.spectrum_model.mag_recon_ui
            )
        else:
            self.line_mag_recon.set_visible(False)
            self.line_mag_recon.set_data([], [])
        self.line_phase_in.set_data(
            self.spectrum_model.f_ui, self.spectrum_model.phase_input_ui
        )
        if self.spectrum_model.phase_recon_ui is not None:
            self.line_phase_recon.set_visible(True)
            self.line_phase_recon.set_data(
                self.spectrum_model.f_ui, self.spectrum_model.phase_recon_ui
            )
        else:
            self.line_phase_recon.set_visible(False)
            self.line_phase_recon.set_data([], [])

    def _render_fwhm(self) -> None:
        if self.time_model is None:
            raise ValueError(
                "Time model is not initialized. Call render_input() first."
            )
        ci_fwhm = self.time_model.current_input_fwhm
        cr_fwhm = self.time_model.current_recon_fwhm

        if ci_fwhm is not None:
            self.line_fwhm_input.set_data(
                [ci_fwhm.left_half_max, ci_fwhm.right_half_max],
                [ci_fwhm.half_max_value, ci_fwhm.half_max_value],
            )
            self.line_fwhm_input.set_visible(True)
            self.line_fwhm_input_caps.set_data(
                [ci_fwhm.left_half_max, ci_fwhm.right_half_max],
                [ci_fwhm.half_max_value, ci_fwhm.half_max_value],
            )
            self.line_fwhm_input_caps.set_visible(True)
            self.text_fwhm_input.set_text(f"Input FWHM: {ci_fwhm.fwhm:.2f} fs")
        else:
            self.line_fwhm_input.set_visible(False)
            self.line_fwhm_input_caps.set_visible(False)
            self.text_fwhm_input.set_text("Input FWHM: n/a")

        if cr_fwhm is not None:
            self.line_fwhm_recon.set_data(
                [cr_fwhm.left_half_max, cr_fwhm.right_half_max],
                [cr_fwhm.half_max_value, cr_fwhm.half_max_value],
            )
            self.line_fwhm_recon.set_visible(True)
            self.line_fwhm_recon_caps.set_data(
                [cr_fwhm.left_half_max, cr_fwhm.right_half_max],
                [cr_fwhm.half_max_value, cr_fwhm.half_max_value],
            )
            self.line_fwhm_recon_caps.set_visible(True)
            self.text_fwhm_recon.set_text(f"Reconstructed FWHM: {cr_fwhm.fwhm:.2f} fs")
        else:
            self.line_fwhm_recon.set_visible(False)
            self.line_fwhm_recon_caps.set_visible(False)
            self.text_fwhm_recon.set_text("Reconstructed FWHM: n/a")

    def _apply_time_axes(self) -> None:
        model = self.time_model
        if model is None:
            raise ValueError(
                "Time model is not initialized. Call render_input() first."
            )
        self.canvas.ax_time.set_xlim(model.t_min_ui, model.t_max_ui)
        self.canvas.ax_time.relim()
        self.canvas.ax_time.autoscale_view(scalex=False, scaley=True)

    def _apply_spectrum_axes(self) -> None:
        model = self.spectrum_model
        if model is None:
            raise ValueError(
                "Spectrum model is not initialized. Call render_input() first."
            )
        self.canvas.ax_mag.set_xlim(model.f_min_ui, model.f_max_ui)
        self.canvas.ax_mag.relim()
        self.canvas.ax_phase.set_xlim(model.f_min_ui, model.f_max_ui)
        self.canvas.ax_phase.relim()
        self.canvas.ax_mag.autoscale_view(scalex=False, scaley=True)
        self.canvas.ax_phase.autoscale_view(scalex=False, scaley=True)

    def _create_artists(self):
        (self.line_current,) = self.canvas.ax_time.plot(
            [], [], label="input", linewidth=2.0
        )
        (self.line_recon,) = self.canvas.ax_time.plot(
            [], [], label="reconstructed", linestyle="--", alpha=0.9
        )

        (self.line_fwhm_input,) = self.canvas.ax_time.plot(
            [],
            [],
            linewidth=3.0,
            color=self.line_current.get_color(),
            alpha=0.95,
            solid_capstyle="butt",
            zorder=5,
        )
        (self.line_fwhm_recon,) = self.canvas.ax_time.plot(
            [],
            [],
            linewidth=3.0,
            color=self.line_recon.get_color(),
            alpha=0.95,
            solid_capstyle="butt",
            zorder=5,
        )

        (self.line_fwhm_input_caps,) = self.canvas.ax_time.plot(
            [],
            [],
            linestyle="None",
            marker="|",
            markersize=12,
            color=self.line_current.get_color(),
            zorder=6,
        )
        (self.line_fwhm_recon_caps,) = self.canvas.ax_time.plot(
            [],
            [],
            linestyle="None",
            marker="|",
            markersize=12,
            color=self.line_recon.get_color(),
            zorder=6,
        )

        self.text_fwhm_input = self.canvas.ax_time.text(
            0.985,
            0.98,
            "",
            transform=self.canvas.ax_time.transAxes,
            ha="right",
            va="top",
            fontsize="small",
            color=self.line_current.get_color(),
            bbox=dict(boxstyle="round,pad=0.2", fc=(0, 0, 0, 0.18), ec="none"),
        )
        self.text_fwhm_recon = self.canvas.ax_time.text(
            0.985,
            0.90,
            "",
            transform=self.canvas.ax_time.transAxes,
            ha="right",
            va="top",
            fontsize="small",
            color=self.line_recon.get_color(),
            bbox=dict(boxstyle="round,pad=0.2", fc=(0, 0, 0, 0.18), ec="none"),
        )

        (self.line_mag,) = self.canvas.ax_mag.plot([], [], label="|F|", color="C0")
        (self.line_mag_recon,) = self.canvas.ax_mag.plot(
            [], [], label="|F| (recon)", color="C3", linestyle="--", alpha=0.9
        )
        (self.line_phase_in,) = self.canvas.ax_phase.plot(
            [], [], label="phase (input)", color="C1"
        )
        (self.line_phase_recon,) = self.canvas.ax_phase.plot(
            [], [], label="phase (recon)", color="C2", linestyle="--", alpha=0.9
        )

    def _style_axes(self):
        self.canvas.ax_time.legend(loc="upper left", fontsize="small")
        self.canvas.ax_time.set_xlabel("t (fs)")
        self.canvas.ax_time.set_ylabel("Current (kA)")

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
        self.ax_lambda.set_xticks([1000, 1500, 2000, 3000, 10000])
