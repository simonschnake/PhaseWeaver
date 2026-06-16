from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib as mpl
from PySide6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

import numpy as np

from phase_weaver.app.plot_model import (
    SpectrumPlotModel,
    TimePlotModel,
)
from phase_weaver.app.utils import nm_to_thz, thz_to_nm
from phase_weaver.qt_theme import APP_THEME

from phase_weaver.core import CurrentProfile, FormFactor, Profile

from .plot_controls_box import PlotControlsBox


from phase_weaver.app.config import PLOT_LINE_MODE
from phase_weaver.app.logic import LoadedMeasurement


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        # Keep the embedded GUI figure at a screen-friendly DPI even when the
        # active Matplotlib style sheet prefers notebook-sized figures.
        fig = Figure(constrained_layout=True, dpi=100)
        self.ax_time = fig.add_subplot(1, 2, 1)
        self.ax_mag = fig.add_subplot(1, 2, 2)
        self.ax_phase = self.ax_mag.twinx()
        super().__init__(fig)
        self.setParent(parent)


class PlotPanel(QWidget):
    def __init__(self, parent=None, theme: APP_THEME = APP_THEME.DARK):
        super().__init__(parent)
        self.theme = theme

        self.canvas = MplCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.plot_controls = PlotControlsBox()
        self.plot_controls.changed.connect(self._render_from_controls)

        plot_col = QVBoxLayout()
        plot_col.addWidget(self.toolbar)
        plot_col.addWidget(self.canvas, 1)

        layout = QHBoxLayout(self)
        layout.addWidget(self.plot_controls)
        layout.addLayout(plot_col, 1)

        self.time_model: TimePlotModel | None = None
        self.spectrum_model: SpectrumPlotModel | None = None
        self.measurement_lines = []

        self._create_artists()
        self._style_axes()
        self.set_theme(theme)

        self.canvas.ax_mag.set_xlim(0.0, 333.0)  # up to 333 THz (900 nm) by default

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
        self._apply_time_axes()
        self._apply_spectrum_axes()
        self._render_fwhm()
        self._apply_line_visibility()
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
        self._apply_line_visibility()
        self.canvas.draw_idle()

    def clear_reconstruction(self) -> None:
        if self.time_model is not None:
            self.time_model.profile_recon = None
        if self.spectrum_model is not None:
            self.spectrum_model.formfactor_recon = None
        if self.time_model is not None and self.spectrum_model is not None:
            self._render_time()
            self._render_spectrum()
            self._apply_time_axes()
            self._apply_spectrum_axes()
            self._render_fwhm()
            self._apply_line_visibility()
            self.canvas.draw_idle()

    def render_measurements(
        self, measurements: tuple[LoadedMeasurement, ...] = ()
    ) -> None:
        for line in self.measurement_lines:
            line.remove()
        self.measurement_lines = []

        for item in measurements:
            (line,) = self.canvas.ax_mag.plot(
                item.measured.freq * 1e-12,
                item.measured.mag,
                linestyle="None",
                marker="o",
                markersize=3,
                alpha=0.75,
                label=item.label,
            )
            self.measurement_lines.append(line)

        self.canvas.ax_mag.legend(loc="upper left", fontsize="small")
        self._apply_theme_to_plot()
        self.canvas.draw_idle()

    def set_theme(self, theme: APP_THEME) -> None:
        self.theme = theme
        self._apply_theme_to_plot()
        self.refresh_canvas()

    def refresh_canvas(self) -> None:
        self.canvas.draw()
        self.canvas.updateGeometry()
        self.updateGeometry()

    def _render_time(self) -> None:
        if self.time_model is None:
            raise ValueError(
                "Time model is not initialized. Call render_input() first."
            )
        input_y = self.time_model.current_input_ui
        recon_y = self.time_model.current_recon_ui
        self.canvas.ax_time.set_ylabel("Current (kA)")

        self.line_current.set_data(self.time_model.t_ui, input_y)

        if recon_y is not None:
            self.line_recon.set_visible(True)
            self.line_recon.set_data(self.time_model.t_ui, recon_y)
        else:
            self.line_recon.set_visible(False)
            self.line_recon.set_data([], [])

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

        visible_lines = self.plot_controls.visible_lines

        show_input_times = PLOT_LINE_MODE.CURRENT_INPUT in visible_lines
        show_recon_times = PLOT_LINE_MODE.CURRENT_RECON in visible_lines

        if ci_fwhm is not None:
            self.line_fwhm_input.set_data(
                [ci_fwhm.left_half_max, ci_fwhm.right_half_max],
                [ci_fwhm.half_max_value, ci_fwhm.half_max_value],
            )
            self.line_fwhm_input.set_visible(show_input_times)
            self.line_fwhm_input_caps.set_data(
                [ci_fwhm.left_half_max, ci_fwhm.right_half_max],
                [ci_fwhm.half_max_value, ci_fwhm.half_max_value],
            )
            self.line_fwhm_input_caps.set_visible(show_input_times)
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
            self.line_fwhm_recon.set_visible(show_recon_times)
            self.line_fwhm_recon_caps.set_data(
                [cr_fwhm.left_half_max, cr_fwhm.right_half_max],
                [cr_fwhm.half_max_value, cr_fwhm.half_max_value],
            )
            self.line_fwhm_recon_caps.set_visible(show_recon_times)
            self.text_fwhm_recon.set_text(f"Reconstructed FWHM: {cr_fwhm.fwhm:.2f} fs")
        else:
            self.line_fwhm_recon.set_visible(False)
            self.line_fwhm_recon_caps.set_visible(False)
            self.text_fwhm_recon.set_text("Reconstructed FWHM: n/a")

    def _render_from_controls(self) -> None:
        if self.time_model is not None:
            self._render_time()
            self._apply_time_axes()
            self._render_fwhm()

        if self.spectrum_model is not None:
            self._render_spectrum()
            self._apply_spectrum_axes()

        self._apply_line_visibility()
        self.canvas.draw_idle()

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

        x0_value = model.f_min_ui
        x1_value = model.f_max_ui
        if x0_value is None or x1_value is None:
            x0, x1 = 0.0, 333.0
        else:
            x0 = float(x0_value)
            x1 = float(x1_value)

        if not np.isfinite(x0) or not np.isfinite(x1):
            x0, x1 = 0.0, 333.0
        elif x0 == x1:
            pad = max(abs(x0) * 0.05, 1.0)
            x0 -= pad
            x1 += pad

        self.canvas.ax_mag.set_xlim(x0, x1)
        self.canvas.ax_phase.set_xlim(x0, x1)

        self.canvas.ax_mag.relim()
        self.canvas.ax_mag.autoscale_view(scalex=False, scaley=True)

        self.canvas.ax_phase.relim()
        self.canvas.ax_phase.autoscale_view(scalex=False, scaley=True)

    def _apply_line_visibility(self) -> None:
        visible = self.plot_controls.visible_lines

        self.line_current.set_visible(PLOT_LINE_MODE.CURRENT_INPUT in visible)

        self.line_recon.set_visible(
            PLOT_LINE_MODE.CURRENT_RECON in visible
            and self.time_model is not None
            and self.time_model.current_recon_ui is not None
        )

        self.line_mag.set_visible(PLOT_LINE_MODE.MAG_INPUT in visible)

        self.line_mag_recon.set_visible(
            PLOT_LINE_MODE.MAG_RECON in visible
            and self.spectrum_model is not None
            and self.spectrum_model.mag_recon_ui is not None
        )

        self.line_phase_in.set_visible(PLOT_LINE_MODE.PHASE_INPUT in visible)

        self.line_phase_recon.set_visible(
            PLOT_LINE_MODE.PHASE_RECON in visible
            and self.spectrum_model is not None
            and self.spectrum_model.phase_recon_ui is not None
        )

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

    def _apply_theme_to_plot(self) -> None:
        colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if len(colors) < 4:
            colors = ["C0", "C1", "C2", "C3"]

        facecolor = mpl.rcParams["axes.facecolor"]
        figure_facecolor = mpl.rcParams["figure.facecolor"]
        text_color = mpl.rcParams["text.color"]
        label_color = mpl.rcParams["axes.labelcolor"]
        edge_color = mpl.rcParams["axes.edgecolor"]
        grid_color = mpl.rcParams["grid.color"]

        self.canvas.figure.set_facecolor(figure_facecolor)
        self.canvas.setStyleSheet(f"background: {figure_facecolor};")

        for ax in self.canvas.figure.axes:
            ax.set_facecolor(facecolor)
            ax.title.set_color(text_color)
            ax.xaxis.label.set_color(label_color)
            ax.yaxis.label.set_color(label_color)
            ax.tick_params(axis="both", colors=mpl.rcParams["xtick.color"])
            ax.grid(
                mpl.rcParams["axes.grid"],
                color=grid_color,
                linestyle=mpl.rcParams["grid.linestyle"],
                alpha=mpl.rcParams["grid.alpha"],
            )
            for spine in ax.spines.values():
                spine.set_color(edge_color)
            self._style_legend(ax.get_legend())

        self.ax_lambda.xaxis.label.set_color(label_color)
        self.ax_lambda.tick_params(axis="x", colors=mpl.rcParams["xtick.color"])
        for spine in self.ax_lambda.spines.values():
            spine.set_color(edge_color)

        self.line_current.set_color(colors[0])
        self.line_recon.set_color(colors[1])
        self.line_mag.set_color(colors[0])
        self.line_mag_recon.set_color(colors[2])
        self.line_phase_in.set_color(colors[1])
        self.line_phase_recon.set_color(colors[3])

        self.line_fwhm_input.set_color(colors[0])
        self.line_fwhm_input_caps.set_color(colors[0])
        self.text_fwhm_input.set_color(colors[0])
        self.line_fwhm_recon.set_color(colors[1])
        self.line_fwhm_recon_caps.set_color(colors[1])
        self.text_fwhm_recon.set_color(colors[1])

        self.canvas.ax_mag.yaxis.label.set_color(colors[0])
        self.canvas.ax_phase.yaxis.label.set_color(colors[1])
        self.canvas.ax_mag.tick_params(axis="y", colors=colors[0])
        self.canvas.ax_phase.tick_params(axis="y", colors=colors[1])

        self._style_fwhm_box(self.text_fwhm_input)
        self._style_fwhm_box(self.text_fwhm_recon)

        for index, line in enumerate(self.measurement_lines):
            line.set_color(colors[(index + 4) % len(colors)])

    def _style_legend(self, legend) -> None:
        if legend is None:
            return

        legend.get_frame().set_facecolor(mpl.rcParams["legend.facecolor"])
        legend.get_frame().set_edgecolor(mpl.rcParams["legend.edgecolor"])
        legend.get_frame().set_alpha(mpl.rcParams["legend.framealpha"])
        for text in legend.get_texts():
            text.set_color(mpl.rcParams["text.color"])

    def _style_fwhm_box(self, text) -> None:
        if self.theme == APP_THEME.LIGHT:
            facecolor = (1.0, 1.0, 1.0, 0.72)
        else:
            facecolor = (0.0, 0.0, 0.0, 0.18)
        text.set_bbox(dict(boxstyle="round,pad=0.2", fc=facecolor, ec="none"))
