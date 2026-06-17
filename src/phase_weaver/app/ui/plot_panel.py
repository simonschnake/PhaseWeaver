from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget
import pyqtgraph as pg

from phase_weaver.app.config import PLOT_LINE_MODE
from phase_weaver.app.logic import LoadedMeasurement
from phase_weaver.app.plot_model import SpectrumPlotModel, TimePlotModel
from phase_weaver.app.plot_theme import COLORBLIND_FRIENDLY_CYCLE, PLOT_THEMES
from phase_weaver.app.utils import hz_to_m
from phase_weaver.core import CurrentProfile, FormFactor, Profile
from phase_weaver.qt_theme import APP_THEME

from .plot_controls_box import PlotControlsBox


MIN_LOG_MAG = 1e-12
LINE_WIDTH = 5
FWHM_WIDTH = 6
CAP_WIDTH = 4

pg.setConfigOptions(antialias=True)


class LegendStrip(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.labels: list[str] = []
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 4)
        self._layout.setSpacing(12)
        self._layout.addStretch(1)

    def set_items(
        self,
        items: list[tuple[str, str, Qt.PenStyle, bool]],
        background: str,
        text_color: str,
    ) -> None:
        self.labels = [label for label, *_ in items]
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.setStyleSheet(f"background: {background};")
        for label, color, style, marker in items:
            self._layout.addWidget(
                self._make_item(label, color, style, marker, background, text_color)
            )
        self._layout.addStretch(1)

    def _make_item(
        self,
        label: str,
        color: str,
        style: Qt.PenStyle,
        marker: bool,
        background: str,
        text_color: str,
    ) -> QWidget:
        item = QWidget(self)
        layout = QHBoxLayout(item)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        swatch = QLabel("●" if marker else "", item)
        swatch.setFixedSize(34, 14)
        if marker:
            swatch.setAlignment(Qt.AlignmentFlag.AlignCenter)
            swatch.setStyleSheet(
                f"color: {color}; background: {background}; font-size: 13px;"
            )
        else:
            border_style = "dashed" if style == Qt.PenStyle.DashLine else "solid"
            swatch.setStyleSheet(
                "QLabel {"
                f"background: {background};"
                f"border-bottom: {LINE_WIDTH}px {border_style} {color};"
                "}"
            )

        text = QLabel(label, item)
        text.setStyleSheet(f"color: {text_color}; background: {background};")
        layout.addWidget(swatch)
        layout.addWidget(text)
        return item


class LambdaAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):  # noqa: N802
        labels = []
        for value in values:
            wavelength = hz_to_m(float(value))
            if np.isfinite(wavelength) and wavelength > 0:
                labels.append(pg.siFormat(wavelength, suffix="m", precision=3))
            else:
                labels.append("")
        return labels


class PlotCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.time_plot = pg.PlotWidget()
        self.spectrum_plot = pg.PlotWidget(axisItems={"top": LambdaAxis("top")})
        self.phase_view = pg.ViewBox()
        self.time_legend = LegendStrip()
        self.spectrum_legend = LegendStrip()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        time_col = QVBoxLayout()
        time_col.setContentsMargins(0, 0, 0, 0)
        time_col.setSpacing(0)
        time_col.addWidget(self.time_legend)
        time_col.addWidget(self.time_plot, 1)

        spectrum_col = QVBoxLayout()
        spectrum_col.setContentsMargins(0, 0, 0, 0)
        spectrum_col.setSpacing(0)
        spectrum_col.addWidget(self.spectrum_legend)
        spectrum_col.addWidget(self.spectrum_plot, 1)

        layout.addLayout(time_col, 1)
        layout.addLayout(spectrum_col, 1)

        self._configure_axes()

    def _configure_axes(self) -> None:
        self.time_plot.setLabel("bottom", "t", units="s")
        self.time_plot.setLabel("left", "Current", units="A")
        self.time_plot.showGrid(x=False, y=False)

        plot_item = self.spectrum_plot.getPlotItem()
        plot_item.setLabel("bottom", "f", units="Hz")
        plot_item.setLabel("left", "|F(f)|")
        plot_item.setLabel("right", "phase", units="rad")
        plot_item.setLabel("top", "lambda")
        plot_item.showAxis("top")
        plot_item.showAxis("right")
        plot_item.showGrid(x=False, y=False)
        plot_item.setLogMode(y=True)

        plot_item.scene().addItem(self.phase_view)
        plot_item.getAxis("right").linkToView(self.phase_view)
        self.phase_view.setXLink(plot_item.vb)
        plot_item.vb.sigResized.connect(self._sync_phase_view)
        self._sync_phase_view()

    def _sync_phase_view(self) -> None:
        plot_item = self.spectrum_plot.getPlotItem()
        self.phase_view.setGeometry(plot_item.vb.sceneBoundingRect())
        self.phase_view.linkedViewChanged(plot_item.vb, self.phase_view.XAxis)


class PlotPanel(QWidget):
    def __init__(self, parent=None, theme: APP_THEME = APP_THEME.DARK):
        super().__init__(parent)
        self.theme = theme

        self.canvas = PlotCanvas(self)
        self.plot_controls = PlotControlsBox(self)
        self.plot_controls.changed.connect(self._render_from_controls)

        layout = QHBoxLayout(self)
        layout.addWidget(self.canvas, 1)

        self.time_model: TimePlotModel | None = None
        self.spectrum_model: SpectrumPlotModel | None = None
        self.measurement_lines: list[pg.PlotDataItem] = []
        self.measurement_labels: list[str] = []

        self._create_artists()
        self._style_axes()
        self.set_theme(theme)
        self.canvas.spectrum_plot.setXRange(0.0, 333e12, padding=0.0)

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
        self.refresh_canvas()

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
        self.refresh_canvas()

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
            self.refresh_canvas()

    def render_measurements(
        self, measurements: tuple[LoadedMeasurement, ...] = ()
    ) -> None:
        for line in self.measurement_lines:
            self.canvas.spectrum_plot.removeItem(line)
        self.measurement_lines = []
        self.measurement_labels = []

        colors = self._colors()
        for index, item in enumerate(measurements):
            color = colors[(index + 4) % len(colors)]
            line = self.canvas.spectrum_plot.plot(
                item.measured.freq,
                self._magnitude_for_log_axis(item.measured.mag),
                pen=None,
                symbol="o",
                symbolSize=4,
                symbolPen=pg.mkPen(color, width=2),
                symbolBrush=pg.mkBrush(color),
            )
            self.measurement_lines.append(line)
            self.measurement_labels.append(item.label)

        self._refresh_legend_strips()
        self.refresh_canvas()

    def set_theme(self, theme: APP_THEME) -> None:
        self.theme = theme
        self._apply_theme_to_plot()
        self.refresh_canvas()

    def refresh_canvas(self) -> None:
        self.canvas.updateGeometry()
        self.updateGeometry()

    def _render_time(self) -> None:
        if self.time_model is None:
            raise ValueError(
                "Time model is not initialized. Call render_input() first."
            )

        self.line_current.setData(
            self.time_model.t_plot_s, self.time_model.current_input_plot_A
        )
        recon_y = self.time_model.current_recon_plot_A
        if recon_y is not None:
            self.line_recon.setData(self.time_model.t_plot_s, recon_y)
        else:
            self.line_recon.setData([], [])

    def _render_spectrum(self) -> None:
        if self.spectrum_model is None:
            raise ValueError(
                "Spectrum model is not initialized. Call render_input() first."
            )

        f_ui = self.spectrum_model.f_plot_Hz
        self.line_mag.setData(
            f_ui, self._magnitude_for_log_axis(self.spectrum_model.mag_input_ui)
        )

        mag_recon = self.spectrum_model.mag_recon_ui
        if mag_recon is not None:
            self.line_mag_recon.setData(
                f_ui, self._magnitude_for_log_axis(mag_recon)
            )
        else:
            self.line_mag_recon.setData([], [])

        self.line_phase_in.setData(f_ui, self.spectrum_model.phase_input_ui)

        phase_recon = self.spectrum_model.phase_recon_ui
        if phase_recon is not None:
            self.line_phase_recon.setData(f_ui, phase_recon)
        else:
            self.line_phase_recon.setData([], [])

    def _render_fwhm(self) -> None:
        if self.time_model is None:
            raise ValueError(
                "Time model is not initialized. Call render_input() first."
            )

        visible_lines = self.plot_controls.visible_lines
        show_fwhm = self.plot_controls.show_fwhm
        show_input_times = PLOT_LINE_MODE.CURRENT_INPUT in visible_lines
        show_recon_times = PLOT_LINE_MODE.CURRENT_RECON in visible_lines

        self._set_fwhm_items(
            self.time_model.current_input_fwhm_plot,
            self.line_fwhm_input,
            self.line_fwhm_input_caps,
            self.text_fwhm_input,
            "Input FWHM",
            show_fwhm and show_input_times,
        )
        self._set_fwhm_items(
            self.time_model.current_recon_fwhm_plot,
            self.line_fwhm_recon,
            self.line_fwhm_recon_caps,
            self.text_fwhm_recon,
            "Reconstructed FWHM",
            show_fwhm and show_recon_times,
        )
        if show_fwhm:
            self._position_fwhm_labels()

    def _render_from_controls(self) -> None:
        if self.time_model is not None:
            self._render_time()
            self._apply_time_axes()
            self._render_fwhm()

        if self.spectrum_model is not None:
            self._render_spectrum()
            self._apply_spectrum_axes()

        self._apply_line_visibility()
        self.refresh_canvas()

    def _apply_time_axes(self) -> None:
        model = self.time_model
        if model is None:
            raise ValueError(
                "Time model is not initialized. Call render_input() first."
            )
        self.canvas.time_plot.setXRange(
            model.t_min_plot_s, model.t_max_plot_s, padding=0.0
        )
        self.canvas.time_plot.enableAutoRange(axis="y")

    def _apply_spectrum_axes(self) -> None:
        model = self.spectrum_model
        if model is None:
            raise ValueError(
                "Spectrum model is not initialized. Call render_input() first."
            )

        x0, x1 = self._spectrum_x_range(model)
        self.canvas.spectrum_plot.setXRange(x0, x1, padding=0.0)
        self.canvas.spectrum_plot.enableAutoRange(axis="y")
        self.canvas.phase_view.enableAutoRange(axis="y")
        self.canvas._sync_phase_view()

    def _apply_line_visibility(self) -> None:
        visible = self.plot_controls.visible_lines

        self.line_current.setVisible(PLOT_LINE_MODE.CURRENT_INPUT in visible)
        self.line_recon.setVisible(
            PLOT_LINE_MODE.CURRENT_RECON in visible
            and self.time_model is not None
            and self.time_model.current_recon_plot_A is not None
        )
        self.line_mag.setVisible(PLOT_LINE_MODE.MAG_INPUT in visible)
        self.line_mag_recon.setVisible(
            PLOT_LINE_MODE.MAG_RECON in visible
            and self.spectrum_model is not None
            and self.spectrum_model.mag_recon_ui is not None
        )
        self.line_phase_in.setVisible(PLOT_LINE_MODE.PHASE_INPUT in visible)
        self.line_phase_recon.setVisible(
            PLOT_LINE_MODE.PHASE_RECON in visible
            and self.spectrum_model is not None
            and self.spectrum_model.phase_recon_ui is not None
        )
        self._refresh_legend_strips()

    def _create_artists(self):
        self.line_current = self.canvas.time_plot.plot([], [])
        self.line_recon = self.canvas.time_plot.plot([], [])

        self.line_fwhm_input = self.canvas.time_plot.plot([], [])
        self.line_fwhm_recon = self.canvas.time_plot.plot([], [])
        self.line_fwhm_input_caps = self.canvas.time_plot.plot([], [])
        self.line_fwhm_recon_caps = self.canvas.time_plot.plot([], [])

        self.text_fwhm_input = pg.TextItem("", anchor=(1, 0))
        self.text_fwhm_recon = pg.TextItem("", anchor=(1, 0))
        self.canvas.time_plot.addItem(self.text_fwhm_input)
        self.canvas.time_plot.addItem(self.text_fwhm_recon)

        self.line_mag = self.canvas.spectrum_plot.plot([], [])
        self.line_mag_recon = self.canvas.spectrum_plot.plot([], [])
        self.line_phase_in = pg.PlotDataItem([], [])
        self.line_phase_recon = pg.PlotDataItem([], [])
        self.canvas.phase_view.addItem(self.line_phase_in)
        self.canvas.phase_view.addItem(self.line_phase_recon)

    def _style_axes(self):
        pass

    def _apply_theme_to_plot(self) -> None:
        theme = PLOT_THEMES[self.theme]
        colors = self._colors()

        self.canvas.setStyleSheet(f"background-color: {theme.background};")
        for widget in (self.canvas.time_plot, self.canvas.spectrum_plot):
            widget.setBackground(theme.plot_background)
            widget.setStyleSheet(f"background-color: {theme.plot_background};")
            plot_item = widget.getPlotItem()
            plot_item.showGrid(x=False, y=False)
            for axis_name in ("left", "right", "top", "bottom"):
                axis = plot_item.getAxis(axis_name)
                axis.setPen(pg.mkPen(theme.axis))
                axis.setTextPen(pg.mkPen(theme.text))
                axis.setStyle(tickTextOffset=6)

        self.canvas.spectrum_plot.getPlotItem().getAxis("left").setTextPen(
            pg.mkPen(colors[0])
        )
        self.canvas.spectrum_plot.getPlotItem().getAxis("right").setTextPen(
            pg.mkPen(colors[1])
        )

        self.line_current.setPen(pg.mkPen(colors[0], width=LINE_WIDTH))
        self.line_recon.setPen(pg.mkPen(colors[1], width=LINE_WIDTH, style=Qt.DashLine))
        self.line_mag.setPen(pg.mkPen(colors[0], width=LINE_WIDTH))
        self.line_mag_recon.setPen(
            pg.mkPen(colors[1], width=LINE_WIDTH, style=Qt.DashLine)
        )
        self.line_phase_in.setPen(pg.mkPen(colors[2], width=LINE_WIDTH))
        self.line_phase_recon.setPen(
            pg.mkPen(colors[3], width=LINE_WIDTH, style=Qt.DashLine)
        )
        self.line_fwhm_input.setPen(pg.mkPen(colors[0], width=FWHM_WIDTH))
        self.line_fwhm_recon.setPen(pg.mkPen(colors[1], width=FWHM_WIDTH))
        self.line_fwhm_input_caps.setPen(pg.mkPen(colors[0], width=CAP_WIDTH))
        self.line_fwhm_recon_caps.setPen(pg.mkPen(colors[1], width=CAP_WIDTH))

        self.text_fwhm_input.setColor(QColor(colors[0]))
        self.text_fwhm_recon.setColor(QColor(colors[1]))
        fill = QBrush(QColor(*theme.fwhm_fill))
        for text in (self.text_fwhm_input, self.text_fwhm_recon):
            text.fill = fill
            text.border = QPen(Qt.PenStyle.NoPen)

        for index, line in enumerate(self.measurement_lines):
            color = colors[(index + 4) % len(colors)]
            line.setSymbolPen(pg.mkPen(color, width=2))
            line.setSymbolBrush(pg.mkBrush(color))

        self._refresh_legend_strips()

    def _colors(self) -> list[str]:
        return list(COLORBLIND_FRIENDLY_CYCLE)

    def _refresh_legend_strips(self) -> None:
        theme = PLOT_THEMES[self.theme]
        colors = self._colors()
        visible = self.plot_controls.visible_lines

        time_items = []
        if PLOT_LINE_MODE.CURRENT_INPUT in visible:
            time_items.append(("input", colors[0], Qt.PenStyle.SolidLine, False))
        if (
            PLOT_LINE_MODE.CURRENT_RECON in visible
            and self.time_model is not None
            and self.time_model.current_recon_plot_A is not None
        ):
            time_items.append(("reconstructed", colors[1], Qt.PenStyle.DashLine, False))
        self.canvas.time_legend.set_items(
            time_items, theme.legend_background, theme.text
        )

        spectrum_items = []
        if PLOT_LINE_MODE.MAG_INPUT in visible:
            spectrum_items.append(("|F|", colors[0], Qt.PenStyle.SolidLine, False))
        if (
            PLOT_LINE_MODE.MAG_RECON in visible
            and self.spectrum_model is not None
            and self.spectrum_model.mag_recon_ui is not None
        ):
            spectrum_items.append(
                ("|F| (recon)", colors[1], Qt.PenStyle.DashLine, False)
            )
        if PLOT_LINE_MODE.PHASE_INPUT in visible:
            spectrum_items.append(
                ("phase (input)", colors[2], Qt.PenStyle.SolidLine, False)
            )
        if (
            PLOT_LINE_MODE.PHASE_RECON in visible
            and self.spectrum_model is not None
            and self.spectrum_model.phase_recon_ui is not None
        ):
            spectrum_items.append(
                ("phase (recon)", colors[3], Qt.PenStyle.DashLine, False)
            )
        for index, label in enumerate(self.measurement_labels):
            spectrum_items.append(
                (
                    label,
                    colors[(index + 4) % len(colors)],
                    Qt.PenStyle.SolidLine,
                    True,
                )
            )
        self.canvas.spectrum_legend.set_items(
            spectrum_items,
            theme.legend_background,
            theme.text,
        )

    def _set_fwhm_items(
        self,
        fwhm,
        line: pg.PlotDataItem,
        caps: pg.PlotDataItem,
        text: pg.TextItem,
        label: str,
        visible: bool,
    ) -> None:
        if fwhm is None:
            line.setData([], [])
            caps.setData([], [])
            text.setText(f"{label}: n/a")
            line.setVisible(False)
            caps.setVisible(False)
            text.setVisible(visible)
            return

        x0 = float(fwhm.left_half_max)
        x1 = float(fwhm.right_half_max)
        y = float(fwhm.half_max_value)
        cap_height = self._fwhm_cap_height()
        line.setData([x0, x1], [y, y])
        caps.setData(
            [x0, x0, np.nan, x1, x1],
            [y - cap_height, y + cap_height, np.nan, y - cap_height, y + cap_height],
        )
        text.setText(f"{label}: {pg.siFormat(fwhm.fwhm, suffix='s', precision=3)}")
        line.setVisible(visible)
        caps.setVisible(visible)
        text.setVisible(visible)

    def _fwhm_cap_height(self) -> float:
        (_, _), (y0, y1) = self.canvas.time_plot.getViewBox().viewRange()
        if math.isfinite(y0) and math.isfinite(y1) and y0 != y1:
            return abs(y1 - y0) * 0.025
        return 1.0

    def _position_fwhm_labels(self) -> None:
        (x0, x1), (y0, y1) = self.canvas.time_plot.getViewBox().viewRange()
        if not all(math.isfinite(v) for v in (x0, x1, y0, y1)):
            return
        x = x1 - (x1 - x0) * 0.015
        self.text_fwhm_input.setPos(x, y1 - (y1 - y0) * 0.03)
        self.text_fwhm_recon.setPos(x, y1 - (y1 - y0) * 0.11)

    def _spectrum_x_range(self, model: SpectrumPlotModel) -> tuple[float, float]:
        x0_value = model.f_min_plot_Hz
        x1_value = model.f_max_plot_Hz
        if x0_value is None or x1_value is None:
            x0, x1 = 0.0, 333e12
        else:
            x0 = float(x0_value)
            x1 = float(x1_value)

        if not np.isfinite(x0) or not np.isfinite(x1):
            return 0.0, 333e12
        if x0 == x1:
            pad = max(abs(x0) * 0.05, 1.0)
            return x0 - pad, x1 + pad
        return x0, x1

    def _magnitude_for_log_axis(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        clipped = np.clip(values, MIN_LOG_MAG, None)
        return np.where(finite, clipped, np.nan)
