import numpy as np
from PySide6.QtWidgets import QApplication

from phase_weaver.app.config import PLOT_LINE_MODE
from phase_weaver.app.logic import LoadedMeasurement
from phase_weaver.app.state import ProfileModel, ProfileModelState
from phase_weaver.app.ui.plot_panel import PlotPanel
from phase_weaver.core.measurement import MeasuredFormFactor
from phase_weaver.qt_theme import APP_THEME


def _app():
    return QApplication.instance() or QApplication([])


def _plot_inputs():
    profile = ProfileModel(ProfileModelState()).compute_profile()
    formfactor = profile.to_form_factor()
    return profile, formfactor


def test_plot_panel_initializes_without_matplotlib():
    app = _app()

    panel = PlotPanel()

    assert panel.canvas.time_plot is not None
    assert panel.canvas.spectrum_plot is not None
    assert app is not None


def test_plot_panel_render_methods_run():
    _app()
    panel = PlotPanel()
    profile, formfactor = _plot_inputs()
    measurement = LoadedMeasurement(
        label="measurement",
        measured=MeasuredFormFactor(
            freq=np.array([10e12, 20e12]),
            mag=np.array([0.9, 0.7]),
        ),
    )

    panel.render_input(profile, formfactor)
    panel.render_reconstruction(profile, formfactor)
    panel.render_measurements((measurement,))
    panel.clear_reconstruction()

    assert panel.time_model is not None
    assert panel.spectrum_model is not None
    assert len(panel.measurement_lines) == 1


def test_spectrum_magnitude_is_log_transformed_but_phase_is_linear():
    _app()
    panel = PlotPanel()
    profile, formfactor = _plot_inputs()

    panel.render_input(profile, formfactor)

    mag_x, mag_y = panel.line_mag.getData()
    phase_x, phase_y = panel.line_phase_in.getData()

    assert np.asarray(mag_x).shape == np.asarray(phase_x).shape
    np.testing.assert_allclose(
        mag_y,
        np.log10(np.clip(panel.spectrum_model.mag_input_ui, 1e-12, None)),
    )
    np.testing.assert_allclose(phase_y, panel.spectrum_model.phase_input_ui)


def test_plot_panel_line_visibility_follows_controls():
    _app()
    panel = PlotPanel()
    profile, formfactor = _plot_inputs()
    panel.render_input(profile, formfactor)
    panel.render_reconstruction(profile, formfactor)

    panel.plot_controls.line_actions[PLOT_LINE_MODE.CURRENT_INPUT].setChecked(False)
    panel.plot_controls.line_actions[PLOT_LINE_MODE.PHASE_RECON].setChecked(False)

    assert not panel.line_current.isVisible()
    assert not panel.line_phase_recon.isVisible()

    panel.plot_controls.show_all_lines()

    assert panel.line_current.isVisible()
    assert panel.line_phase_recon.isVisible()


def test_plot_panel_legend_follows_line_visibility():
    _app()
    panel = PlotPanel()
    profile, formfactor = _plot_inputs()
    panel.render_input(profile, formfactor)
    panel.render_reconstruction(profile, formfactor)

    assert "input" in panel.canvas.time_legend.labels
    assert "phase (recon)" in panel.canvas.spectrum_legend.labels

    panel.plot_controls.line_actions[PLOT_LINE_MODE.CURRENT_INPUT].setChecked(False)
    panel.plot_controls.line_actions[PLOT_LINE_MODE.PHASE_RECON].setChecked(False)

    assert "input" not in panel.canvas.time_legend.labels
    assert "phase (recon)" not in panel.canvas.spectrum_legend.labels

    panel.plot_controls.show_all_lines()

    assert "input" in panel.canvas.time_legend.labels
    assert "phase (recon)" in panel.canvas.spectrum_legend.labels


def test_plot_panel_theme_switch_keeps_items():
    _app()
    panel = PlotPanel(theme=APP_THEME.DARK)
    profile, formfactor = _plot_inputs()
    panel.render_input(profile, formfactor)

    panel.set_theme(APP_THEME.LIGHT)
    panel.set_theme(APP_THEME.DARK)

    assert panel.line_current is not None
    assert panel.line_mag is not None
