import numpy as np
from PySide6.QtWidgets import QApplication

from phase_weaver.app.config import PLOT_LINE_MODE
from phase_weaver.app.logic import LoadedMeasurement, ReferenceCurrentProfile
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


def _view_ranges(panel):
    return {
        "time": panel.canvas.time_plot.getViewBox().viewRange(),
        "spectrum": panel.canvas.spectrum_plot.getViewBox().viewRange(),
        "phase": panel.canvas.phase_view.viewRange(),
    }


def _assert_view_ranges_equal(actual, expected):
    for key in expected:
        for actual_axis, expected_axis in zip(actual[key], expected[key]):
            np.testing.assert_allclose(actual_axis, expected_axis)


def _set_manual_axis_ranges(panel):
    panel.canvas.time_plot.getViewBox().setRange(
        xRange=(-1.1e-13, 1.3e-13),
        yRange=(2.0, 1800.0),
        padding=0.0,
        disableAutoRange=True,
    )
    panel.canvas.spectrum_plot.getViewBox().setRange(
        xRange=(15e12, 95e12),
        yRange=(-9.0, -0.05),
        padding=0.0,
        disableAutoRange=True,
    )
    panel.canvas.phase_view.setRange(
        yRange=(-2.2, 2.4),
        padding=0.0,
        disableAutoRange=True,
    )


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
        reference_current=ReferenceCurrentProfile(
            label="CRISP SA1",
            time_s=np.array([-1e-15, 0.0, 1e-15]),
            current_a=np.array([0.0, 10.0, 0.0]),
            inferred_max_frequency_thz=250.0,
        ),
    )

    panel.render_input(profile, formfactor)
    panel.render_reconstruction(profile, formfactor)
    panel.render_measurements((measurement,))
    panel.clear_reconstruction()

    assert panel.time_model is not None
    assert panel.spectrum_model is not None
    assert len(panel.measurement_lines) == 1
    measurement_x, measurement_y = panel.measurement_lines[0].getOriginalDataset()
    np.testing.assert_allclose(measurement_x, [10e12, 20e12])
    np.testing.assert_allclose(measurement_y, [0.9, 0.7])
    reference_x, reference_y = panel.line_reference_current.getOriginalDataset()
    np.testing.assert_allclose(reference_x, [-1e-15, 0.0, 1e-15])
    np.testing.assert_allclose(reference_y, [0.0, 10.0, 0.0])
    assert "CRISP SA1" in panel.canvas.time_legend.labels


def test_plot_panel_uses_si_units_as_plot_data():
    _app()
    panel = PlotPanel()
    profile, formfactor = _plot_inputs()

    panel.render_input(profile, formfactor)

    time_x, current_y = panel.line_current.getOriginalDataset()
    mag_x, _mag_y = panel.line_mag.getOriginalDataset()
    phase_x, _phase_y = panel.line_phase_in.getData()

    np.testing.assert_allclose(time_x, panel.time_model.t_plot_s)
    np.testing.assert_allclose(current_y, panel.time_model.current_input_plot_A)
    np.testing.assert_allclose(mag_x, panel.spectrum_model.f_plot_Hz)
    np.testing.assert_allclose(phase_x, panel.spectrum_model.f_plot_Hz)


def test_spectrum_magnitude_uses_pyqtgraph_log_axis_and_phase_is_linear():
    _app()
    panel = PlotPanel()
    profile, formfactor = _plot_inputs()

    panel.render_input(profile, formfactor)

    mag_x, mag_y = panel.line_mag.getOriginalDataset()
    _display_mag_x, display_mag_y = panel.line_mag.getData()
    phase_x, phase_y = panel.line_phase_in.getData()

    assert np.asarray(mag_x).shape == np.asarray(phase_x).shape
    np.testing.assert_allclose(mag_x, panel.spectrum_model.f_plot_Hz)
    assert panel.canvas.spectrum_plot.getPlotItem().ctrl.logYCheck.isChecked()
    expected_mag = np.clip(panel.spectrum_model.mag_input_ui, 1e-12, None)
    np.testing.assert_allclose(
        mag_y,
        expected_mag,
    )
    np.testing.assert_allclose(display_mag_y, np.log10(expected_mag))
    np.testing.assert_allclose(phase_y, panel.spectrum_model.phase_input_ui)


def test_plot_panel_line_visibility_follows_controls():
    _app()
    panel = PlotPanel()
    profile, formfactor = _plot_inputs()
    panel.render_input(profile, formfactor)
    panel.render_reconstruction(profile, formfactor)
    panel.render_measurements(
        (
            LoadedMeasurement(
                label="measurement",
                measured=MeasuredFormFactor(freq=np.array([10e12]), mag=np.array([0.9])),
                reference_current=ReferenceCurrentProfile(
                    label="CRISP SA1",
                    time_s=np.array([0.0]),
                    current_a=np.array([1.0]),
                    inferred_max_frequency_thz=250.0,
                ),
            ),
        )
    )

    panel.plot_controls.line_actions[PLOT_LINE_MODE.CURRENT_INPUT].setChecked(False)
    panel.plot_controls.line_actions[PLOT_LINE_MODE.CURRENT_REFERENCE].setChecked(False)
    panel.plot_controls.line_actions[PLOT_LINE_MODE.PHASE_RECON].setChecked(False)

    assert not panel.line_current.isVisible()
    assert not panel.line_reference_current.isVisible()
    assert not panel.line_phase_recon.isVisible()

    panel.plot_controls.show_all_lines()

    assert panel.line_current.isVisible()
    assert panel.line_reference_current.isVisible()
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


def test_plot_panel_fwhm_is_hidden_by_default_and_toggles_on():
    _app()
    panel = PlotPanel()
    profile, formfactor = _plot_inputs()
    panel.render_input(profile, formfactor)
    panel.render_reconstruction(profile, formfactor)

    assert not panel.plot_controls.show_fwhm
    assert not panel.line_fwhm_input.isVisible()
    assert not panel.line_fwhm_recon.isVisible()
    assert not panel.line_fwhm_input_caps.isVisible()
    assert not panel.line_fwhm_recon_caps.isVisible()
    assert not panel.text_fwhm_input.isVisible()
    assert not panel.text_fwhm_recon.isVisible()
    assert "FWHM" not in " ".join(panel.canvas.time_legend.labels)

    panel.plot_controls.fwhm_action.setChecked(True)

    assert panel.line_fwhm_input.isVisible()
    assert panel.line_fwhm_recon.isVisible()
    assert panel.line_fwhm_input_caps.isVisible()
    assert panel.line_fwhm_recon_caps.isVisible()
    assert panel.text_fwhm_input.isVisible()
    assert panel.text_fwhm_recon.isVisible()
    assert "FWHM" not in " ".join(panel.canvas.time_legend.labels)


def test_plot_panel_theme_switch_keeps_items():
    _app()
    panel = PlotPanel(theme=APP_THEME.DARK)
    profile, formfactor = _plot_inputs()
    panel.render_input(profile, formfactor)

    panel.set_theme(APP_THEME.LIGHT)
    panel.set_theme(APP_THEME.DARK)

    assert panel.line_current is not None
    assert panel.line_mag is not None


def test_plot_panel_preserves_manual_axis_ranges_when_input_changes():
    _app()
    panel = PlotPanel()
    profile, formfactor = _plot_inputs()
    changed_profile = ProfileModel(
        ProfileModelState(charge=profile.charge * 1.5)
    ).compute_profile()
    changed_formfactor = changed_profile.to_form_factor()

    panel.render_input(profile, formfactor)
    _set_manual_axis_ranges(panel)
    expected = _view_ranges(panel)

    panel.render_input(changed_profile, changed_formfactor)

    _assert_view_ranges_equal(_view_ranges(panel), expected)


def test_plot_panel_preserves_manual_axis_ranges_when_reconstruction_changes():
    _app()
    panel = PlotPanel()
    profile, formfactor = _plot_inputs()
    panel.render_input(profile, formfactor)
    _set_manual_axis_ranges(panel)
    expected = _view_ranges(panel)

    panel.render_reconstruction(profile, formfactor)
    _assert_view_ranges_equal(_view_ranges(panel), expected)

    panel.clear_reconstruction()
    _assert_view_ranges_equal(_view_ranges(panel), expected)


def test_plot_panel_preserves_manual_axis_ranges_when_plot_controls_change():
    _app()
    panel = PlotPanel()
    profile, formfactor = _plot_inputs()
    panel.render_input(profile, formfactor)
    panel.render_reconstruction(profile, formfactor)
    _set_manual_axis_ranges(panel)
    expected = _view_ranges(panel)

    panel.plot_controls.line_actions[PLOT_LINE_MODE.CURRENT_INPUT].setChecked(False)

    _assert_view_ranges_equal(_view_ranges(panel), expected)
