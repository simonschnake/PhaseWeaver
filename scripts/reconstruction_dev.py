#!/usr/bin/env python
"""Development Script for the reconstruction"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt

from phase_weaver.app.state import (
    ControlsState,
    ProfileModelState,
    MeasurementState,
    ReconstructionState,
)
from phase_weaver.app.plot_model import TimePlotModel, SpectrumPlotModel
from phase_weaver.core import CurrentProfile
from phase_weaver.model.profiles import AsymSuperGaussParams
from phase_weaver.app import config as CFG


from phase_weaver.app.logic import AppLogic


def main() -> int:
    scenario = ProfileModelState(
        background=AsymSuperGaussParams(
            width=21.92 * CFG.T_TO_S, skew=0.95, order=0.41, amplitude=5819
        ),
        peak=AsymSuperGaussParams(
            width=0.721 * CFG.T_TO_S, skew=0.3, order=0.5, amplitude=26900
        ),
    )

    measurement = MeasurementState(crisp=True, infrared=True)

    reconstruction = ReconstructionState(
        phase_init_mode=CFG.PHASE_INIT_MODE.ZERO,
    )

    control_state = ControlsState(
        scenario=scenario, measurement=measurement, reconstruction=reconstruction
    )

    logic = AppLogic()

    prof_input, ff_input, _measurement = logic.compute_initial(control_state)
    measurements, source = logic.active_measurements(ff_input, control_state.measurement)

    prof_recon, ff_recon, _summary = logic.compute_reconstruction(
        grid=ff_input.grid,
        measurements=measurements,
        controls_state=control_state,
        ff_input=ff_input,
        measurement_source=source,
    )

    prof_recon.charge = prof_input.charge

    # Plot Stuff
    time_model = TimePlotModel(
        profile_input=CurrentProfile.from_profile(prof_input),
        profile_recon=CurrentProfile.from_profile(prof_recon),
    )

    spec_model = SpectrumPlotModel(formfactor_input=ff_input, formfactor_recon=ff_recon)

    app = pg.mkQApp("PhaseWeaver reconstruction dev")
    win = pg.GraphicsLayoutWidget(show=True, title="PhaseWeaver reconstruction dev")
    win.resize(1100, 450)

    time_plot = win.addPlot(row=0, col=0)
    time_plot.setLabel("left", "Current", units="kA")
    time_plot.setLabel("bottom", "t", units="fs")
    time_plot.addLegend()
    time_plot.plot(time_model.t_ui, time_model.current_input_ui, pen="b", name="Input")
    if time_model.current_recon_ui is not None:
        time_plot.plot(
            time_model.t_ui,
            time_model.current_recon_ui,
            pen=pg.mkPen("r", style=Qt.DashLine),
            name="Recon",
        )

    spec_plot = win.addPlot(row=0, col=1)
    spec_plot.setLabel("left", "log10 |F|")
    spec_plot.setLabel("bottom", "f", units="THz")
    spec_plot.addLegend()
    spec_plot.plot(
        spec_model.f_ui,
        np.log10(np.clip(spec_model.mag_input_ui, 1e-12, None)),
        pen="b",
        name="Mag Input",
    )
    if spec_model.mag_recon_ui is not None:
        spec_plot.plot(
            spec_model.f_ui,
            np.log10(np.clip(spec_model.mag_recon_ui, 1e-12, None)),
            pen=pg.mkPen("r", style=Qt.DashLine),
            name="Mag Recon",
        )
    spec_plot.setXRange(spec_model.f_min_ui, spec_model.f_max_ui, padding=0.0)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
