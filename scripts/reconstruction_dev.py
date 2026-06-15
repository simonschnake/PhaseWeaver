#!/usr/bin/env python
"""Development Script for the reconstruction"""

from __future__ import annotations

from matplotlib import pyplot as plt

import numpy as np

from phase_weaver.app.state import (
    ControlsState,
    ProfileModelState,
    MeasurementState,
    ReconstructionState,
)
from phase_weaver.app.plot_model import TimePlotModel, SpectrumPlotModel
from phase_weaver.core import CurrentProfile, FormFactor
from phase_weaver.core.reconstruction import CombineMeasuredMagnitudes
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

    prof_input, ff_input, measurement = logic.compute_initial(control_state)

    prof_recon, ff_recon = logic.compute_reconstruction(
        controls_state=control_state,
        ff_input=ff_input,
        measured_ff=measurement,
    )

    prof_recon.charge = prof_input.charge

    # Plot Stuff
    time_model = TimePlotModel(
        profile_input=CurrentProfile.from_profile(prof_input),
        profile_recon=CurrentProfile.from_profile(prof_recon),
    )

    spec_model = SpectrumPlotModel(formfactor_input=ff_input, formfactor_recon=ff_recon)

    _, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    ax[0].plot(time_model.t_ui, time_model.current_input_ui, label="Input")
    ax[0].plot(time_model.t_ui, time_model.current_recon_ui, "--", label="Init")
    ax[0].set_ylabel("Current (kA)")
    ax[0].set_xlabel("t (fs)")
    ax[0].legend()

    ax[1].plot(spec_model.f_ui, spec_model.mag_input_ui, "-", label="Mag Input")
    ax[1].plot(spec_model.f_ui, spec_model.mag_recon_ui, "--", label="Mag Init")
    ax[1].set_ylabel(r"$|F|^2$ (a.u.)")
    ax[1].set_xlabel(r"$f$ (THz)")
    ax[1].set_yscale("log")
    ax[1].set_xlim(spec_model.f_min_ui, spec_model.f_max_ui)
    ax[1].legend()

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
