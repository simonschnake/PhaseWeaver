from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from phase_weaver.app.config import (
    PHASE_INIT_MODE,
    RECON_FREQUENCY_CONSTRAINT,
    RECON_STOP_CONDITION,
    RECON_TIME_CONSTRAINT,
)
from phase_weaver.app.logic import AppLogic
from phase_weaver.app.state import (
    ControlsState,
    MeasurementState,
    ProfileModelState,
    ReconstructionState,
)
from phase_weaver.core.base import DCPhysicalRFFT, FormFactor, Grid, Profile
from phase_weaver.core.measurement import MeasuredFormFactor
from phase_weaver.core.reconstruction import (
    CombinedStopCriterion,
    GerchbergSaxton,
    MaxIter,
    MeasurementsStoppedChanging,
    MinIter,
    PhaseStoppedChanging,
    _high_frequency_decay_bounds,
    minimum_phase_from_mag,
)
from phase_weaver.core.utils import trapz_uniform


def first_moment(t: np.ndarray, p: np.ndarray, dt: float) -> float:
    area = trapz_uniform(dt, p)
    if area <= 0:
        return float("nan")
    return trapz_uniform(dt, t * p) / area


@pytest.fixture
def measured_mag(grid: Grid, gaussian_density: np.ndarray) -> np.ndarray:
    prof = Profile(grid=grid, values=gaussian_density, charge=None)
    tr = DCPhysicalRFFT(unwrap_phase=True, dc_normalize=False)
    ff = prof.to_form_factor(transform=tr)
    return ff.mag.copy()


def test_minimum_phase_from_mag_returns_finite(grid: Grid):
    mag = np.linspace(1.0, 0.0, grid.N // 2 + 1)
    phase = minimum_phase_from_mag(grid, mag)

    assert phase.shape == mag.shape
    assert np.isfinite(phase).all()


def test_maxiter_stops_after_n_updates():
    stop = MaxIter(3)
    stop.reset()

    assert stop() is False
    stop.update()
    assert stop() is False
    stop.update()
    assert stop() is False
    stop.update()
    assert stop() is True


def test_miniter_defers_stop_until_threshold():
    stop = MinIter(2)
    stop.reset()

    assert stop() is False
    stop.update()
    assert stop() is False
    stop.update()
    assert stop() is True


def test_phase_stopped_changing_patience(grid: Grid):
    phase = np.zeros(grid.N // 2 + 1)
    ff = FormFactor(grid=grid, mag=np.ones_like(phase), phase=phase)

    stop = PhaseStoppedChanging(tol=1e-12, patience=2)
    stop.reset()

    stop.update(ff=ff)
    assert stop() is False
    stop.update(ff=ff)
    assert stop() is False
    assert stop() is True


def test_measurements_stopped_changing_patience(grid: Grid):
    measured = MeasuredFormFactor(freq=grid.f_pos, mag=np.ones_like(grid.f_pos))
    ff = FormFactor(grid=grid, mag=np.ones_like(grid.f_pos), phase=np.zeros_like(grid.f_pos))

    stop = MeasurementsStoppedChanging(measured_ff=(measured,), tol=1e-12, patience=2)
    stop.reset()

    stop.update(ff=ff)
    assert stop() is False
    stop.update(ff=ff)
    assert stop() is True


def test_combined_stop_criterion_uses_all_stop_when_only_iter_limits_apply():
    stop = MinIter(2) + MinIter(3)
    assert isinstance(stop, CombinedStopCriterion)

    stop.reset()
    assert stop() is False
    stop.update()
    assert stop() is False
    stop.update()
    assert stop() is False
    stop.update()
    assert stop() is True


def test_combined_stop_criterion_honors_direct_stop():
    stop = MaxIter(1) + MinIter(10)
    assert isinstance(stop, CombinedStopCriterion)

    stop.reset()
    assert stop() is False
    stop.update()
    assert stop() is True


def test_combined_stop_criterion_excludes_direct_only_criteria_from_all_stop():
    stop = MaxIter(100) + MinIter(2)

    stop.reset()
    assert stop() is False
    stop.update()
    assert stop() is False
    stop.update()
    assert stop() is True


def test_gerchberg_saxton_initializes_with_zero_magnitudes(grid: Grid):
    measured = MeasuredFormFactor(
        freq=grid.f_pos,
        mag=np.linspace(1.0, 0.0, len(grid.f_pos)),
    )

    alg = GerchbergSaxton(
        grid=grid,
        measurements=(measured,),
        reconstruction_state=SimpleNamespace(phase_init_mode=PHASE_INIT_MODE.ZERO),
    )

    assert alg.formfactor_init.mag.shape == measured.mag.shape
    assert np.isfinite(alg.formfactor_init.mag).all()
    assert np.all(alg.formfactor_init.mag >= 0)


def test_gerchberg_saxton_uses_selected_reconstruction_options(grid: Grid):
    measured = MeasuredFormFactor(
        freq=grid.f_pos,
        mag=np.linspace(1.0, 0.1, len(grid.f_pos)),
    )
    state = ReconstructionState(
        phase_init_mode=PHASE_INIT_MODE.ZERO,
        time_constraints={
            RECON_TIME_CONSTRAINT.NON_NEGATIVE,
            RECON_TIME_CONSTRAINT.NORMALIZE_AREA,
        },
        frequency_constraints={RECON_FREQUENCY_CONSTRAINT.BLEND_MEASURED},
        stop_conditions={RECON_STOP_CONDITION.MAX_ITER},
    )

    alg = GerchbergSaxton(
        grid=grid,
        measurements=(measured,),
        reconstruction_state=state,
    )

    assert [type(c).__name__ for c in alg.time_constraints._constraints] == [
        "NonNegativity",
        "NormalizeArea",
    ]
    assert [type(c).__name__ for c in alg.frequency_constraints._constraints] == [
        "BlendMeasuredMagnitude"
    ]
    assert [type(c).__name__ for c in alg.stop._criteria] == ["MaxIter"]


def test_empty_stop_selection_falls_back_to_max_iter(grid: Grid):
    measured = MeasuredFormFactor(
        freq=grid.f_pos,
        mag=np.linspace(1.0, 0.1, len(grid.f_pos)),
    )
    state = ReconstructionState(
        phase_init_mode=PHASE_INIT_MODE.ZERO,
        stop_conditions=set(),
    )

    alg = GerchbergSaxton(
        grid=grid,
        measurements=(measured,),
        reconstruction_state=state,
    )

    assert [type(c).__name__ for c in alg.stop._criteria] == ["MaxIter"]


def test_high_frequency_decay_bounds_start_after_last_measurement(grid: Grid):
    measured = MeasuredFormFactor(
        freq=grid.f_pos[:11],
        mag=np.ones(11),
    )

    bounds = _high_frequency_decay_bounds((measured,), grid)

    assert bounds == (grid.f_pos[10], grid.f_pos[20])


def test_gerchberg_saxton_last_phase_initialization_uses_previous_phase(grid: Grid):
    measured = MeasuredFormFactor(
        freq=grid.f_pos,
        mag=np.linspace(1.0, 0.1, len(grid.f_pos)),
    )
    phase_last = np.linspace(-0.5, 0.5, len(grid.f_pos))

    alg = GerchbergSaxton(
        grid=grid,
        measurements=(measured,),
        reconstruction_state=SimpleNamespace(phase_init_mode=PHASE_INIT_MODE.LAST),
        phase_last=phase_last,
    )

    assert_allclose(alg.formfactor_init.phase, phase_last)


def test_gerchberg_saxton_last_phase_initialization_falls_back_without_phase(
    grid: Grid, capsys: pytest.CaptureFixture[str]
):
    measured = MeasuredFormFactor(
        freq=grid.f_pos,
        mag=np.linspace(1.0, 0.1, len(grid.f_pos)),
    )

    alg = GerchbergSaxton(
        grid=grid,
        measurements=(measured,),
        reconstruction_state=SimpleNamespace(phase_init_mode=PHASE_INIT_MODE.LAST),
    )

    assert_allclose(alg.formfactor_init.phase, 0.0)
    assert "no previous phase found" in capsys.readouterr().out


def test_gerchberg_saxton_last_phase_initialization_rejects_wrong_shape(grid: Grid):
    measured = MeasuredFormFactor(
        freq=grid.f_pos,
        mag=np.linspace(1.0, 0.1, len(grid.f_pos)),
    )

    with pytest.raises(ValueError, match="last phase must match"):
        GerchbergSaxton(
            grid=grid,
            measurements=(measured,),
            reconstruction_state=SimpleNamespace(phase_init_mode=PHASE_INIT_MODE.LAST),
            phase_last=np.zeros(1),
        )


def test_gerchberg_saxton_run_can_be_forced_to_stop_quickly(
    grid: Grid, measured_mag: np.ndarray
):
    measured = MeasuredFormFactor(freq=grid.f_pos, mag=measured_mag)
    alg = GerchbergSaxton(
        grid=grid,
        measurements=(measured,),
        reconstruction_state=SimpleNamespace(phase_init_mode=PHASE_INIT_MODE.ZERO),
    )
    alg.stop = MaxIter(1)

    prof, ff = alg.run()

    assert prof.grid.N == grid.N
    assert prof.values.shape == (grid.N,)
    assert np.all(np.isfinite(prof.values))
    assert ff.mag.shape == measured_mag.shape
    assert np.isfinite(ff.mag).all()
    assert prof.charge is None

    area = np.trapezoid(prof.values, grid.t)
    assert np.isfinite(area)
    assert abs(area - 1.0) < 1e-8
    assert abs(first_moment(grid.t, prof.values, grid.dt)) < 2.0 * grid.dt


def test_default_app_reconstruction_stops_on_convergence_before_max_iter():
    logic = AppLogic()
    state = ControlsState(
        scenario=ProfileModelState(),
        measurement=MeasurementState(),
        reconstruction=ReconstructionState(phase_init_mode=PHASE_INIT_MODE.ZERO),
    )

    prof_input, ff_input, measurements = logic.compute_initial(state)
    _prof_recon, _ff_recon, summary = logic.compute_reconstruction(
        grid=prof_input.grid,
        measurements=measurements,
        controls_state=state,
        ff_input=ff_input,
    )

    assert summary.iterations < 1_000
    assert summary.stop_reason == (
        "phase_stopped_changing+min_iter+measurement_distance_below"
    )
    assert summary.measurement_error is not None
    assert summary.measurement_error < 1e-4


def test_crisp_only_reconstruction_matches_measured_band_before_max_iter():
    logic = AppLogic()
    state = ControlsState(
        scenario=ProfileModelState(),
        measurement=MeasurementState(crisp=True),
        reconstruction=ReconstructionState(phase_init_mode=PHASE_INIT_MODE.ZERO),
    )

    prof_input, ff_input, measurements = logic.compute_initial(state)
    _prof_recon, _ff_recon, summary = logic.compute_reconstruction(
        grid=prof_input.grid,
        measurements=measurements,
        controls_state=state,
        ff_input=ff_input,
    )

    assert summary.iterations < 1_000
    assert summary.stop_reason == (
        "phase_stopped_changing+min_iter+measurement_distance_below"
    )
    assert summary.measurement_error is not None
    assert summary.measurement_error < 1e-4

    high_freq = measurements[0].freq[-1] + (
        measurements[0].freq[-1] - measurements[0].freq[0]
    )
    high_mask = _ff_recon.grid.f_pos >= high_freq
    assert np.any(high_mask)
    assert_allclose(_ff_recon.mag[high_mask], 0.0, atol=1e-12)
