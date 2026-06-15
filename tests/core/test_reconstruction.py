from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from phase_weaver.app.config import PHASE_INIT_MODE
from phase_weaver.core.base import DCPhysicalRFFT, FormFactor, Grid, Profile
from phase_weaver.core.measurement import MeasuredFormFactor
from phase_weaver.core.reconstruction import (
    CombinedStopCriterion,
    GerchbergSaxton,
    MaxIter,
    MeasurementsStoppedChanging,
    MinIter,
    PhaseStoppedChanging,
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
