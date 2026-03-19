# tests/test_reconstruction.py
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from phase_weaver.core.base import DCPhysicalRFFT, FormFactor, Grid, Profile
from phase_weaver.core.constraints import (
    ClampMagnitude,
    EnforceDCOne,
    CombinedFrequencyConstraint,
)
from phase_weaver.core.reconstruction import (
    GSMagnitudeOnly,
    GaussianInitializer,
    MagnitudeInitializer,
    MaxIter,
    MinimumPhaseCepstrum,
    MinIter,
    PhaseStoppedChanging,
    PredefinedPhase,
    RandomPhase,
    CombinedStopCriterion,
    ZeroPhase,
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


def _processed_meas_mag(grid: Grid, mag: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    """Apply the same frequency constraints as GSMagnitudeOnly uses."""
    ff0 = FormFactor(grid=grid, mag=mag.copy(), phase=np.zeros_like(mag), eps=eps)
    CombinedFrequencyConstraint(ClampMagnitude(eps=1e-6), EnforceDCOne()).apply(ff0)
    return ff0.mag.copy()


# -----------------------------------------------------------------------------
# Phase initializers
# -----------------------------------------------------------------------------


def test_zero_phase_initializer_shape(grid: Grid):
    mag = np.ones(grid.N // 2 + 1)
    ff = ZeroPhase()(grid=grid, mag=mag)
    assert ff.mag.shape == ff.phase.shape
    assert_allclose(ff.phase, 0.0)


def test_random_phase_initializer_deterministic(grid: Grid):
    init1 = RandomPhase().initialize_phase(grid, eps_mag=1e-30, seed=42)
    init2 = RandomPhase().initialize_phase(grid, eps_mag=1e-30, seed=42)
    assert init1.shape == (grid.N // 2 + 1,)
    assert_allclose(init1, init2)


def test_minimum_phase_initializer_shape_mismatch_raises(grid: Grid):
    mag_bad = np.ones(grid.N // 2)  # wrong length
    with pytest.raises(ValueError):
        MinimumPhaseCepstrum().initialize_phase(grid, mag=mag_bad)


def test_minimum_phase_initializer_mag_not_defined_raises(grid: Grid):
    with pytest.raises(ValueError):
        MinimumPhaseCepstrum().initialize_phase(grid, mag=None)


def test_minimum_phase_initializer_returns_finite(grid: Grid):
    mag = np.linspace(1.0, 0.1, grid.N // 2 + 1)
    ph = MinimumPhaseCepstrum().initialize_phase(grid, mag=mag)
    assert ph.shape == mag.shape
    assert np.isfinite(ph).all()


# -----------------------------------------------------------------------------
# Stop criteria
# -----------------------------------------------------------------------------


def test_maxiter_stops_after_n_updates(
    grid: Grid, gaussian_density: np.ndarray, measured_mag: np.ndarray
):
    stop = MaxIter(3)
    stop.reset()
    assert stop() is False
    stop.update()  # 1
    assert stop() is False
    stop.update()  # 2
    assert stop() is False
    stop.update()  # 3
    assert stop() is True


def test_stop_criteria_maxiter_miniter():
    max_iter = MaxIter(5)
    min_iter = MinIter(3)
    stop = max_iter + min_iter  # combined stop criterion
    stop.reset()
    assert stop() is False
    assert max_iter() is False
    assert min_iter() is False
    stop.update()  # iter 1
    stop.update()  # iter 2
    stop.update()  # iter 3 - min_iter should now allow stopping, but max_iter not yet
    assert stop() is False  # min_iter allows stop, but max_iter does not yet
    assert min_iter() is True
    stop.update()  # iter 4
    assert stop() is False  # still not at max_iter
    stop.update()  # iter 5
    assert stop() is True  # now max_iter allows stopping


def test_phase_stopped_changing_patience(
    grid: Grid, gaussian_density: np.ndarray, measured_mag: np.ndarray
):
    phase = np.zeros_like(measured_mag)
    ff = FormFactor(grid=grid, mag=measured_mag.copy(), phase=phase)

    s = PhaseStoppedChanging(tol=1e-12, patience=2)
    s.reset()
    s.update(last_ff=None, post_ff=ff)  # first update, last_ff is None, should not stop
    assert s.stop is False
    s.update(
        last_ff=ff, post_ff=ff
    )  # identical phase, mse = 0, but should not stop yet due to patience
    assert s.stop is False
    s.update(
        last_ff=ff, post_ff=ff
    )  # identical phase again, mse = 0, should now stop due to patience
    assert s.stop is True


def test_magnitude_initializer(grid: Grid):
    class DummyMagInit(MagnitudeInitializer):
        name = "dummy"

        def initialize_magnitude(self, grid: Grid) -> np.ndarray:
            return np.ones_like(grid.f_pos)

    mag_init = DummyMagInit()
    ff = mag_init(grid)
    assert ff.grid.N == grid.N
    assert ff.mag.shape == (grid.N // 2 + 1,)


def test_gaussian_initializer(grid: Grid):
    mag_init = GaussianInitializer()
    ff = mag_init(grid, sigma=0.1e-12)
    assert ff.grid.N == grid.N
    assert ff.mag.shape == (grid.N // 2 + 1,)
    assert np.all(ff.mag > 0)
    assert np.all(ff.phase == 0)
    with pytest.raises(KeyError):
        mag_init(grid)
    with pytest.raises(ValueError):
        mag_init(grid, sigma=-0.1e-12)


# -----------------------------------------------------------------------------
# GSMagnitudeOnly: validation and end-to-end
# -----------------------------------------------------------------------------


def test_gs_magnitude_only_rejects_wrong_mag_shape(grid: Grid):
    alg = GSMagnitudeOnly(stop=MaxIter(1))
    mag_bad = np.ones(grid.N // 2)  # wrong length
    with pytest.raises(ValueError):
        alg.run(mag_bad, grid, charge_C=250e-12)


def test_gs_does_not_mutate_input_mag(grid: Grid, measured_mag: np.ndarray):
    alg = GSMagnitudeOnly(stop=MaxIter(2))
    mag_in = measured_mag.copy()
    _prof, _ff = alg.run(mag_in, grid, charge_C=250e-12)
    assert_allclose(mag_in, measured_mag)


def test_gs_end_to_end_constraints_and_outputs(grid: Grid, measured_mag: np.ndarray):
    # Use a small fixed iteration count for deterministic runtime
    alg = GSMagnitudeOnly(stop=MaxIter(6))

    charge = 250e-12
    prof, ff = alg.run(measured_mag, grid, charge_C=charge)

    # --- Profile invariants from constraints ---
    assert prof.grid.N == grid.N
    assert prof.values.shape == (grid.N,)
    assert np.all(prof.values >= 0)

    area = np.trapezoid(prof.values, grid.t)
    assert np.isfinite(area)
    assert abs(area - 1.0) < 1e-8

    # Centering by first moment (should be close to 0; allow rounding to nearest sample)
    mu = first_moment(grid.t, prof.values, grid.dt)
    assert abs(mu) < 2.0 * grid.dt

    # charge attached
    assert prof.charge == pytest.approx(charge)

    # current property should match charge * density
    assert_allclose(prof.current, charge * prof.values)

    # --- FormFactor magnitude is the imposed measurement magnitude (after preprocessing) ---
    mag_meas = _processed_meas_mag(grid, measured_mag, eps=alg.eps_mag)
    assert ff.mag.shape == mag_meas.shape
    assert_allclose(
        ff.mag, mag_meas, rtol=0, atol=0.0
    )  # exact copy assignment in algorithm

    # DC normalized by EnforceDCOne
    assert ff.mag[0] == pytest.approx(1.0)


def test_gs_end_to_end_reconstruction_is_reasonable(
    grid: Grid, gaussian_density: np.ndarray
):
    """
    True end-to-end: generate a profile -> compute its magnitude -> reconstruct -> compare moments.

    This does NOT require pointwise equality (constraints + phase retrieval),
    but should preserve basic shape stats.
    """
    charge = 250e-12
    prof0 = Profile(grid=grid, values=gaussian_density, charge=charge)

    tr = DCPhysicalRFFT(unwrap_phase=True, dc_normalize=False)
    ff0 = prof0.to_form_factor(transform=tr)
    mag = ff0.mag.copy()

    alg = GSMagnitudeOnly(stop=MaxIter(10))
    prof1, _ff1 = alg.run(mag, grid, charge_C=charge)

    # Compare RMS width (should be close)gg
    t = grid.t

    def rms(p):
        mu = np.trapezoid(t * p, t)
        return np.sqrt(np.trapezoid((t - mu) ** 2 * p, t))

    w0 = rms(prof0.values)
    w1 = rms(prof1.values)
    assert abs(w1 - w0) / w0 < 5e-2  # loose but meaningful

    # Ensure reconstruction still satisfies constraints
    assert np.all(prof1.values >= 0)
    assert abs(np.trapezoid(prof1.values, t) - 1.0) < 1e-8


def test_predefined_phase(grid: Grid, measured_mag: np.ndarray):
    phase_init = PredefinedPhase(phase=np.zeros_like(measured_mag))

    phase = phase_init.initialize_phase(grid, mag=measured_mag, eps_mag=1e-30)
    assert phase.shape == measured_mag.shape
    assert_allclose(phase, 0.0)

    phase_init = PredefinedPhase(phase=np.zeros(1))  # wrong shape
    with pytest.raises(ValueError):
        phase_init.initialize_phase(grid, mag=measured_mag, eps_mag=1e-30)


def test_merge_stop_criteria():
    c1 = MinIter(3)
    c2 = PhaseStoppedChanging(tol=1e-8, patience=5)
    c3 = MaxIter(5)
    stop1 = c1 + c2
    stop2 = c2 + c3
    assert isinstance(stop1, CombinedStopCriterion)
    assert isinstance(stop2, CombinedStopCriterion)
    assert len(stop1.criteria) == 2
    assert len(stop2.criteria) == 2
    stop3 = stop1 + c3
    assert isinstance(stop3, CombinedStopCriterion)
    assert len(stop3.criteria) == 3
    stop4 = stop1 + stop2
    assert isinstance(stop4, CombinedStopCriterion)
    assert len(stop4.criteria) == 4

    with pytest.raises(ValueError):
        stop1 + "not a criterion"

