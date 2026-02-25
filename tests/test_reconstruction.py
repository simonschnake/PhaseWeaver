# tests/test_reconstruction.py
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from phase_weaver.core.base import Grid, Profile, FormFactor, DCPhysicalRFFT
from phase_weaver.core.reconstruction import (
    PhaseInitializer,
    StopCriterion,
    ZeroPhase,
    RandomPhase,
    MinimumPhaseCepstrum,
    StopCriteria,
    MaxIter,
    PhaseStoppedChanging,
    GSMagnitudeOnly,
)
from phase_weaver.core.constraints import (
    ClampMagnitude,
    EnforceDCOne,
    FrequencyConstraints,
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
    FrequencyConstraints(ClampMagnitude(eps=1e-6), EnforceDCOne()).apply(ff0)
    return ff0.mag.copy()


# -----------------------------------------------------------------------------
# Phase initializers
# -----------------------------------------------------------------------------

def test_zero_phase_initializer_shape(grid: Grid):
    mag = np.ones(grid.N // 2 + 1)
    ph = ZeroPhase().init_phase(mag, grid, eps_mag=1e-30)
    assert ph.shape == mag.shape
    assert_allclose(ph, 0.0)


def test_random_phase_initializer_deterministic(grid: Grid):
    mag = np.ones(grid.N // 2 + 1)
    init1 = RandomPhase(seed=123).init_phase(mag, grid, eps_mag=1e-30)
    init2 = RandomPhase(seed=123).init_phase(mag, grid, eps_mag=1e-30)
    assert init1.shape == mag.shape
    assert_allclose(init1, init2)


def test_minimum_phase_initializer_shape_mismatch_raises(grid: Grid):
    mag_bad = np.ones(grid.N // 2)  # wrong length
    with pytest.raises(ValueError):
        MinimumPhaseCepstrum().init_phase(mag_bad, grid, eps_mag=1e-30)


def test_minimum_phase_initializer_returns_finite(grid: Grid):
    mag = np.linspace(1.0, 0.1, grid.N // 2 + 1)
    ph = MinimumPhaseCepstrum().init_phase(mag, grid, eps_mag=1e-30)
    assert ph.shape == mag.shape
    assert np.isfinite(ph).all()


# -----------------------------------------------------------------------------
# Stop criteria
# -----------------------------------------------------------------------------

def test_maxiter_stops_after_n_updates(grid: Grid, gaussian_density: np.ndarray, measured_mag: np.ndarray):
    prof = Profile(grid=grid, values=gaussian_density, charge=None)
    ff = FormFactor(grid=grid, mag=measured_mag.copy(), phase=np.zeros_like(measured_mag))

    s = MaxIter(3)
    s.reset()
    assert s.update(prof, ff) is False  # 1
    assert s.update(prof, ff) is False  # 2
    assert s.update(prof, ff) is True   # 3


def test_phase_stopped_changing_patience(grid: Grid, gaussian_density: np.ndarray, measured_mag: np.ndarray):
    prof = Profile(grid=grid, values=gaussian_density, charge=None)
    phase = np.zeros_like(measured_mag)
    ff = FormFactor(grid=grid, mag=measured_mag.copy(), phase=phase)

    s = PhaseStoppedChanging(tol=1e-12, patience=3)
    s.reset()

    # First call: last phase not set => inf mse => no stop
    assert s.update(prof, ff) is False

    # Now keep phase identical -> mse ~ 0, should stop after patience
    assert s.update(prof, ff) is False
    assert s.update(prof, ff) is False
    assert s.update(prof, ff) is True


def test_stopcriteria_any_and_all(grid: Grid, gaussian_density: np.ndarray, measured_mag: np.ndarray):
    prof = Profile(grid=grid, values=gaussian_density, charge=None)
    ff = FormFactor(grid=grid, mag=measured_mag.copy(), phase=np.zeros_like(measured_mag))

    class AlwaysTrue(MaxIter):
        def update(self, prof, ff) -> bool:
            return True

    class AlwaysFalse(MaxIter):
        def update(self, prof, ff) -> bool:
            return False

    s_any = StopCriteria(AlwaysFalse(1), AlwaysTrue(1), logic="any")
    s_any.reset()
    assert s_any.update(prof, ff) is True

    s_all = StopCriteria(AlwaysFalse(1), AlwaysTrue(1), logic="all")
    s_all.reset()
    assert s_all.update(prof, ff) is False


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
    assert_allclose(ff.mag, mag_meas, rtol=0, atol=0.0)  # exact copy assignment in algorithm

    # DC normalized by EnforceDCOne
    assert ff.mag[0] == pytest.approx(1.0)


def test_gs_end_to_end_reconstruction_is_reasonable(grid: Grid, gaussian_density: np.ndarray):
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


def test_raise_nonimpl_phase_initializer(grid: Grid):
    class NotImplementedPhaseInit(PhaseInitializer):
        name = "not_implemented"
        def init_phase(self, mag_pos: np.ndarray, grid: Grid, eps_mag: float) -> np.ndarray:
            super().init_phase(mag_pos, grid, eps_mag=eps_mag)
    not_imp = NotImplementedPhaseInit()


    with pytest.raises(NotImplementedError):
        not_imp.init_phase(np.ones(grid.N // 2 + 1), grid, eps_mag=1e-30)


def test_raise_logic_error_in_stopcriteria(grid: Grid):
    with pytest.raises(ValueError):
        StopCriteria(MaxIter(1), logic="invalid_logic")


def test_stopcriteria_all_logic(grid: Grid, gaussian_density: np.ndarray, measured_mag: np.ndarray):
    prof = Profile(grid=grid, values=gaussian_density, charge=None)
    ff = FormFactor(grid=grid, mag=measured_mag.copy(), phase=np.zeros_like(measured_mag))

    class AlwaysTrue(MaxIter):
        def update(self, prof, ff) -> bool:
            return True

    class AlwaysFalse(MaxIter):
        def update(self, prof, ff) -> bool:
            return False

    s_all = StopCriteria(AlwaysTrue(1), AlwaysTrue(1), logic="all")
    s_all.reset()
    assert s_all.update(prof, ff) is True  # both must be True to stop
