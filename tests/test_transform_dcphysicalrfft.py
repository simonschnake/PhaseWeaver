import numpy as np
from numpy.testing import assert_allclose
from phase_weaver.core.current_profile import CurrentProfile
from phase_weaver.core.transform import DCPhysicalRFFT
from phase_weaver.core.form_factor import FormFactor

def test_dc_normalization_makes_mag0_one(grid, gaussian_density):
    prof = CurrentProfile(grid=grid, values=gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=True, unwrap_phase=True)

    # Depending on your final API:
    # ff = tr.profile_to_form_factor(prof)
    # For your CURRENT code, it returns (grid, mag, phase, eps):
    g2, mag, phase, eps = tr.profile_to_form_factor(prof)

    assert_allclose(mag[0], 1.0, atol=1e-12)

def test_inverse_produces_nonneg_and_normalized_density(grid, gaussian_density):
    prof = CurrentProfile(grid=grid, values=gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=True, unwrap_phase=True)

    g2, mag, phase, eps = tr.profile_to_form_factor(prof)
    ff = FormFactor(grid=g2, mag=mag, phase=phase, eps=eps)

    g3, dens = tr.form_factor_to_profile(ff)

    assert np.all(dens >= 0)
    assert_allclose(np.trapezoid(dens, g3.t), 1.0, atol=1e-12)

def test_gaussian_fixture_is_density(grid, gaussian_density):
    assert np.all(gaussian_density >= 0)
    assert np.isclose(np.trapezoid(gaussian_density, grid.t), 1.0, atol=1e-12)

def test_roundtrip_profile_to_ff_to_profile_close(grid, gaussian_density):
    prof0 = CurrentProfile(grid=grid, values=gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=True)

    g2, mag, phase, eps = tr.profile_to_form_factor(prof0)
    ff = FormFactor(grid=g2, mag=mag, phase=phase, eps=eps)

    g3, dens = tr.form_factor_to_profile(ff)
    prof1 = CurrentProfile(grid=g3, values=dens)

    # --- hard invariants ---
    assert np.all(prof1.density >= 0)
    assert np.isclose(np.trapezoid(prof1.density, grid.t), 1.0, atol=1e-12)

    # --- compare only where prof0 is "non-negligible" ---
    # (tails can change a lot because of clipping/renormalization)
    thresh = prof0.density.max() * 1e-8
    mask = prof0.density > thresh
    assert mask.any()

    # Compare central/support region
    assert_allclose(prof1.density[mask], prof0.density[mask], rtol=1e-6, atol=0.0)

    # --- global difference metric ---
    l1 = np.trapezoid(np.abs(prof1.density - prof0.density), grid.t)
    assert l1 < 1e-4


def test_dc_matches_integral_when_not_normalized(grid, gaussian_density):
    prof = CurrentProfile(grid=grid, values=gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=False)
    _, mag, _, _ = tr.profile_to_form_factor(prof)
    assert np.isclose(mag[0], 1.0, atol=1e-10)

def test_time_shift_produces_linear_phase_ramp(grid):
    t = grid.t
    sigma = 30e-15
    p0 = np.exp(-0.5 * (t / sigma) ** 2)
    p0 /= np.trapezoid(p0, t)

    k = 15
    p1 = np.roll(p0, k)

    prof0 = CurrentProfile(grid, p0)
    prof1 = CurrentProfile(grid, p1)

    # IMPORTANT: unwrap_phase=False here; we'll unwrap the ratio ourselves
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=False)

    _, mag0, ph0, _ = tr.profile_to_form_factor(prof0)
    _, mag1, ph1, _ = tr.profile_to_form_factor(prof1)

    F0 = mag0 * np.exp(1j * ph0)
    F1 = mag1 * np.exp(1j * ph1)

    f = grid.f_pos
    t0 = k * grid.dt

    # Phase of the ratio cancels arbitrary global offsets
    dphi = np.unwrap(np.angle(F1 * np.conj(F0)))

    # Only trust bins where magnitude is not tiny
    mag_floor = max(mag0.max(), mag1.max()) * 1e-6
    m = (mag0 > mag_floor) & (mag1 > mag_floor) & (f > 0)

    # Fit dphi ≈ slope * f + intercept; slope should be about -2π t0
    slope, intercept = np.polyfit(f[m], dphi[m], 1)

    assert abs(slope + 2 * np.pi * t0) < 0.05  # rad/Hz tolerance on slope

    # Optional: check residual is small too (after removing best-fit line)
    resid = dphi[m] - (slope * f[m] + intercept)
    assert np.max(np.abs(resid)) < 0.2
    
def test_centered_gaussian_phase_near_zero(grid, gaussian_density):
    prof = CurrentProfile(grid, gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=True)
    _, mag, phase, _ = tr.profile_to_form_factor(prof)

    # avoid bins where magnitude is tiny (phase meaningless)
    mask = mag > mag.max() * 1e-6
    assert np.max(np.abs(phase[mask])) < 1e-2


def test_inverse_enforces_projection_constraints(grid, gaussian_density):
    prof = CurrentProfile(grid, gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=True)
    g2, mag, phase, eps = tr.profile_to_form_factor(prof)

    # perturb phase to force some negative ringing in time after inverse
    phase2 = phase + 0.3*np.sin(np.linspace(0, 10, phase.size))
    ff = FormFactor(g2, mag, phase2, eps=eps)

    g3, dens = tr.form_factor_to_profile(ff)
    assert np.all(dens >= 0)
    assert np.isclose(np.trapezoid(dens, g3.t), 1.0, atol=1e-12)

def rms(t, p):
    mu = np.trapezoid(t*p, t)
    return np.sqrt(np.trapezoid((t-mu)**2*p, t))

def test_roundtrip_preserves_rms_width(grid, gaussian_density):
    prof0 = CurrentProfile(grid, gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=True)
    g2, mag, phase, eps = tr.profile_to_form_factor(prof0)
    ff = FormFactor(g2, mag, phase, eps=eps)
    g3, dens = tr.form_factor_to_profile(ff)

    w0 = rms(grid.t, prof0.density)
    w1 = rms(g3.t, dens)
    assert abs(w1 - w0) / w0 < 1e-3


def test_centered_gaussian_has_near_zero_phase(grid, gaussian_density):
    prof = CurrentProfile(grid, gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=True)
    _, mag, phase, _ = tr.profile_to_form_factor(prof)

    m = mag > mag.max() * 1e-6
    # Gaussian FT is positive-real; phase should be ~0 where signal exists
    assert np.max(np.abs(phase[m])) < 1e-2

def test_formfactor_from_profile_smoke(grid, gaussian_density):
    prof = CurrentProfile(grid, gaussian_density)
    ff = FormFactor.from_profile(prof)  # uses default transform
    assert ff.mag.shape == (grid.N // 2 + 1,)

def test_currentprofile_from_formfactor_smoke(grid, gaussian_density):
    prof = CurrentProfile(grid, gaussian_density)
    ff = prof.to_form_factor()
    prof2 = CurrentProfile.from_form_factor(ff)
    assert prof2.density.shape == (grid.N,)