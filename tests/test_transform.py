import numpy as np
from numpy.testing import assert_allclose
import pytest

from phase_weaver.core import Profile, CurrentProfile, DCPhysicalRFFT, FormFactor, Transform


def test_dc_normalization_makes_mag0_one(grid, gaussian_density):
    prof = Profile(grid=grid, values=gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=True, unwrap_phase=True)
    ff = prof.to_form_factor(transform=tr)
    assert_allclose(ff.mag[0], 1.0, atol=1e-12)


def test_inverse_produces_nonneg_and_normalized_density(grid, gaussian_density):
    prof = Profile(grid=grid, values=gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=True, unwrap_phase=True)
    ff = FormFactor.from_profile(prof, transform=tr)
    prof2 = ff.to_profile(transform=tr)
    assert np.all(prof2.values >= 0)
    assert_allclose(np.trapezoid(prof2.values, grid.t), 1.0, atol=1e-12)


def test_gaussian_fixture_is_density(grid, gaussian_density):
    assert np.all(gaussian_density >= 0)
    assert np.isclose(np.trapezoid(gaussian_density, grid.t), 1.0, atol=1e-12)


def test_roundtrip_profile_to_ff_to_profile_close(grid, gaussian_density):
    prof0 = Profile(grid=grid, values=gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=True)
    ff = FormFactor.from_profile(prof0, transform=tr)
    prof1 = Profile.from_form_factor(ff, transform=tr)

    # --- hard invariants ---
    assert np.all(
        prof1.values >= -0.001 * prof1.values.max()
    )  # allow tiny negative due to numerical issues
    assert np.isclose(np.trapezoid(prof1.values, grid.t), 1.0, atol=1e-12)

    # --- compare only where prof0 is "non-negligible" ---
    # (tails can change a lot because of clipping/renormalization)
    thresh = prof0.values.max() * 1e-8
    mask = prof0.values > thresh
    assert mask.any()

    # Compare central/support region
    assert_allclose(prof1.values[mask], prof0.values[mask], rtol=1e-6, atol=0.0)

    # --- global difference metric ---
    l1 = np.trapezoid(np.abs(prof1.values - prof0.values), grid.t)
    assert l1 < 1e-4


def test_dc_matches_integral_when_not_normalized(grid, gaussian_density):
    prof = Profile(grid=grid, values=gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=False)
    ff = FormFactor.from_profile(prof, transform=tr)
    assert np.isclose(ff.mag[0], 1.0, atol=1e-10)


def test_time_shift_produces_linear_phase_ramp(grid):
    t = grid.t
    sigma = 30e-15
    p0 = np.exp(-0.5 * (t / sigma) ** 2)
    p0 /= np.trapezoid(p0, t)

    k = 15
    p1 = np.roll(p0, k)

    prof0 = Profile(grid, p0)
    prof1 = Profile(grid, p1)

    # IMPORTANT: unwrap_phase=False here; we'll unwrap the ratio ourselves
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=False)

    ff1 = FormFactor.from_profile(prof1, transform=tr)
    ff2 = FormFactor.from_profile(prof0, transform=tr)

    F0 = ff1.value
    F1 = ff2.value

    f = grid.f_pos
    t0 = k * grid.dt

    # Phase of the ratio cancels arbitrary global offsets
    dphi = np.unwrap(np.angle(F1 * np.conj(F0)))

    # Only trust bins where magnitude is not tiny
    mag_floor = max(ff1.mag.max(), ff2.mag.max()) * 1e-6
    m = (ff1.mag > mag_floor) & (ff2.mag > mag_floor) & (f > 0)

    # Fit dphi ≈ slope * f + intercept; slope should be about -2π t0
    slope, intercept = np.polyfit(f[m], dphi[m], 1)

    assert abs(slope + 2 * np.pi * t0) < 0.05  # rad/Hz tolerance on slope

    # Optional: check residual is small too (after removing best-fit line)
    resid = dphi[m] - (slope * f[m] + intercept)
    assert np.max(np.abs(resid)) < 0.2


def test_centered_gaussian_phase_near_zero(grid, gaussian_density):
    prof = CurrentProfile(grid, gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=True)
    ff = FormFactor.from_profile(prof, transform=tr)

    # avoid bins where magnitude is tiny (phase meaningless)
    mask = ff.mag > ff.mag.max() * 1e-6
    assert np.max(np.abs(ff.phase[mask])) < 1e-2


def test_inverse_enforces_projection_constraints(grid, gaussian_density):
    prof0 = Profile(grid, gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=True)
    ff = FormFactor.from_profile(prof0, transform=tr)

    # perturb phase to force some negative ringing in time after inverse
    phase1 = ff.phase + 0.3 * np.sin(np.linspace(0, 10, ff.phase.size))
    ff1 = FormFactor(ff.grid, ff.mag, phase1, eps=ff.eps)

    prof1 = Profile.from_form_factor(ff1)

    assert np.all(
        prof1.values >= -0.0001 * prof1.values.max()
    )  # allow small negative due to numerical issues
    assert np.isclose(np.trapezoid(prof1.values, prof1.grid.t), 1.0, atol=1e-12)


def rms(t, p):
    mu = np.trapezoid(t * p, t)
    return np.sqrt(np.trapezoid((t - mu) ** 2 * p, t))


def test_roundtrip_preserves_rms_width(grid, gaussian_density):
    prof0 = Profile(grid, gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=True)
    g2, mag, phase, eps = tr.profile_to_form_factor(prof0)
    ff = FormFactor(g2, mag, phase, eps=eps)
    prof1 = Profile.from_form_factor(ff, transform=tr)

    w0 = rms(grid.t, prof0.values)
    w1 = rms(prof1.grid.t, prof1.values)
    assert abs(w1 - w0) / w0 < 1e-3


def test_centered_gaussian_has_near_zero_phase(grid, gaussian_density):
    prof = Profile(grid, gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=True)
    _, mag, phase, _ = tr.profile_to_form_factor(prof)

    m = mag > mag.max() * 1e-6
    # Gaussian FT is positive-real; phase should be ~0 where signal exists
    assert np.max(np.abs(phase[m])) < 1e-2


def test_formfactor_from_profile_smoke(grid, gaussian_density):
    prof = Profile(grid, gaussian_density)
    ff = FormFactor.from_profile(prof)  # uses default transform
    assert ff.mag.shape == (grid.N // 2 + 1,)


def test_currentprofile_from_formfactor_smoke(grid, gaussian_density):
    prof = Profile(grid, gaussian_density)
    ff = prof.to_form_factor()
    prof2 = CurrentProfile.from_form_factor(ff)
    assert prof2.values.shape == (grid.N,)


def test_transform_raise_not_implemented(grid, gaussian_density):
    class DummyTransform(Transform):
        def profile_to_form_factor(self, profile: CurrentProfile):
            return super().profile_to_form_factor(profile)

        def form_factor_to_profile(self, form_factor: FormFactor):
            return super().form_factor_to_profile(form_factor)

    tr = DummyTransform()
    prof = Profile(grid=grid, values=gaussian_density)
    ff = FormFactor.from_profile(prof)
    
    with pytest.raises(NotImplementedError):
        tr.profile_to_form_factor(prof)
    with pytest.raises(NotImplementedError):
        tr.form_factor_to_profile(ff)