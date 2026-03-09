from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from phase_weaver.core import (
    BandLimitedDCPhysicalRFFT,
    CurrentProfile,
    DCPhysicalRFFT,
    FormFactor,
    Profile,
    Transform,
)


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
    ff1 = FormFactor(ff.grid, ff.mag, phase1)

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
    g2, mag, phase = tr.profile_to_form_factor(prof0)
    ff = FormFactor(g2, mag, phase)
    prof1 = Profile.from_form_factor(ff, transform=tr)

    w0 = rms(grid.t, prof0.values)
    w1 = rms(prof1.grid.t, prof1.values)
    assert abs(w1 - w0) / w0 < 1e-3


def test_centered_gaussian_has_near_zero_phase(grid, gaussian_density):
    prof = Profile(grid, gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=True)
    _, mag, phase = tr.profile_to_form_factor(prof)

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


def test_bandlimited_rejects_nonpositive_cutoff():
    with pytest.raises(ValueError, match="f_cut must be positive."):
        BandLimitedDCPhysicalRFFT(f_cut=0.0)

    with pytest.raises(ValueError, match="f_cut must be positive."):
        BandLimitedDCPhysicalRFFT(f_cut=-1.0)


def test_compact_forward_truncates_and_f_pos_used_matches(grid, gaussian_density):
    prof = Profile(grid=grid, values=gaussian_density)

    k_cut = 12
    f_cut = (k_cut + 0.49) * grid.df

    tr = BandLimitedDCPhysicalRFFT(
        f_cut=f_cut,
        compact=True,
        dc_normalize=False,
        unwrap_phase=True,
    )
    base = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=True)

    g1, mag, phase = tr.profile_to_form_factor(prof)
    g2, mag_full, phase_full = base.profile_to_form_factor(prof)

    assert g1 == g2 == grid
    assert mag.shape == (k_cut + 1,)
    assert phase.shape == (k_cut + 1,)

    assert_allclose(mag, mag_full[: k_cut + 1], rtol=0, atol=0)
    assert_allclose(phase, phase_full[: k_cut + 1], rtol=0, atol=0)
    assert_allclose(tr.f_pos_used(grid), grid.f_pos[: k_cut + 1], rtol=0, atol=0)


def test_cutoff_at_or_above_nyquist_keeps_full_spectrum(grid, gaussian_density):
    prof = Profile(grid=grid, values=gaussian_density)

    tr = BandLimitedDCPhysicalRFFT(
        f_cut=2.0 * grid.f_nyq,
        compact=True,
        dc_normalize=False,
        unwrap_phase=True,
    )
    base = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=True)

    _, mag, phase = tr.profile_to_form_factor(prof)
    _, mag_ref, phase_ref = base.profile_to_form_factor(prof)

    assert mag.shape == (grid.N // 2 + 1,)
    assert phase.shape == (grid.N // 2 + 1,)
    assert_allclose(mag, mag_ref, rtol=0, atol=0)
    assert_allclose(phase, phase_ref, rtol=0, atol=0)
    assert_allclose(tr.f_pos_used(grid), grid.f_pos, rtol=0, atol=0)


def test_noncompact_forward_zeroes_bins_above_cutoff(grid, gaussian_density):
    prof = Profile(grid=grid, values=gaussian_density)

    k_cut = 9
    f_cut = (k_cut + 0.25) * grid.df

    tr = BandLimitedDCPhysicalRFFT(
        f_cut=f_cut,
        compact=False,
        dc_normalize=False,
        unwrap_phase=True,
    )
    base = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=True)

    _, mag_lp, phase_lp = tr.profile_to_form_factor(prof)
    _, mag_full, phase_full = base.profile_to_form_factor(prof)

    assert mag_lp.shape == (grid.N // 2 + 1,)
    assert phase_lp.shape == (grid.N // 2 + 1,)

    assert_allclose(mag_lp[: k_cut + 1], mag_full[: k_cut + 1], rtol=0, atol=0)
    assert_allclose(phase_lp[: k_cut + 1], phase_full[: k_cut + 1], rtol=0, atol=0)

    assert_allclose(mag_lp[k_cut + 1 :], 0.0, rtol=0, atol=0)
    assert_allclose(phase_lp[k_cut + 1 :], 0.0, rtol=0, atol=0)


def test_compact_inverse_matches_full_length_zero_padded_reference(
    grid, gaussian_density
):
    prof = Profile(grid=grid, values=gaussian_density)

    k_cut = 11
    f_cut = (k_cut + 0.1) * grid.df

    tr_compact = BandLimitedDCPhysicalRFFT(
        f_cut=f_cut,
        compact=True,
        dc_normalize=False,
        unwrap_phase=True,
    )
    base = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=True)

    _, mag_c, phase_c = tr_compact.profile_to_form_factor(prof)

    # compact path: len(mag) < N//2 + 1
    ff_compact = SimpleNamespace(grid=grid, mag=mag_c, phase=phase_c)
    g_compact, x_compact = tr_compact.form_factor_to_profile(ff_compact)

    # reference: same spectrum, explicitly zero-padded to full one-sided length
    n_full = grid.N // 2 + 1
    mag_full = np.zeros(n_full, dtype=float)
    phase_full = np.zeros(n_full, dtype=float)
    mag_full[: len(mag_c)] = mag_c
    phase_full[: len(phase_c)] = phase_c

    ff_full = FormFactor(grid=grid, mag=mag_full, phase=phase_full)
    g_ref, x_ref = base.form_factor_to_profile(ff_full)

    assert g_compact == g_ref == grid
    assert x_compact.shape == (grid.N,)
    assert_allclose(x_compact, x_ref, rtol=1e-12, atol=1e-12)


def test_full_length_inverse_matches_base_inverse(grid, gaussian_density):
    prof = Profile(grid=grid, values=gaussian_density)

    tr = BandLimitedDCPhysicalRFFT(
        f_cut=7.5 * grid.df,
        compact=False,  # ensures full-length mag/phase
        dc_normalize=False,
        unwrap_phase=True,
    )
    base = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=True)

    ff = FormFactor.from_profile(prof, transform=tr)

    g1, x1 = tr.form_factor_to_profile(ff)
    g2, x2 = base.form_factor_to_profile(ff)

    assert g1 == g2 == grid
    assert x1.shape == (grid.N,)
    assert_allclose(x1, x2, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "mag, phase, msg",
    [
        (np.zeros((2, 2)), np.zeros((2, 2)), "mag and phase must be 1D arrays."),
        (np.zeros(3), np.zeros(4), "mag and phase must have the same shape."),
    ],
)
def test_inverse_validates_mag_phase_shape(grid, mag, phase, msg):
    tr = BandLimitedDCPhysicalRFFT(f_cut=1.0)

    ff_bad = SimpleNamespace(grid=grid, mag=mag, phase=phase)
    with pytest.raises(ValueError, match=msg):
        tr.form_factor_to_profile(ff_bad)


def test_inverse_rejects_too_many_bins(grid):
    tr = BandLimitedDCPhysicalRFFT(f_cut=1.0)

    n_too_many = grid.N // 2 + 2
    ff_bad = SimpleNamespace(
        grid=grid,
        mag=np.zeros(n_too_many, dtype=float),
        phase=np.zeros(n_too_many, dtype=float),
    )

    with pytest.raises(
        ValueError, match="Too many positive-frequency bins for this grid."
    ):
        tr.form_factor_to_profile(ff_bad)


def test_sub_df_cutoff_keeps_only_dc_and_reconstructs_constant_profile(
    grid, gaussian_density
):
    prof = Profile(grid=grid, values=gaussian_density)

    tr = BandLimitedDCPhysicalRFFT(
        f_cut=0.5 * grid.df,  # floor(f_cut / df) == 0
        compact=True,
        dc_normalize=False,
        unwrap_phase=True,
    )

    _, mag, phase = tr.profile_to_form_factor(prof)
    ff_dc = SimpleNamespace(grid=grid, mag=mag, phase=phase)
    _, x = tr.form_factor_to_profile(ff_dc)

    assert mag.shape == (1,)
    assert phase.shape == (1,)
    assert x.shape == (grid.N,)
    assert_allclose(x, np.full_like(x, x[0]), atol=1e-12)
