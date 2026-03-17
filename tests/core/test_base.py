# tests/core/test_base.py
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from phase_weaver.core import (
    BandLimitedDCPhysicalRFFT,
    CurrentProfile,
    DCPhysicalRFFT,
    FormFactor,
    Grid,
    Profile,
    Transform,
)
from phase_weaver.core.base import next_pow2
from phase_weaver.core.constraints import ClampMagnitude


def test_next_pow2_basic():
    assert next_pow2(0) == 1
    assert next_pow2(1) == 1
    assert next_pow2(2) == 2
    assert next_pow2(3) == 4
    assert next_pow2(4) == 4
    assert next_pow2(5) == 8
    assert next_pow2(63) == 64
    assert next_pow2(64) == 64
    assert next_pow2(65) == 128


def test_from_dt_tmax_rejects_nonpositive():
    with pytest.raises(ValueError):
        Grid.from_dt_tmax(dt=0.0, t_max=1.0)
    with pytest.raises(ValueError):
        Grid.from_dt_tmax(dt=1.0, t_max=0.0)
    with pytest.raises(ValueError):
        Grid.from_dt_tmax(dt=-1.0, t_max=1.0)
    with pytest.raises(ValueError):
        Grid(N=-64, dt=1e-15)
    with pytest.raises(ValueError):
        Grid(N=64, dt=-1e-15)


def test_from_df_fmax_rejects_nonpositive():
    with pytest.raises(ValueError):
        Grid.from_df_fmax(df=0.0, f_max=1.0)
    with pytest.raises(ValueError):
        Grid.from_df_fmax(df=1.0, f_max=0.0)
    with pytest.raises(ValueError):
        Grid.from_df_fmax(df=-1.0, f_max=1.0)


def test_from_dt_tmax_even_and_minN_pow2():
    g = Grid.from_dt_tmax(dt=1e-15, t_max=2e-12, snap_pow2=True, min_N=64)
    assert g.N >= 64
    assert g.N % 2 == 0
    # snap_pow2=True => power-of-two
    assert (g.N & (g.N - 1)) == 0


def test_from_dt_tmax_no_snap_not_forced_pow2():
    g = Grid.from_dt_tmax(dt=1e-15, t_max=2e-12, snap_pow2=False, min_N=64)
    assert g.N >= 64
    assert g.N % 2 == 0
    # not necessarily power-of-two; just ensure it's a plain int and consistent
    assert isinstance(g.N, int)


def test_from_df_fmax_df_consistency():
    df = 5e11
    fmax = 2e13
    g = Grid.from_df_fmax(df=df, f_max=fmax, snap_pow2=True, min_N=64)
    # df should match the provided target within floating error:
    # dt is chosen as 1/(N*df_target), and df property recomputes exactly df_target.
    assert np.isclose(g.df, df, rtol=0, atol=0.0)


def test_from_df_fmax_nyquist_covers_requested_fmax():
    df = 5e11
    fmax = 2e13
    g = Grid.from_df_fmax(df=df, f_max=fmax, snap_pow2=True, min_N=64)
    # construction ensures (N/2)*df >= fmax, which is f_nyq >= fmax
    assert g.f_nyq >= fmax


def test_df_dt_N_identity():
    g = Grid.from_dt_tmax(dt=2e-15, t_max=1e-12, snap_pow2=True, min_N=64)
    # df*dt*N == 1
    assert np.isclose(g.df * g.dt * g.N, 1.0, rtol=0, atol=1e-12)


def test_axes_lengths_and_endpoints():
    g = Grid.from_dt_tmax(dt=1e-15, t_max=2e-12, snap_pow2=True, min_N=64)

    t = g.t
    f_pos = g.f_pos
    f_shift = g.f_shift

    assert len(t) == g.N
    assert len(f_pos) == g.N // 2 + 1
    assert len(f_shift) == g.N

    # time axis is centered: symmetric range with step dt
    assert np.isclose(t[1] - t[0], g.dt, atol=0.0)
    assert np.isclose(t[g.N // 2], 0.0, atol=0.0)  # index half is exactly 0

    # frequency axes
    assert np.isclose(f_pos[0], 0.0, atol=0.0)
    assert np.isclose(f_pos[-1], g.f_nyq, atol=1e-12)
    assert np.isclose(f_shift[0], -g.f_nyq, atol=1e-12)  # -N/2 * df
    assert np.isclose(f_shift[g.N // 2], 0.0, atol=1e-12)


def test_tmax_coverage_condition():
    # Ensure (N/2)*dt >= t_max for from_dt_tmax
    dt = 1e-15
    t_max = 2.5e-12
    g = Grid.from_dt_tmax(dt=dt, t_max=t_max, snap_pow2=True, min_N=64)
    assert (g.N / 2) * g.dt >= t_max


def test_fmax_coverage_condition():
    # Ensure (N/2)*df >= f_max for from_df_fmax
    df = 1e12
    f_max = 3.3e13
    g = Grid.from_df_fmax(df=df, f_max=f_max, snap_pow2=True, min_N=64)
    assert (g.N / 2) * g.df >= f_max


def test_f_pos_monotonic_and_nyquist(grid):
    f = grid.f_pos
    assert np.all(np.diff(f) > 0)
    assert np.isclose(f[0], 0.0)
    assert np.isclose(f[-1], grid.f_nyq)


def test_uneven_n_for_grid():
    # Ensure that if we try to create a grid with odd N, it gets adjusted to even
    g = Grid.from_dt_tmax(dt=1, t_max=33.5, snap_pow2=False)
    assert g.N % 2 == 0
    g = Grid.from_df_fmax(df=1, f_max=33.5, snap_pow2=False)
    assert g.N % 2 == 0


def test_formfactor_clamps_magnitude(grid):
    mag = np.array([0.0, 1.0, 2.0])
    phase = np.zeros_like(mag)
    ff = FormFactor(
        grid=grid, mag=mag, phase=phase, frequency_constraint=ClampMagnitude(eps=0.5)
    )
    assert ff.mag[0] >= 1e-6


def test_formfactor_shapes_match_grid(grid, gaussian_density):
    prof = CurrentProfile(grid, gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=False)
    g2, mag, phase = tr.profile_to_form_factor(prof)
    assert g2.N == grid.N
    assert mag.shape == (grid.N // 2 + 1,)
    assert phase.shape == (grid.N // 2 + 1,)


def test_formfactor_apply_constraint(grid, gaussian_density):
    prof = CurrentProfile(grid, gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=False)
    ff = FormFactor.from_profile(prof, transform=tr)
    ff2 = FormFactor(
        grid=ff.grid,
        mag=ff.mag.copy(),
        phase=ff.phase.copy(),
        frequency_constraint=ClampMagnitude(eps=0.5),
    )
    assert np.all(ff2.mag >= 0.5)


def test_formfactor_copy(grid, gaussian_density):
    prof = CurrentProfile(grid, gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=False)
    ff = FormFactor.from_profile(prof, transform=tr)
    ff2 = ff.copy()
    assert ff2.grid.N == grid.N
    assert ff2.mag.shape == (grid.N // 2 + 1,)
    assert ff2.phase.shape == (grid.N // 2 + 1,)


def test_current_profile_enforces_nonnegativity_and_normalization(grid):
    x = np.sin(2 * np.pi * grid.t / (50e-15))  # oscillating density, not normalized
    p = CurrentProfile(grid, values=x, charge=None)
    assert np.all(p.values >= 0), "Density should be non-negative"
    assert_allclose(np.trapezoid(p.values, grid.t), 1.0, rtol=0, atol=1e-12)


def test_formfactor_raise_value_error(grid):
    with pytest.raises(ValueError):
        CurrentProfile(grid, values=np.array([-1.0, 0.0, 1.0]), charge=None)
    with pytest.raises(ValueError):
        CurrentProfile(grid, values=np.zeros_like(grid.t), charge=-1.0)


def test_current_profile_charge_normalization(grid, gaussian_density):
    p = CurrentProfile(grid, values=gaussian_density, charge=None)
    with pytest.raises(ValueError):
        p.current
    p.charge = 1.0

    assert np.isclose(np.sum(p.current * p.grid.dt), 1.0, rtol=0, atol=1e-12)


def test_profile_copy(grid, gaussian_density):

    p1 = CurrentProfile(grid, values=gaussian_density, charge=1.0)
    p2 = p1.copy()
    assert np.all(p1.values == p2.values)
    assert np.isclose(p1.charge, p2.charge, rtol=0, atol=1e-12)
    assert p1.grid.N == p2.grid.N




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