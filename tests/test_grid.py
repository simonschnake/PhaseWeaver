# tests/test_grid.py
import numpy as np
import pytest

# Adjust this import to your actual module path
# e.g. from phase_weaver.core.grid import Grid, next_pow2
from phase_weaver.core import Grid
from phase_weaver.core.base import next_pow2


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
