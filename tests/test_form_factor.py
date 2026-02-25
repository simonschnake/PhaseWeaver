import numpy as np
from phase_weaver.core import CurrentProfile, FormFactor, DCPhysicalRFFT
from phase_weaver.core.constraints import ClampMagnitude

def test_formfactor_clamps_magnitude(grid):
    mag = np.array([0.0, 1.0, 2.0])
    phase = np.zeros_like(mag)
    ff = FormFactor(grid=grid, mag=mag, phase=phase, eps=1e-6)
    assert ff.mag[0] >= 1e-6


def test_formfactor_shapes_match_grid(grid, gaussian_density):
    prof = CurrentProfile(grid, gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=False)
    g2, mag, phase, eps = tr.profile_to_form_factor(prof)
    assert g2.N == grid.N
    assert mag.shape == (grid.N // 2 + 1,)
    assert phase.shape == (grid.N // 2 + 1,)
    assert eps > 0

def test_formfactor_apply_constraint(grid, gaussian_density):
    prof = CurrentProfile(grid, gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=False)
    ff = FormFactor.from_profile(prof, transform=tr)
    ff2 = FormFactor(grid=ff.grid, mag=ff.mag.copy(), phase=ff.phase.copy(), eps=ff.eps, frequency_constraint=ClampMagnitude(eps=0.5))
    assert np.all(ff2.mag >= 0.5)


def test_formfactor_from_form_factor_smoke(grid, gaussian_density):
    prof = CurrentProfile(grid, gaussian_density)
    tr = DCPhysicalRFFT(dc_normalize=False, unwrap_phase=False)
    ff = FormFactor.from_profile(prof, transform=tr)
    ff2 = FormFactor.from_form_factor(ff)
    assert ff2.grid.N == grid.N
    assert ff2.mag.shape == (grid.N // 2 + 1,)
    assert ff2.phase.shape == (grid.N // 2 + 1,)
    assert ff2.eps == ff.eps