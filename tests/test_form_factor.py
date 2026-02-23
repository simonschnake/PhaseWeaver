import numpy as np
from phase_weaver.core.current_profile import CurrentProfile
from phase_weaver.core.form_factor import FormFactor
from phase_weaver.core.transform import DCPhysicalRFFT

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