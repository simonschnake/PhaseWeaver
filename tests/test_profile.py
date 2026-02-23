import numpy as np
import pytest
from numpy.testing import assert_allclose
from phase_weaver.core.current_profile import CurrentProfile

def test_current_profile_enforces_nonnegativity_and_normalization(grid):
    x = np.sin(2 * np.pi * grid.t / (50e-15))  # oscillating density, not normalized
    p = CurrentProfile(grid, values=x, charge=None)
    assert np.all(p.density >= 0), "Density should be non-negative"
    assert_allclose(np.trapezoid(p.density, grid.t), 1.0, rtol=0, atol=1e-12)


def test_current_profile_raises_if_integral_nonpositive(grid):
    x = -np.ones_like(grid.t)  # becomes all zeros after max(x,0)
    with pytest.raises(ValueError):
        CurrentProfile(grid=grid, values=x, charge=None)


def test_currentprofile_rejects_all_zero(grid):
    with pytest.raises(ValueError):
        CurrentProfile(grid, np.zeros(grid.N))