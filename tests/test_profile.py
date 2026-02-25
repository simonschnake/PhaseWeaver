import numpy as np
from numpy.testing import assert_allclose
import pytest
from phase_weaver.core import CurrentProfile
from phase_weaver.core.base import Profile

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

def test_profile_from_profile(grid, gaussian_density):
    p1 = CurrentProfile(grid, values=gaussian_density, charge=1.0)
    p2 = CurrentProfile.from_profile(p1)
    p3 = Profile.from_profile(p1)
    assert np.all(p1.values == p2.values)
    assert np.isclose(p1.charge, p2.charge, rtol=0, atol=1e-12)
    assert p1.grid.N == p2.grid.N
    assert np.all(p1.values == p3.values)
    assert np.isclose(p1.charge, p3.charge, rtol=0, atol=1e-12)
    assert p1.grid.N == p3.grid.N