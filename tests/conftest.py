import numpy as np
import pytest
from phase_weaver.core import Grid

@pytest.fixture
def grid():
    # Example: uniform grid centered around 0
    N = 2048
    dt = 1e-15  # 1 fs
    g = Grid(N=N, dt=dt)
    return g

@pytest.fixture
def gaussian_density(grid):
    # normalized, centered gaussian density
    sigma = 30e-15
    x = np.exp(-0.5 * (grid.t / sigma) ** 2)
    x /= np.trapezoid(x, grid.t)
    return x