
import pytest

from phase_weaver.core.constraints import FrequencyConstraint, NonNegativity, NormalizeArea, TimeConstraint
from phase_weaver.core.base import Profile
import numpy as np

def test_nonnegativity_constraint(grid, gaussian_density):
    prof = Profile(grid, values=gaussian_density, charge=1.0)
    constraint = NonNegativity()
    constraint.apply(prof)
    assert np.all(prof.values >= 0), "Values should be non-negative after applying NonNegativity constraint"


def test_frequency_constraint_is_abstract():
    with pytest.raises(TypeError):
        FrequencyConstraint()  # can't instantiate abstract class

def test_apply_raises_not_implemented_error_when_called_directly():
    # create a concrete subclass dynamically
    class DummyFreqConstr(FrequencyConstraint):
        def apply(self, ff):
            return super().apply(ff)  # call the base implementation

    d = DummyFreqConstr()

    with pytest.raises(NotImplementedError):
        d.apply(ff=None)

    class DummyTimeConstr(TimeConstraint):
        def apply(self, prof):
            return super().apply(prof)  # call the base implementation
        
    dt = DummyTimeConstr()
    with pytest.raises(NotImplementedError):
        dt.apply(prof=None)


def test_normalize_area_constraint_with_zero_area(grid):
    prof = Profile(grid, values=np.zeros_like(grid.t), charge=1.0)
    area_before = np.trapezoid(prof.values, grid.t)
    constraint = NormalizeArea()
    constraint.apply(prof)
    area_after = np.trapezoid(prof.values, grid.t)
    assert area_before == area_after



