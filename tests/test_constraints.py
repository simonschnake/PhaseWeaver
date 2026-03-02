
import pytest

from phase_weaver.core.constraints import FrequencyConstraint, FrequencyConstraints, NonNegativity, NormalizeArea, TimeConstraint, TimeConstraints, ReplacePhaseEndLinear
from phase_weaver.core.base import FormFactor, Profile, Grid
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


def test_add_non_constraint_raises_not_implemented():
    c1 = NormalizeArea()
    with pytest.raises(NotImplementedError):
        c1 + 5

def test_add_time_constraints():
    c1 = NonNegativity()
    c2 = NormalizeArea()
    combined = c1 + c2
    assert isinstance(combined, TimeConstraints)
    combined2 = combined + c1  # adding another constraint should work
    assert isinstance(combined2, TimeConstraints)
    assert len(combined2.constraints) == 3
    combined3 = combined + combined2  # adding two TimeConstraints should work
    assert isinstance(combined3, TimeConstraints)
    assert len(combined3.constraints) == 5
    with pytest.raises(NotImplementedError):
        combined + 5

def test_add_freq_constraints():
    class DummyFreqConstr(FrequencyConstraint):
        def apply(self, ff):
            return super().apply(ff)  # call the base implementation

    c1 = DummyFreqConstr()
    c2 = DummyFreqConstr()
    with pytest.raises(NotImplementedError):
        c1 + 1
    combined = c1 + c2
    assert isinstance(combined, FrequencyConstraint)
    with pytest.raises(NotImplementedError):
        combined + 5

    combined2 = combined + c1  # adding another constraint should work
    assert isinstance(combined2, FrequencyConstraints)
    assert len(combined2.constraints) == 3
    combined3 = combined + combined2  # adding two FrequencyConstraints should work
    assert isinstance(combined3, FrequencyConstraints)
    assert len(combined3.constraints) == 5

def test_replace_phase_end_linear(grid):
    start_freq = 0.5 * grid.f_pos[-1]  # start at half the max frequency
    alpha = 0.5
    ff = FormFactor(grid=grid, mag=np.ones(grid.N // 2 + 1), phase=np.zeros(grid.N // 2 + 1))
    constraint = ReplacePhaseEndLinear(grid=grid, start_freq=start_freq, alpha=alpha)
    constraint.apply(ff)
    assert np.all(ff.phase[ff.grid.f_pos >= start_freq] == alpha * ff.grid.f_pos[ff.grid.f_pos >= start_freq])

    with pytest.raises(ValueError):
        ReplacePhaseEndLinear(grid=grid, start_freq=-1.0, alpha=alpha)

    with pytest.raises(ValueError):
        ReplacePhaseEndLinear(grid=grid, start_freq=2 * grid.f_pos[-1], alpha=alpha)
