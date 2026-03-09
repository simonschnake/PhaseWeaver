import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from phase_weaver.core import FormFactor, Grid, Profile
from phase_weaver.core.constraints import (
    FrequencyConstraint,
    FrequencyConstraints,
    NonNegativity,
    NormalizeArea,
    ReplacePhaseEndLinear,
    ReplacePhaseEndLinearShift,
    ReplacePhaseEndLinearSmooth,
    TimeConstraint,
    TimeConstraints,
)
from phase_weaver.core.utils import smooth_insert


def test_nonnegativity_constraint(grid, gaussian_density):
    prof = Profile(grid, values=gaussian_density, charge=1.0)
    constraint = NonNegativity()
    constraint.apply(prof)
    assert np.all(prof.values >= 0), (
        "Values should be non-negative after applying NonNegativity constraint"
    )


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
    ff = FormFactor(
        grid=grid, mag=np.ones(grid.N // 2 + 1), phase=np.zeros(grid.N // 2 + 1)
    )
    constraint = ReplacePhaseEndLinear(grid=grid, start_freq=start_freq, alpha=alpha)
    constraint.apply(ff)
    assert np.all(
        ff.phase[ff.grid.f_pos >= start_freq]
        == alpha * ff.grid.f_pos[ff.grid.f_pos >= start_freq]
    )

    with pytest.raises(ValueError):
        ReplacePhaseEndLinear(grid=grid, start_freq=-1.0, alpha=alpha)

    with pytest.raises(ValueError):
        ReplacePhaseEndLinear(grid=grid, start_freq=2 * grid.f_pos[-1], alpha=alpha)


def test_replace_phase_end_linear_shift():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)
    start_freq = 50.0
    freq_x = 100.0
    freq_y = 150.0
    phase = np.arange(101, dtype=float)  # some arbitrary phase profile
    phase_after = np.copy(phase)

    phase_after[phase_after >= 50.0] = 2 * phase_after[phase_after >= 50.0] - 50.0
    ff = FormFactor(grid=grid, mag=np.ones(grid.N // 2 + 1), phase=phase)
    constraint = ReplacePhaseEndLinearShift(
        grid=grid, start_freq=start_freq, freq_x=freq_x, freq_y=freq_y
    )
    constraint.apply(ff)
    assert np.allclose(ff.phase, phase_after, atol=1e-6)

    with pytest.raises(ValueError):
        ReplacePhaseEndLinearShift(
            grid=grid, start_freq=-1.0, freq_x=freq_x, freq_y=freq_y
        )

    with pytest.raises(ValueError):
        ReplacePhaseEndLinearShift(
            grid=grid, start_freq=120.0, freq_x=freq_x, freq_y=freq_y
        )

    with pytest.raises(ValueError):
        ReplacePhaseEndLinearShift(
            grid=grid, start_freq=start_freq, freq_x=40.0, freq_y=freq_y
        )

    with pytest.raises(ValueError):
        ReplacePhaseEndLinearShift(
            grid=grid, start_freq=start_freq, freq_x=120.0, freq_y=freq_y
        )


def test_replace_phase_end_linear_smooth_matches_smooth_insert():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)
    start_freq = 98.0
    freq_x = 100.0
    freq_y = 50.0
    favor = 0.8
    power = 2.0

    phase = np.arange(grid.N // 2 + 1, dtype=float)
    ff = FormFactor(
        grid=grid,
        mag=np.ones(grid.N // 2 + 1),
        phase=phase.copy(),
    )

    constraint = ReplacePhaseEndLinearSmooth(
        grid=grid,
        start_freq=start_freq,
        freq_x=freq_x,
        freq_y=freq_y,
        favor=favor,
        power=power,
    )
    constraint.apply(ff)

    mask = grid.f_pos >= start_freq
    expected = smooth_insert(
        x1=grid.f_pos,
        y1=phase,
        x2=grid.f_pos[mask],
        y2=(freq_y / freq_x) * grid.f_pos[mask],
        favor=favor,
        power=power,
    )

    assert_allclose(ff.phase, expected, atol=1e-12)


def test_replace_phase_end_linear_smooth_preserves_phase_before_start():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)
    start_freq = 50.0
    freq_x = 100.0
    freq_y = 50.0

    phase = np.arange(grid.N // 2 + 1, dtype=float)
    phase_before = phase.copy()

    ff = FormFactor(
        grid=grid,
        mag=np.ones(grid.N // 2 + 1),
        phase=phase,
    )

    constraint = ReplacePhaseEndLinearSmooth(
        grid=grid,
        start_freq=start_freq,
        freq_x=freq_x,
        freq_y=freq_y,
        favor=0.8,
        power=2.0,
    )
    constraint.apply(ff)

    mask_before = grid.f_pos < start_freq
    assert_array_equal(ff.phase[mask_before], phase_before[mask_before])


def test_replace_phase_end_linear_smooth_end_overlap_exact_values():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)
    start_freq = 98.0
    freq_x = 100.0
    freq_y = 50.0
    favor = 1.0
    power = 2.0

    phase = np.full(grid.N // 2 + 1, 10.0, dtype=float)

    ff = FormFactor(
        grid=grid,
        mag=np.ones(grid.N // 2 + 1),
        phase=phase.copy(),
    )

    constraint = ReplacePhaseEndLinearSmooth(
        grid=grid,
        start_freq=start_freq,
        freq_x=freq_x,
        freq_y=freq_y,
        favor=favor,
        power=power,
    )
    constraint.apply(ff)

    expected = phase.copy()
    # overlap points are 98, 99, 100
    # t = [0, 0.5, 1.0]
    # w = favor * t^power = [0.0, 0.25, 1.0]
    expected[98] = 10.0
    expected[99] = (1.0 - 0.25) * 10.0 + 0.25 * ((freq_y / freq_x) * 99.0)
    expected[100] = (freq_y / freq_x) * 100.0

    assert_allclose(ff.phase, expected, atol=1e-12)


def test_replace_phase_end_linear_smooth_favor_zero_leaves_phase_unchanged():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)
    start_freq = 50.0
    freq_x = 100.0
    freq_y = 50.0

    phase = np.arange(grid.N // 2 + 1, dtype=float)
    ff = FormFactor(
        grid=grid,
        mag=np.ones(grid.N // 2 + 1),
        phase=phase.copy(),
    )

    constraint = ReplacePhaseEndLinearSmooth(
        grid=grid,
        start_freq=start_freq,
        freq_x=freq_x,
        freq_y=freq_y,
        favor=0.0,
        power=2.0,
    )
    constraint.apply(ff)

    assert_allclose(ff.phase, phase, atol=1e-12)


def test_replace_phase_end_linear_smooth_reaches_linear_value_at_last_point_for_favor_one():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)
    start_freq = 50.0
    freq_x = 100.0
    freq_y = 50.0

    phase = np.zeros(grid.N // 2 + 1, dtype=float)
    ff = FormFactor(
        grid=grid,
        mag=np.ones(grid.N // 2 + 1),
        phase=phase,
    )

    constraint = ReplacePhaseEndLinearSmooth(
        grid=grid,
        start_freq=start_freq,
        freq_x=freq_x,
        freq_y=freq_y,
        favor=1.0,
        power=2.0,
    )
    constraint.apply(ff)

    assert ff.phase[-1] == pytest.approx((freq_y / freq_x) * grid.f_pos[-1])


def test_replace_phase_end_linear_smooth_invalid_start_freq(grid):
    freq_x = grid.f_pos[-1]
    freq_y = 50.0

    with pytest.raises(ValueError):
        ReplacePhaseEndLinearSmooth(
            grid=grid,
            start_freq=-1.0,
            freq_x=freq_x,
            freq_y=freq_y,
        )

    with pytest.raises(ValueError):
        ReplacePhaseEndLinearSmooth(
            grid=grid,
            start_freq=2 * grid.f_pos[-1],
            freq_x=freq_x,
            freq_y=freq_y,
        )


def test_replace_phase_end_linear_smooth_invalid_freq_x():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)
    start_freq = 50.0
    freq_y = 150.0

    with pytest.raises(ValueError):
        ReplacePhaseEndLinearSmooth(
            grid=grid,
            start_freq=start_freq,
            freq_x=40.0,
            freq_y=freq_y,
        )

    with pytest.raises(ValueError):
        ReplacePhaseEndLinearSmooth(
            grid=grid,
            start_freq=start_freq,
            freq_x=120.0,
            freq_y=freq_y,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            dict(start_freq=np.nan, freq_x=100.0, freq_y=50.0, favor=0.8, power=2.0),
            "start_freq must be finite and non-negative",
        ),
        (
            dict(start_freq=-1.0, freq_x=100.0, freq_y=50.0, favor=0.8, power=2.0),
            "start_freq must be finite and non-negative",
        ),
        (
            dict(start_freq=50.0, freq_x=np.nan, freq_y=50.0, favor=0.8, power=2.0),
            "freq_x must be finite",
        ),
        (
            dict(start_freq=50.0, freq_x=100.0, freq_y=np.nan, favor=0.8, power=2.0),
            "freq_y must be finite",
        ),
        (
            dict(start_freq=50.0, freq_x=100.0, freq_y=50.0, favor=np.nan, power=2.0),
            "favor must be in \\[0, 1\\]",
        ),
        (
            dict(start_freq=50.0, freq_x=100.0, freq_y=50.0, favor=-0.1, power=2.0),
            "favor must be in \\[0, 1\\]",
        ),
        (
            dict(start_freq=50.0, freq_x=100.0, freq_y=50.0, favor=1.1, power=2.0),
            "favor must be in \\[0, 1\\]",
        ),
        (
            dict(start_freq=50.0, freq_x=100.0, freq_y=50.0, favor=0.8, power=np.nan),
            "power must be positive",
        ),
        (
            dict(start_freq=50.0, freq_x=100.0, freq_y=50.0, favor=0.8, power=0.0),
            "power must be positive",
        ),
        (
            dict(start_freq=50.0, freq_x=100.0, freq_y=50.0, favor=0.8, power=-1.0),
            "power must be positive",
        ),
    ],
)
def test_replace_phase_end_linear_smooth_validation_errors(kwargs, match):
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    with pytest.raises(ValueError, match=match):
        ReplacePhaseEndLinearSmooth(grid=grid, **kwargs)


def test_replace_phase_end_linear_smooth_raises_for_mismatched_formfactor_grid():
    grid_constraint = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)
    grid_other = Grid.from_df_fmax(df=2, f_max=100, snap_pow2=False)

    constraint = ReplacePhaseEndLinearSmooth(
        grid=grid_constraint,
        start_freq=50.0,
        freq_x=100.0,
        freq_y=50.0,
        favor=0.8,
        power=2.0,
    )

    ff = FormFactor(
        grid=grid_other,
        mag=np.ones(grid_other.N // 2 + 1),
        phase=np.zeros(grid_other.N // 2 + 1),
    )

    with pytest.raises(
        ValueError, match="FormFactor grid does not match constraint grid"
    ):
        constraint.apply(ff)
