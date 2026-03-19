import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from phase_weaver.core import FormFactor, Grid, Profile
from phase_weaver.core.constraints import (
    BlendMeasuredMagnitude,
    FrequencyConstraint,
    CombinedFrequencyConstraint,
    NonNegativity,
    NormalizeArea,
    ReplacePhaseEndLinear,
    ReplacePhaseEndLinearShift,
    ReplacePhaseEndLinearSmooth,
    TimeConstraint,
    CombinedTimeConstraint,
)
from phase_weaver.core.measurement import MeasuredFormFactor
from phase_weaver.core.utils import smooth_overlap


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
    assert isinstance(combined, CombinedTimeConstraint)
    combined2 = combined + c1  # adding another constraint should work
    assert isinstance(combined2, CombinedTimeConstraint)
    assert len(combined2.constraints) == 3
    combined3 = combined + combined2  # adding two TimeConstraints should work
    assert isinstance(combined3, CombinedTimeConstraint)
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
    assert isinstance(combined2, CombinedFrequencyConstraint)
    assert len(combined2.constraints) == 3
    combined3 = combined + combined2  # adding two FrequencyConstraints should work
    assert isinstance(combined3, CombinedFrequencyConstraint)
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
        power=power,
    )
    constraint.apply(ff)

    mask = grid.f_pos >= start_freq
    expected = smooth_overlap(
        x_target=grid.f_pos,
        y_target=phase,
        x_source=grid.f_pos[mask],
        y_source=(freq_y / freq_x) * grid.f_pos[mask],
        power=power,
        transition_width=None,
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
        power=2.0,
    )
    constraint.apply(ff)

    mask_before = grid.f_pos < start_freq
    assert_array_equal(ff.phase[mask_before], phase_before[mask_before])


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


def test_replace_phase_end_linear_smooth_rejects_freq_x_not_greater_than_start_freq():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    with pytest.raises(ValueError, match="freq_x must be greater than start_freq"):
        ReplacePhaseEndLinearSmooth(
            grid=grid,
            start_freq=50.0,
            freq_x=50.0,
            freq_y=10.0,
        )

    with pytest.raises(ValueError, match="freq_x must be greater than start_freq"):
        ReplacePhaseEndLinearSmooth(
            grid=grid,
            start_freq=50.0,
            freq_x=40.0,
            freq_y=10.0,
        )


def test_replace_phase_end_linear_smooth_rejects_freq_x_above_grid_max():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    with pytest.raises(
        ValueError,
        match="freq_x must be less than or equal to the maximum frequency in the grid",
    ):
        ReplacePhaseEndLinearSmooth(
            grid=grid,
            start_freq=50.0,
            freq_x=120.0,
            freq_y=10.0,
        )


def test_replace_phase_end_linear_smooth_matches_smooth_overlap():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)
    start_freq = 80.0
    freq_x = 100.0
    freq_y = 50.0
    power = 2.5
    transition_width = 7.0

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
        power=power,
        transition_width=transition_width,
    )
    constraint.apply(ff)

    mask = grid.f_pos >= start_freq
    slope = freq_y / freq_x
    expected = smooth_overlap(
        x_target=grid.f_pos,
        y_target=phase,
        x_source=grid.f_pos[mask],
        y_source=slope * grid.f_pos[mask],
        transition_width=transition_width,
        power=power,
    )

    assert_allclose(ff.phase, expected, atol=1e-12)


def test_replace_phase_end_linear_smooth_preserves_phase_before_overlap():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    phase = np.arange(grid.N // 2 + 1, dtype=float)
    ff = FormFactor(
        grid=grid,
        mag=np.ones(grid.N // 2 + 1),
        phase=phase.copy(),
    )

    start_freq = 70.0
    constraint = ReplacePhaseEndLinearSmooth(
        grid=grid,
        start_freq=start_freq,
        freq_x=100.0,
        freq_y=25.0,
        power=2.0,
        transition_width=5.0,
    )
    constraint.apply(ff)

    mask = grid.f_pos < start_freq
    assert_array_equal(ff.phase[mask], phase[mask])


def test_replace_phase_end_linear_smooth_only_changes_phase_not_magnitude():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    mag = np.linspace(1.0, 2.0, grid.N // 2 + 1)
    phase = np.linspace(0.0, 1.0, grid.N // 2 + 1)

    ff = FormFactor(
        grid=grid,
        mag=mag.copy(),
        phase=phase.copy(),
    )

    constraint = ReplacePhaseEndLinearSmooth(
        grid=grid,
        start_freq=60.0,
        freq_x=100.0,
        freq_y=30.0,
        power=2.0,
        transition_width=8.0,
    )
    constraint.apply(ff)

    assert_array_equal(ff.mag, mag)


def test_replace_phase_end_linear_smooth_end_overlap_exact_values():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    phase = np.full(grid.N // 2 + 1, 10.0, dtype=float)
    ff = FormFactor(
        grid=grid,
        mag=np.ones(grid.N // 2 + 1),
        phase=phase.copy(),
    )

    constraint = ReplacePhaseEndLinearSmooth(
        grid=grid,
        start_freq=98.0,
        freq_x=100.0,
        freq_y=50.0,
        power=2.0,
        transition_width=2.0,
    )
    constraint.apply(ff)

    expected = phase.copy()
    # source values: 49.0, 49.5, 50.0
    # overlap_weights on [98, 99, 100] with left=2, right=0 and power=2:
    # [0.0, 0.25, 1.0]
    expected[98] = 10.0
    expected[99] = 0.75 * 10.0 + 0.25 * 49.5
    expected[100] = 50.0

    assert_allclose(ff.phase, expected, atol=1e-12)


def test_replace_phase_end_linear_smooth_none_transition_width_matches_helper_default():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    phase = np.arange(grid.N // 2 + 1, dtype=float)
    ff = FormFactor(
        grid=grid,
        mag=np.ones(grid.N // 2 + 1),
        phase=phase.copy(),
    )

    start_freq = 50.0
    freq_x = 100.0
    freq_y = 40.0

    constraint = ReplacePhaseEndLinearSmooth(
        grid=grid,
        start_freq=start_freq,
        freq_x=freq_x,
        freq_y=freq_y,
        power=2.0,
        transition_width=None,
    )
    constraint.apply(ff)

    mask = grid.f_pos >= start_freq
    slope = freq_y / freq_x
    expected = smooth_overlap(
        x_target=grid.f_pos,
        y_target=phase,
        x_source=grid.f_pos[mask],
        y_source=slope * grid.f_pos[mask],
        transition_width=None,
        power=2.0,
    )

    assert_allclose(ff.phase, expected, atol=1e-12)


def test_replace_phase_end_linear_smooth_accepts_equal_frequency_grid_values_from_other_instance():
    grid_constraint = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)
    grid_same_values = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    constraint = ReplacePhaseEndLinearSmooth(
        grid=grid_constraint,
        start_freq=50.0,
        freq_x=100.0,
        freq_y=50.0,
        power=2.0,
        transition_width=5.0,
    )

    phase = np.arange(grid_same_values.N // 2 + 1, dtype=float)
    ff = FormFactor(
        grid=grid_same_values,
        mag=np.ones(grid_same_values.N // 2 + 1),
        phase=phase.copy(),
    )

    constraint.apply(ff)

    assert ff.phase.shape == phase.shape


@pytest.mark.parametrize("power", [np.nan, 0.0, -1.0])
def test_blend_measured_magnitude_invalid_power_raises(power):
    measured = MeasuredFormFactor(
        freq=np.array([10.0, 20.0, 30.0]),
        mag=np.array([1.0, 2.0, 3.0]),
    )

    with pytest.raises(ValueError, match="power must be positive"):
        BlendMeasuredMagnitude(
            measured=measured,
            power=power,
            transition_width=1.0,
        )


@pytest.mark.parametrize("transition_width", [np.nan, -1.0])
def test_blend_measured_magnitude_invalid_transition_width_raises(transition_width):
    measured = MeasuredFormFactor(
        freq=np.array([10.0, 20.0, 30.0]),
        mag=np.array([1.0, 2.0, 3.0]),
    )

    with pytest.raises(
        ValueError, match="transition_width must be finite and non-negative"
    ):
        BlendMeasuredMagnitude(
            measured=measured,
            power=2.0,
            transition_width=transition_width,
        )


def test_blend_measured_magnitude_no_overlap_leaves_formfactor_unchanged():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    mag = np.arange(grid.N // 2 + 1, dtype=float)
    phase = np.linspace(0.0, 1.0, grid.N // 2 + 1)

    ff = FormFactor(
        grid=grid,
        mag=mag.copy(),
        phase=phase.copy(),
    )

    measured = MeasuredFormFactor(
        freq=np.array([150.0, 175.0, 200.0]),
        mag=np.array([10.0, 20.0, 30.0]),
    )

    constraint = BlendMeasuredMagnitude(
        measured=measured,
        power=2.0,
        transition_width=5.0,
    )
    constraint.apply(ff)

    assert_allclose(ff.mag, mag, atol=1e-12)
    assert_array_equal(ff.phase, phase)


def test_blend_measured_magnitude_full_overlap_replaces_everywhere():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    mag = np.full(grid.N // 2 + 1, -1.0, dtype=float)
    phase = np.arange(grid.N // 2 + 1, dtype=float)
    ff = FormFactor(
        grid=grid,
        mag=mag.copy(),
        phase=phase.copy(),
    )

    measured = MeasuredFormFactor(
        freq=np.array([0.0, 50.0, 100.0]),
        mag=np.array([0.0, 25.0, 50.0]),
    )

    constraint = BlendMeasuredMagnitude(
        measured=measured,
        power=2.0,
        transition_width=7.0,
    )
    constraint.apply(ff)

    expected_mag = np.interp(grid.f_pos, measured.freq, measured.mag)

    assert_allclose(ff.mag, expected_mag, atol=1e-12)
    assert_array_equal(ff.phase, phase)


def test_blend_measured_magnitude_beginning_overlap_matches_smooth_overlap():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    mag = np.full(grid.N // 2 + 1, 10.0, dtype=float)
    phase = np.arange(grid.N // 2 + 1, dtype=float)
    ff = FormFactor(
        grid=grid,
        mag=mag.copy(),
        phase=phase.copy(),
    )

    measured = MeasuredFormFactor(
        freq=np.array([0.0, 20.0, 50.0]),
        mag=np.array([100.0, 80.0, 60.0]),
    )

    constraint = BlendMeasuredMagnitude(
        measured=measured,
        power=2.0,
        transition_width=8.0,
    )
    constraint.apply(ff)

    expected = smooth_overlap(
        x_target=grid.f_pos,
        y_target=mag,
        x_source=measured.freq,
        y_source=measured.mag,
        transition_width=8.0,
        power=2.0,
    )

    assert_allclose(ff.mag, expected, atol=1e-12)
    assert_array_equal(ff.phase, phase)
    assert_array_equal(
        ff.mag[grid.f_pos > measured.freq[-1]],
        mag[grid.f_pos > measured.freq[-1]],
    )


def test_blend_measured_magnitude_middle_overlap_matches_smooth_overlap():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    mag = np.full(grid.N // 2 + 1, 5.0, dtype=float)
    phase = np.arange(grid.N // 2 + 1, dtype=float)
    ff = FormFactor(
        grid=grid,
        mag=mag.copy(),
        phase=phase.copy(),
    )

    measured = MeasuredFormFactor(
        freq=np.array([25.0, 50.0, 75.0]),
        mag=np.array([30.0, 40.0, 35.0]),
    )

    constraint = BlendMeasuredMagnitude(
        measured=measured,
        power=2.0,
        transition_width=10.0,
    )
    constraint.apply(ff)

    expected = smooth_overlap(
        x_target=grid.f_pos,
        y_target=mag,
        x_source=measured.freq,
        y_source=measured.mag,
        transition_width=10.0,
        power=2.0,
    )

    assert_allclose(ff.mag, expected, atol=1e-12)
    assert_array_equal(ff.phase, phase)

    mask_before = grid.f_pos < measured.freq[0]
    mask_after = grid.f_pos > measured.freq[-1]
    assert_array_equal(ff.mag[mask_before], mag[mask_before])
    assert_array_equal(ff.mag[mask_after], mag[mask_after])


def test_blend_measured_magnitude_end_overlap_matches_smooth_overlap():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    mag = np.full(grid.N // 2 + 1, 7.0, dtype=float)
    phase = np.arange(grid.N // 2 + 1, dtype=float)
    ff = FormFactor(
        grid=grid,
        mag=mag.copy(),
        phase=phase.copy(),
    )

    measured = MeasuredFormFactor(
        freq=np.array([50.0, 80.0, 100.0]),
        mag=np.array([10.0, 20.0, 30.0]),
    )

    constraint = BlendMeasuredMagnitude(
        measured=measured,
        power=2.0,
        transition_width=9.0,
    )
    constraint.apply(ff)

    expected = smooth_overlap(
        x_target=grid.f_pos,
        y_target=mag,
        x_source=measured.freq,
        y_source=measured.mag,
        transition_width=9.0,
        power=2.0,
    )

    assert_allclose(ff.mag, expected, atol=1e-12)
    assert_array_equal(ff.phase, phase)
    assert_array_equal(
        ff.mag[grid.f_pos < measured.freq[0]],
        mag[grid.f_pos < measured.freq[0]],
    )


def test_blend_measured_magnitude_none_transition_width_matches_helper_default():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    mag = np.linspace(1.0, 2.0, grid.N // 2 + 1)
    phase = np.linspace(0.0, 1.0, grid.N // 2 + 1)
    ff = FormFactor(
        grid=grid,
        mag=mag.copy(),
        phase=phase.copy(),
    )

    measured = MeasuredFormFactor(
        freq=np.array([20.0, 50.0, 80.0]),
        mag=np.array([100.0, 200.0, 300.0]),
    )

    constraint = BlendMeasuredMagnitude(
        measured=measured,
        power=2.0,
        transition_width=None,
    )
    constraint.apply(ff)

    expected = smooth_overlap(
        x_target=grid.f_pos,
        y_target=mag,
        x_source=measured.freq,
        y_source=measured.mag,
        transition_width=None,
        power=2.0,
    )

    assert_allclose(ff.mag, expected, atol=1e-12)
    assert_array_equal(ff.phase, phase)


def test_blend_measured_magnitude_unsorted_measured_input_behaves_like_sorted_case():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    mag = np.zeros(grid.N // 2 + 1, dtype=float)

    ff_sorted = FormFactor(
        grid=grid,
        mag=mag.copy(),
        phase=np.zeros(grid.N // 2 + 1),
    )
    ff_unsorted = FormFactor(
        grid=grid,
        mag=mag.copy(),
        phase=np.zeros(grid.N // 2 + 1),
    )

    measured_sorted = MeasuredFormFactor(
        freq=np.array([25.0, 50.0, 75.0]),
        mag=np.array([10.0, 20.0, 30.0]),
    )
    measured_unsorted = MeasuredFormFactor(
        freq=np.array([75.0, 25.0, 50.0]),
        mag=np.array([30.0, 10.0, 20.0]),
    )

    BlendMeasuredMagnitude(
        measured=measured_sorted,
        power=2.0,
        transition_width=8.0,
    ).apply(ff_sorted)

    BlendMeasuredMagnitude(
        measured=measured_unsorted,
        power=2.0,
        transition_width=8.0,
    ).apply(ff_unsorted)

    assert_allclose(ff_unsorted.mag, ff_sorted.mag, atol=1e-12)


def test_blend_measured_magnitude_does_not_modify_phase():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    mag = np.arange(grid.N // 2 + 1, dtype=float)
    phase = np.linspace(0.0, 3.0, grid.N // 2 + 1)

    ff = FormFactor(
        grid=grid,
        mag=mag.copy(),
        phase=phase.copy(),
    )

    measured = MeasuredFormFactor(
        freq=np.array([25.0, 50.0, 75.0]),
        mag=np.array([100.0, 200.0, 300.0]),
    )

    BlendMeasuredMagnitude(
        measured=measured,
        power=2.0,
        transition_width=5.0,
    ).apply(ff)

    assert_array_equal(ff.phase, phase)


def test_blend_measured_magnitude_exact_middle_values_for_known_case():
    grid = Grid.from_df_fmax(df=1, f_max=10, snap_pow2=False)

    mag = np.zeros(grid.N // 2 + 1, dtype=float)
    ff = FormFactor(
        grid=grid,
        mag=mag.copy(),
        phase=np.zeros(grid.N // 2 + 1),
    )

    measured = MeasuredFormFactor(
        freq=np.array([2.0, 4.0, 6.0]),
        mag=np.array([20.0, 40.0, 60.0]),
    )

    BlendMeasuredMagnitude(
        measured=measured,
        power=2.0,
        transition_width=2.0,
    ).apply(ff)

    # overlap is x = [2, 3, 4, 5, 6]
    # interpolated source = [20, 30, 40, 50, 60]
    # with both-side transition width 2 and power 2:
    # weights = [0.0, 0.25, 1.0, 0.25, 0.0]
    expected = np.zeros_like(mag)
    expected[2] = 0.0
    expected[3] = 0.25 * 30.0
    expected[4] = 40.0
    expected[5] = 0.25 * 50.0
    expected[6] = 0.0

    assert_allclose(ff.mag, expected, atol=1e-12)


@pytest.mark.parametrize("start_freq", [np.nan, -1.0])
def test_replace_phase_end_linear_smooth_raises_for_invalid_start_freq(start_freq):
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    with pytest.raises(ValueError, match="start_freq must be finite and non-negative"):
        ReplacePhaseEndLinearSmooth(
            grid=grid,
            start_freq=start_freq,
            freq_x=80.0,
            freq_y=10.0,
        )


@pytest.mark.parametrize("freq_x", [np.nan, np.inf, -np.inf])
def test_replace_phase_end_linear_smooth_raises_for_non_finite_freq_x(freq_x):
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    with pytest.raises(ValueError, match="freq_x must be finite"):
        ReplacePhaseEndLinearSmooth(
            grid=grid,
            start_freq=50.0,
            freq_x=freq_x,
            freq_y=10.0,
        )


@pytest.mark.parametrize("freq_y", [np.nan, np.inf, -np.inf])
def test_replace_phase_end_linear_smooth_raises_for_non_finite_freq_y(freq_y):
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    with pytest.raises(ValueError, match="freq_y must be finite"):
        ReplacePhaseEndLinearSmooth(
            grid=grid,
            start_freq=50.0,
            freq_x=80.0,
            freq_y=freq_y,
        )


@pytest.mark.parametrize("power", [np.nan, 0.0, -1.0])
def test_replace_phase_end_linear_smooth_raises_for_invalid_power(power):
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    with pytest.raises(ValueError, match="power must be positive"):
        ReplacePhaseEndLinearSmooth(
            grid=grid,
            start_freq=50.0,
            freq_x=80.0,
            freq_y=10.0,
            power=power,
        )


@pytest.mark.parametrize("transition_width", [np.nan, np.inf, -np.inf, -1.0])
def test_replace_phase_end_linear_smooth_raises_for_invalid_transition_width(
    transition_width,
):
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    with pytest.raises(
        ValueError, match="transition_width must be finite and non-negative"
    ):
        ReplacePhaseEndLinearSmooth(
            grid=grid,
            start_freq=50.0,
            freq_x=80.0,
            freq_y=10.0,
            transition_width=transition_width,
        )


def test_replace_phase_end_linear_smooth_raises_when_freq_x_not_greater_than_start_freq():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    with pytest.raises(ValueError, match="freq_x must be greater than start_freq"):
        ReplacePhaseEndLinearSmooth(
            grid=grid,
            start_freq=50.0,
            freq_x=50.0,
            freq_y=10.0,
        )

    with pytest.raises(ValueError, match="freq_x must be greater than start_freq"):
        ReplacePhaseEndLinearSmooth(
            grid=grid,
            start_freq=50.0,
            freq_x=40.0,
            freq_y=10.0,
        )


def test_replace_phase_end_linear_smooth_raises_when_freq_x_above_grid_max():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)

    with pytest.raises(
        ValueError,
        match="freq_x must be less than or equal to the maximum frequency in the grid",
    ):
        ReplacePhaseEndLinearSmooth(
            grid=grid,
            start_freq=50.0,
            freq_x=120.0,
            freq_y=10.0,
        )


def test_replace_phase_end_linear_smooth_raises_for_mismatched_formfactor_grid():
    grid_constraint = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)
    grid_other = Grid.from_df_fmax(df=2, f_max=100, snap_pow2=False)

    constraint = ReplacePhaseEndLinearSmooth(
        grid=grid_constraint,
        start_freq=50.0,
        freq_x=100.0,
        freq_y=50.0,
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

