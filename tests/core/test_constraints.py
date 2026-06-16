import numpy as np
import pytest
from numpy.testing import assert_allclose

from phase_weaver.core.base import FormFactor, Grid, Profile
from phase_weaver.core.constraints import (
    BlendMeasuredMagnitude,
    ClampMagnitude,
    CenterFirstMoment,
    CombinedFrequencyConstraint,
    CombinedTimeConstraint,
    CutAfterNthZeroFromPeak,
    EnforceDCOne,
    FrequencyConstraint,
    HighFrequencyMagnitudeDecay,
    NonNegativity,
    NormalizeArea,
    ReplacePhaseEndLinear,
    ReplacePhaseEndLinearShift,
    ReplacePhaseEndLinearSmooth,
    TimeConstraint,
)
from phase_weaver.core.measurement import MeasuredFormFactor
from phase_weaver.core.utils import smooth_overlap


def test_time_and_frequency_constraints_are_abstract():
    with pytest.raises(TypeError):
        TimeConstraint()

    with pytest.raises(TypeError):
        FrequencyConstraint()


def test_time_constraints_can_be_combined_and_applied(grid):
    prof = Profile(grid=grid, values=np.linspace(-1.0, 1.0, grid.N), charge=1.0)

    combined = NonNegativity() + NormalizeArea()
    assert isinstance(combined, CombinedTimeConstraint)

    combined.apply(prof)

    assert np.all(prof.values >= 0)
    assert_allclose(np.trapezoid(prof.values, grid.t), 1.0, atol=1e-12)


def test_normalize_area_leaves_zero_area_profile_unchanged(grid, capsys):
    values = np.zeros(grid.N)
    prof = Profile(grid=grid, values=values.copy())

    NormalizeArea().apply(prof)

    assert_allclose(prof.values, values)
    assert "area is zero" in capsys.readouterr().out


def test_center_first_moment_rolls_profile_to_center():
    grid = Grid(N=8, dt=1.0)
    values = np.zeros(grid.N)
    values[5] = 1.0
    prof = Profile(grid=grid, values=values)

    CenterFirstMoment().apply(prof)

    assert np.argmax(prof.values) == 4


def test_frequency_constraints_can_be_combined_and_applied(grid):
    mag = np.array([0.0, 0.5, 1.0, 2.0, 4.0], dtype=float)
    phase = np.zeros_like(mag)
    ff = FormFactor(grid=grid, mag=mag.copy(), phase=phase.copy())

    combined = ClampMagnitude(eps=0.5) + EnforceDCOne()
    assert isinstance(combined, CombinedFrequencyConstraint)

    combined.apply(ff)

    assert ff.mag[0] == pytest.approx(1.0)
    assert np.all(np.isfinite(ff.mag))


def test_high_frequency_magnitude_decay_tapers_tail_to_zero(grid):
    mag = np.ones(grid.N // 2 + 1, dtype=float)
    phase = np.zeros_like(mag)
    ff = FormFactor(grid=grid, mag=mag.copy(), phase=phase.copy())

    start_freq = 5.0 * grid.df
    end_freq = 8.0 * grid.df
    constraint = HighFrequencyMagnitudeDecay(
        start_freq=start_freq,
        end_freq=end_freq,
    )
    constraint.apply(ff)

    assert_allclose(ff.mag[grid.f_pos <= start_freq], 1.0)
    assert np.all(ff.mag[(grid.f_pos > start_freq) & (grid.f_pos < end_freq)] < 1.0)
    assert_allclose(ff.mag[grid.f_pos >= end_freq], 0.0)


def test_high_frequency_magnitude_decay_validates_inputs():
    with pytest.raises(ValueError, match="start_freq must be finite and non-negative"):
        HighFrequencyMagnitudeDecay(start_freq=-1.0, end_freq=2.0)

    with pytest.raises(ValueError, match="end_freq must be greater than start_freq"):
        HighFrequencyMagnitudeDecay(start_freq=2.0, end_freq=2.0)

    with pytest.raises(ValueError, match="floor must be finite and non-negative"):
        HighFrequencyMagnitudeDecay(start_freq=1.0, end_freq=2.0, floor=-1.0)

    with pytest.raises(ValueError, match="end_freq must be finite"):
        HighFrequencyMagnitudeDecay(start_freq=1.0, end_freq=np.inf)


def test_high_frequency_magnitude_decay_noops_when_start_above_grid(grid):
    mag = np.ones(grid.N // 2 + 1, dtype=float)
    ff = FormFactor(grid=grid, mag=mag.copy(), phase=np.zeros_like(mag))

    HighFrequencyMagnitudeDecay(
        start_freq=grid.f_pos[-1],
        end_freq=grid.f_pos[-1] + grid.df,
    ).apply(ff)

    assert_allclose(ff.mag, mag)


def test_replace_phase_end_linear(grid):
    start_freq = 0.5 * grid.f_pos[-1]
    alpha = 0.5
    ff = FormFactor(
        grid=grid, mag=np.ones(grid.N // 2 + 1), phase=np.zeros(grid.N // 2 + 1)
    )

    constraint = ReplacePhaseEndLinear(grid=grid, start_freq=start_freq, alpha=alpha)
    constraint.apply(ff)

    tail_mask = grid.f_pos >= start_freq
    assert_allclose(ff.phase[tail_mask], alpha * grid.f_pos[tail_mask])
    assert_allclose(ff.phase[~tail_mask], 0.0)


def test_replace_phase_end_linear_validates_inputs(grid):
    with pytest.raises(ValueError, match="start_freq must be finite and non-negative"):
        ReplacePhaseEndLinear(grid=grid, start_freq=np.nan, alpha=1.0)

    with pytest.raises(ValueError, match="maximum frequency"):
        ReplacePhaseEndLinear(grid=grid, start_freq=grid.f_pos[-1] + grid.df, alpha=1.0)


def test_replace_phase_end_linear_shift():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)
    start_freq = 50.0
    freq_x = 100.0
    freq_y = 150.0
    phase = np.zeros(grid.N // 2 + 1, dtype=float)

    ff = FormFactor(grid=grid, mag=np.ones_like(phase), phase=phase.copy())
    constraint = ReplacePhaseEndLinearShift(
        grid=grid, start_freq=start_freq, freq_x=freq_x, freq_y=freq_y
    )
    constraint.apply(ff)

    tail_mask = grid.f_pos >= start_freq
    start_x = grid.f_pos[tail_mask][0]
    expected_tail = ((freq_y - 0.0) / (freq_x - start_x)) * (
        grid.f_pos[tail_mask] - start_x
    )

    assert_allclose(ff.phase[~tail_mask], 0.0)
    assert_allclose(ff.phase[tail_mask], expected_tail)


def test_replace_phase_end_linear_shift_validates_inputs(grid):
    with pytest.raises(ValueError, match="start_freq must be non-negative"):
        ReplacePhaseEndLinearShift(grid=grid, start_freq=-1.0, freq_x=1.0, freq_y=1.0)

    with pytest.raises(ValueError, match="start_freq is above"):
        ReplacePhaseEndLinearShift(
            grid=grid,
            start_freq=grid.f_pos[-1] + grid.df,
            freq_x=grid.f_pos[-1] + 2.0 * grid.df,
            freq_y=1.0,
        )

    with pytest.raises(ValueError, match="freq_x must be greater"):
        ReplacePhaseEndLinearShift(grid=grid, start_freq=1.0, freq_x=1.0, freq_y=1.0)

    with pytest.raises(ValueError, match="maximum frequency"):
        ReplacePhaseEndLinearShift(
            grid=grid,
            start_freq=0.0,
            freq_x=grid.f_pos[-1] + grid.df,
            freq_y=1.0,
        )


def test_replace_phase_end_linear_smooth_matches_helper():
    grid = Grid.from_df_fmax(df=1, f_max=100, snap_pow2=False)
    start_freq = 50.0
    freq_x = 100.0
    freq_y = 50.0
    phase = np.arange(grid.N // 2 + 1, dtype=float)
    phase_before = phase.copy()

    ff = FormFactor(grid=grid, mag=np.ones_like(phase), phase=phase)
    constraint = ReplacePhaseEndLinearSmooth(
        start_freq=start_freq,
        freq_x=freq_x,
        freq_y=freq_y,
        power=2.0,
    )
    constraint.apply(ff)

    replace_mask = grid.f_pos >= start_freq
    expected = smooth_overlap(
        x_target=grid.f_pos,
        y_target=phase_before,
        x_source=grid.f_pos[replace_mask],
        y_source=(freq_y / freq_x) * grid.f_pos[replace_mask],
        power=2.0,
        transition_width=None,
    )
    assert_allclose(ff.phase, expected)


def test_replace_phase_end_linear_smooth_validates_inputs(grid):
    with pytest.raises(ValueError, match="start_freq must be finite and non-negative"):
        ReplacePhaseEndLinearSmooth(start_freq=-1.0, freq_x=10.0, freq_y=5.0)

    with pytest.raises(ValueError, match="freq_x must be finite"):
        ReplacePhaseEndLinearSmooth(start_freq=1.0, freq_x=np.nan, freq_y=5.0)

    with pytest.raises(ValueError, match="freq_y must be finite"):
        ReplacePhaseEndLinearSmooth(start_freq=1.0, freq_x=10.0, freq_y=np.nan)

    with pytest.raises(ValueError, match="power must be positive"):
        ReplacePhaseEndLinearSmooth(start_freq=1.0, freq_x=10.0, freq_y=5.0, power=0.0)

    with pytest.raises(ValueError, match="transition_width must be finite and non-negative"):
        ReplacePhaseEndLinearSmooth(
            start_freq=1.0,
            freq_x=10.0,
            freq_y=5.0,
            transition_width=-1.0,
        )

    with pytest.raises(ValueError, match="freq_x must be greater than start_freq"):
        ReplacePhaseEndLinearSmooth(start_freq=1.0, freq_x=1.0, freq_y=5.0)

    with pytest.raises(
        ValueError,
        match="freq_x must be less than or equal to the maximum frequency in the grid",
    ):
        constraint = ReplacePhaseEndLinearSmooth(
            start_freq=1.0, freq_x=grid.f_pos[-1] * 2.0, freq_y=5.0
        )
        ff = FormFactor(
            grid=grid,
            mag=np.ones(grid.N // 2 + 1),
            phase=np.zeros(grid.N // 2 + 1),
        )
        constraint.apply(ff)


def test_blend_measured_magnitude_matches_helper(grid):
    mag = np.full(grid.N // 2 + 1, 10.0, dtype=float)
    phase = np.zeros_like(mag)
    ff = FormFactor(grid=grid, mag=mag.copy(), phase=phase.copy())

    measured = (
        MeasuredFormFactor(
            freq=np.array([0.0, 50.0, 100.0]),
            mag=np.array([1.0, 2.0, 3.0]),
        ),
    )

    constraint = BlendMeasuredMagnitude(
        measured=measured,
        power=2.0,
        transition_width=5.0,
        scale=True,
    )
    constraint.apply(ff)

    avg_current_mag = 10.0
    avg_measured_mag = measured[0].mag.mean()
    expected = smooth_overlap(
        x_target=grid.f_pos,
        y_target=mag,
        x_source=measured[0].freq,
        y_source=measured[0].mag * avg_current_mag / avg_measured_mag,
        power=2.0,
        transition_width=5.0,
    )
    assert_allclose(ff.mag, expected)


def test_blend_measured_magnitude_without_scaling_uses_raw_measurement(grid):
    mag = np.full(grid.N // 2 + 1, 10.0, dtype=float)
    phase = np.zeros_like(mag)
    ff = FormFactor(grid=grid, mag=mag.copy(), phase=phase.copy())

    measured = (
        MeasuredFormFactor(
            freq=np.array([0.0, 50.0, 100.0]),
            mag=np.array([1.0, 2.0, 3.0]),
        ),
    )

    constraint = BlendMeasuredMagnitude(
        measured=measured,
        power=2.0,
        transition_width=5.0,
        scale=False,
    )
    constraint.apply(ff)

    expected = smooth_overlap(
        x_target=grid.f_pos,
        y_target=mag,
        x_source=measured[0].freq,
        y_source=measured[0].mag,
        power=2.0,
        transition_width=5.0,
    )
    assert_allclose(ff.mag, expected)


def test_blend_measured_magnitude_validates_inputs():
    measured = (MeasuredFormFactor(freq=np.array([0.0, 1.0]), mag=np.ones(2)),)

    with pytest.raises(ValueError, match="power must be positive"):
        BlendMeasuredMagnitude(measured=measured, power=0.0)

    with pytest.raises(ValueError, match="transition_width must be finite and non-negative"):
        BlendMeasuredMagnitude(measured=measured, transition_width=-1.0)


def test_cut_after_nth_zero_from_peak_trims_crossings():
    grid = Grid(N=9, dt=1.0)
    values = np.array([0.0, 1.0, 0.0, 2.0, 5.0, 2.0, 0.0, 1.0, 0.0])
    prof = Profile(grid=grid, values=values.copy())

    CutAfterNthZeroFromPeak(n=1, threshold=0.0).apply(prof)

    assert_allclose(prof.values, [0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0])


def test_cut_after_nth_zero_from_peak_can_keep_crossings():
    grid = Grid(N=9, dt=1.0)
    values = np.array([0.0, 1.0, 0.0, 2.0, 5.0, 2.0, 0.0, 1.0, 0.0])
    prof = Profile(grid=grid, values=values.copy())

    CutAfterNthZeroFromPeak(n=1, threshold=0.0, keep_crossing=True).apply(prof)

    assert_allclose(prof.values, [0.0, 1.0, 0.0, 2.0, 5.0, 2.0, 0.0, 1.0, 0.0])


def test_cut_after_nth_zero_from_peak_validates_inputs(grid):
    with pytest.raises(ValueError, match="n must be a positive integer"):
        CutAfterNthZeroFromPeak(n=0)

    with pytest.raises(ValueError, match="threshold must be finite"):
        CutAfterNthZeroFromPeak(threshold=np.nan)

    constraint = CutAfterNthZeroFromPeak()
    prof = Profile(grid=grid, values=np.ones(grid.N))
    prof.values = np.ones((1, grid.N))
    with pytest.raises(ValueError, match="only supports 1D"):
        constraint.apply(prof)

    empty_prof = Profile(grid=Grid(N=1, dt=1.0), values=np.ones(1))
    empty_prof.values = np.array([])
    constraint.apply(empty_prof)
    assert empty_prof.values.size == 0
