import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from phase_weaver.core.utils import (
    fwhm_highest_peak,
    overlap_weights,
    safe_real,
    smooth_overlap,
)


def test_safe_real_returns_float_for_real_input():
    x = np.array([1, 2, 3], dtype=np.int32)
    y = safe_real(x)

    assert np.array_equal(y, np.array([1.0, 2.0, 3.0]))
    assert y.dtype == float
    assert not np.iscomplexobj(y)


def test_safe_real_extracts_real_part_for_complex_input():
    x = np.array([1 + 2j, 3 - 4j], dtype=np.complex128)
    y = safe_real(x)

    assert np.array_equal(y, np.array([1.0, 3.0]))
    assert y.dtype == float
    assert not np.iscomplexobj(y)


def test_safe_real_handles_complex_leakage_tiny_imag():
    x = np.array([1.5 + 1e-12j, -2.0 - 1e-15j], dtype=np.complex128)
    y = safe_real(x)

    assert np.allclose(y, np.array([1.5, -2.0]))
    assert y.dtype == float


def test_safe_real_converts_python_list_to_float_ndarray():
    x = [1, 2.25, 3]
    y = safe_real(x)

    assert isinstance(y, np.ndarray)
    assert np.array_equal(y, np.array([1.0, 2.25, 3.0]))
    assert y.dtype == float


def test_safe_real_preserves_shape():
    x = np.array([[1 + 0j, 2 + 3j], [4 - 5j, 6 + 0j]])
    y = safe_real(x)

    assert y.shape == x.shape
    assert np.array_equal(y, np.array([[1.0, 2.0], [4.0, 6.0]]))


def test_safe_real_works_for_scalar_inputs():
    y1 = safe_real(3)
    y2 = safe_real(3 + 4j)

    assert np.array_equal(y1, np.array(3.0))
    assert np.array_equal(y2, np.array(3.0))
    assert y1.dtype == float
    assert y2.dtype == float


def test_exact_fwhm_on_symmetric_peak():
    x = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
    y = np.array([0, 0.5, 1, 2, 1, 0.5, 0], dtype=float)

    res = fwhm_highest_peak(x, y)
    assert res.fwhm == pytest.approx(2.0)
    assert res.left_half_max == pytest.approx(-1.0)
    assert res.right_half_max == pytest.approx(1.0)
    assert res.half_max_value == pytest.approx(1.0)


def test_uses_highest_peak_when_multiple_peaks():
    x = np.arange(-10, 11, dtype=float)
    y = np.zeros_like(x)

    # highest peak centered at x = -4, height = 2
    y[x == -5] = 1.0
    y[x == -4] = 2.0
    y[x == -3] = 1.0

    # smaller peak centered at x = 6, height = 1.5
    y[x == 5] = 0.5
    y[x == 6] = 1.5
    y[x == 7] = 0.5

    res = fwhm_highest_peak(x, y)

    assert res.fwhm == pytest.approx(2.0)
    assert res.left_half_max == pytest.approx(-5.0)
    assert res.right_half_max == pytest.approx(-3.0)
    assert res.half_max_value == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [0.5, 2.0, 10.0])
def test_consistency_under_positive_y_scaling(scale):
    x = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
    y = np.array([0, 0.5, 1, 2, 1, 0.5, 0], dtype=float)

    res1 = fwhm_highest_peak(x, y)
    res2 = fwhm_highest_peak(x, y * scale)
    assert res2.fwhm == pytest.approx(res1.fwhm)
    assert res2.left_half_max == pytest.approx(res1.left_half_max)
    assert res2.right_half_max == pytest.approx(res1.right_half_max)
    assert res2.half_max_value == pytest.approx(res1.half_max_value * scale)


@pytest.mark.parametrize("shift", [-10.0, 3.5, 100.0])
def test_consistency_under_x_translation(shift):
    x = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
    y = np.array([0, 0.5, 1, 2, 1, 0.5, 0], dtype=float)

    res1 = fwhm_highest_peak(x, y)
    res2 = fwhm_highest_peak(x + shift, y)

    assert res2.fwhm == pytest.approx(res1.fwhm)
    assert res2.left_half_max == pytest.approx(res1.left_half_max + shift)
    assert res2.right_half_max == pytest.approx(res1.right_half_max + shift)
    assert res2.half_max_value == pytest.approx(res1.half_max_value)


@pytest.mark.parametrize("scale", [0.5, 2.0, 3.0])
def test_consistency_under_positive_x_scaling(scale):
    x = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
    y = np.array([0, 0.5, 1, 2, 1, 0.5, 0], dtype=float)

    res1 = fwhm_highest_peak(x, y)
    res2 = fwhm_highest_peak(x * scale, y)
    assert res2.fwhm == pytest.approx(res1.fwhm * scale)
    assert res2.left_half_max == pytest.approx(res1.left_half_max * scale)
    assert res2.right_half_max == pytest.approx(res1.right_half_max * scale)
    assert res2.half_max_value == pytest.approx(res1.half_max_value)


def test_raises_when_no_left_crossing_exists():
    x = np.array([0, 1, 2, 3], dtype=float)
    y = np.array([5, 4, 2, 0], dtype=float)  # peak is at first point

    with pytest.raises(ValueError, match="No left half-maximum crossing found."):
        fwhm_highest_peak(x, y)


def test_raises_when_no_right_crossing_exists():
    x = np.array([0, 1, 2, 3], dtype=float)
    y = np.array([0, 2, 4, 5], dtype=float)  # peak is at last point

    with pytest.raises(ValueError, match="No right half-maximum crossing found."):
        fwhm_highest_peak(x, y)


def test_overlap_weights_raises_for_non_1d_x_target():
    with pytest.raises(ValueError, match="x_target must be 1D"):
        overlap_weights(np.array([[0.0, 1.0, 2.0]]))


def test_overlap_weights_empty_returns_empty_array():
    w = overlap_weights(np.array([], dtype=float))

    assert isinstance(w, np.ndarray)
    assert w.shape == (0,)
    assert w.dtype == float


def test_overlap_weights_raises_for_non_finite_x_target():
    with pytest.raises(ValueError, match="x_target must contain only finite values"):
        overlap_weights(np.array([0.0, np.nan, 2.0]))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"transition_width_left": -1.0},
            "transition_width_left must be non-negative",
        ),
        (
            {"transition_width_right": -1.0},
            "transition_width_right must be non-negative",
        ),
        (
            {"power": np.nan},
            "power must be positive",
        ),
        (
            {"power": 0.0},
            "power must be positive",
        ),
        (
            {"power": -1.0},
            "power must be positive",
        ),
    ],
)
def test_overlap_weights_validation_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        overlap_weights(np.array([0.0, 1.0, 2.0]), **kwargs)


def test_overlap_weights_raises_for_unsorted_x_target():
    with pytest.raises(
        ValueError, match="x_target must be sorted in non-decreasing order"
    ):
        overlap_weights(np.array([0.0, 2.0, 1.0]))


def test_overlap_weights_single_point_no_transitions_returns_one():
    w = overlap_weights(
        np.array([1.0]),
        transition_width_left=0.0,
        transition_width_right=0.0,
        power=2.0,
    )

    assert_allclose(w, np.array([1.0]))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    ],
)
def test_overlap_weights_single_point_with_any_transition_returns_zero(left, right):
    w = overlap_weights(
        np.array([1.0]),
        transition_width_left=left,
        transition_width_right=right,
        power=2.0,
    )

    assert_allclose(w, np.array([0.0]))


def test_overlap_weights_no_transitions_returns_all_ones():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    w = overlap_weights(
        x,
        transition_width_left=0.0,
        transition_width_right=0.0,
        power=2.0,
    )

    assert_allclose(w, np.ones_like(x))


def test_overlap_weights_left_transition_only():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    w = overlap_weights(
        x,
        transition_width_left=2.0,
        transition_width_right=0.0,
        power=2.0,
    )

    # smoothstep([0, 0.5, 1, 1.5, 2]) clipped -> [0, 0.5, 1, 1, 1]
    # then squared -> [0, 0.25, 1, 1, 1]
    expected = np.array([0.0, 0.25, 1.0, 1.0, 1.0])
    assert_allclose(w, expected, atol=1e-12)


def test_overlap_weights_right_transition_only():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    w = overlap_weights(
        x,
        transition_width_left=0.0,
        transition_width_right=2.0,
        power=2.0,
    )

    expected = np.array([1.0, 1.0, 1.0, 0.25, 0.0])
    assert_allclose(w, expected, atol=1e-12)


def test_overlap_weights_both_transitions_with_plateau():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    w = overlap_weights(
        x,
        transition_width_left=2.0,
        transition_width_right=2.0,
        power=2.0,
    )

    expected = np.array([0.0, 0.25, 1.0, 0.25, 0.0])
    assert_allclose(w, expected, atol=1e-12)


def test_overlap_weights_both_transitions_overlap_without_plateau():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    w = overlap_weights(
        x,
        transition_width_left=4.0,
        transition_width_right=4.0,
        power=2.0,
    )

    # smoothstep(0.25)=0.15625, squared=0.0244140625
    expected = np.array([0.0, 0.0244140625, 0.25, 0.0244140625, 0.0])
    assert_allclose(w, expected, atol=1e-12)


def test_overlap_weights_power_changes_shape():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    w_power_1 = overlap_weights(
        x,
        transition_width_left=2.0,
        transition_width_right=0.0,
        power=1.0,
    )
    w_power_2 = overlap_weights(
        x,
        transition_width_left=2.0,
        transition_width_right=0.0,
        power=2.0,
    )

    assert_allclose(w_power_1, np.array([0.0, 0.5, 1.0, 1.0, 1.0]), atol=1e-12)
    assert_allclose(w_power_2, np.array([0.0, 0.25, 1.0, 1.0, 1.0]), atol=1e-12)
    assert w_power_2[1] < w_power_1[1]


def test_overlap_weights_all_values_stay_in_unit_interval():
    x = np.linspace(0.0, 10.0, 101)
    w = overlap_weights(
        x,
        transition_width_left=3.0,
        transition_width_right=4.0,
        power=2.5,
    )

    assert np.all(w >= 0.0)
    assert np.all(w <= 1.0)


def test_smooth_overlap_raises_for_non_1d_x_target():
    with pytest.raises(ValueError, match="x_target and x_source must be 1D"):
        smooth_overlap(
            x_target=np.array([[0.0, 1.0, 2.0]]),
            y_target=np.array([1.0, 2.0, 3.0]),
            x_source=np.array([0.0, 1.0, 2.0]),
            y_source=np.array([10.0, 20.0, 30.0]),
        )


def test_smooth_overlap_raises_for_non_1d_x_source():
    with pytest.raises(ValueError, match="x_target and x_source must be 1D"):
        smooth_overlap(
            x_target=np.array([0.0, 1.0, 2.0]),
            y_target=np.array([1.0, 2.0, 3.0]),
            x_source=np.array([[0.0, 1.0, 2.0]]),
            y_source=np.array([10.0, 20.0, 30.0]),
        )


def test_smooth_overlap_raises_for_length_mismatch_target():
    with pytest.raises(ValueError, match="x/y lengths must match"):
        smooth_overlap(
            x_target=np.array([0.0, 1.0, 2.0]),
            y_target=np.array([1.0, 2.0]),
            x_source=np.array([0.0, 1.0]),
            y_source=np.array([10.0, 20.0]),
        )


def test_smooth_overlap_raises_for_length_mismatch_source():
    with pytest.raises(ValueError, match="x/y lengths must match"):
        smooth_overlap(
            x_target=np.array([0.0, 1.0, 2.0]),
            y_target=np.array([1.0, 2.0, 3.0]),
            x_source=np.array([0.0, 1.0]),
            y_source=np.array([10.0]),
        )


def test_smooth_overlap_raises_for_empty_target():
    with pytest.raises(ValueError, match="x_target and y_target must not be empty"):
        smooth_overlap(
            x_target=np.array([], dtype=float),
            y_target=np.array([], dtype=float),
            x_source=np.array([0.0, 1.0]),
            y_source=np.array([10.0, 20.0]),
        )


def test_smooth_overlap_raises_for_empty_source():
    with pytest.raises(ValueError, match="x_source and y_source must not be empty"):
        smooth_overlap(
            x_target=np.array([0.0, 1.0], dtype=float),
            y_target=np.array([1.0, 2.0], dtype=float),
            x_source=np.array([], dtype=float),
            y_source=np.array([], dtype=float),
        )


def test_smooth_overlap_raises_for_non_finite_target_values():
    with pytest.raises(
        ValueError, match="x_target and y_target must contain only finite values"
    ):
        smooth_overlap(
            x_target=np.array([0.0, np.nan, 2.0]),
            y_target=np.array([1.0, 2.0, 3.0]),
            x_source=np.array([0.0, 1.0, 2.0]),
            y_source=np.array([10.0, 20.0, 30.0]),
        )


def test_smooth_overlap_raises_for_non_finite_source_values():
    with pytest.raises(
        ValueError, match="x_source and y_source must contain only finite values"
    ):
        smooth_overlap(
            x_target=np.array([0.0, 1.0, 2.0]),
            y_target=np.array([1.0, 2.0, 3.0]),
            x_source=np.array([0.0, 1.0, 2.0]),
            y_source=np.array([10.0, np.inf, 30.0]),
        )


def test_smooth_overlap_raises_for_unsorted_x_target():
    with pytest.raises(
        ValueError, match="x_target must be sorted in non-decreasing order"
    ):
        smooth_overlap(
            x_target=np.array([0.0, 2.0, 1.0]),
            y_target=np.array([1.0, 2.0, 3.0]),
            x_source=np.array([0.0, 1.0, 2.0]),
            y_source=np.array([10.0, 20.0, 30.0]),
        )


def test_smooth_overlap_raises_for_negative_transition_width():
    with pytest.raises(ValueError, match="transition_width must be non-negative"):
        smooth_overlap(
            x_target=np.array([0.0, 1.0, 2.0]),
            y_target=np.array([1.0, 2.0, 3.0]),
            x_source=np.array([0.0, 1.0, 2.0]),
            y_source=np.array([10.0, 20.0, 30.0]),
            transition_width=-1.0,
        )


@pytest.mark.parametrize("power", [np.nan, 0.0, -1.0])
def test_smooth_overlap_raises_for_invalid_power(power):
    with pytest.raises(ValueError, match="power must be positive"):
        smooth_overlap(
            x_target=np.array([0.0, 1.0, 2.0]),
            y_target=np.array([1.0, 2.0, 3.0]),
            x_source=np.array([0.0, 1.0, 2.0]),
            y_source=np.array([10.0, 20.0, 30.0]),
            power=power,
        )


def test_smooth_overlap_no_overlap_returns_copy_and_preserves_values():
    x_target = np.array([0.0, 1.0, 2.0, 3.0])
    y_target = np.array([10.0, 20.0, 30.0, 40.0])
    x_source = np.array([10.0, 11.0, 12.0])
    y_source = np.array([1.0, 2.0, 3.0])

    y_out = smooth_overlap(
        x_target=x_target,
        y_target=y_target,
        x_source=x_source,
        y_source=y_source,
        transition_width=1.0,
        power=2.0,
    )

    assert_array_equal(y_out, y_target)
    assert y_out is not y_target


def test_smooth_overlap_full_coverage_of_target_replaces_everywhere():
    x_target = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_target = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    x_source = np.array([0.0, 2.0, 4.0])
    y_source = np.array([0.0, 20.0, 40.0])

    y_out = smooth_overlap(
        x_target=x_target,
        y_target=y_target,
        x_source=x_source,
        y_source=y_source,
        transition_width=1.5,
        power=2.0,
    )

    expected = np.interp(x_target, x_source, y_source)
    assert_allclose(y_out, expected, atol=1e-12)


def test_smooth_overlap_beginning_overlap_uses_right_transition_only():
    x_target = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_target = np.zeros_like(x_target)
    x_source = np.array([0.0, 1.0, 2.0])
    y_source = np.array([10.0, 20.0, 30.0])

    y_out = smooth_overlap(
        x_target=x_target,
        y_target=y_target,
        x_source=x_source,
        y_source=y_source,
        transition_width=2.0,
        power=2.0,
    )

    overlap = (x_target >= x_source[0]) & (x_target <= x_source[-1])
    x_overlap = x_target[overlap]
    y_interp = np.interp(x_overlap, x_source, y_source)
    w = overlap_weights(
        x=x_overlap,
        transition_width_left=0.0,
        transition_width_right=2.0,
        power=2.0,
    )

    expected = y_target.copy()
    expected[overlap] = (1.0 - w) * y_target[overlap] + w * y_interp

    assert_allclose(y_out, expected, atol=1e-12)
    assert_array_equal(
        y_out[x_target > x_source[-1]], y_target[x_target > x_source[-1]]
    )


def test_smooth_overlap_end_overlap_uses_left_transition_only():
    x_target = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_target = np.zeros_like(x_target)
    x_source = np.array([2.0, 3.0, 4.0])
    y_source = np.array([10.0, 20.0, 30.0])

    y_out = smooth_overlap(
        x_target=x_target,
        y_target=y_target,
        x_source=x_source,
        y_source=y_source,
        transition_width=2.0,
        power=2.0,
    )

    overlap = (x_target >= x_source[0]) & (x_target <= x_source[-1])
    x_overlap = x_target[overlap]
    y_interp = np.interp(x_overlap, x_source, y_source)
    w = overlap_weights(
        x=x_overlap,
        transition_width_left=2.0,
        transition_width_right=0.0,
        power=2.0,
    )

    expected = y_target.copy()
    expected[overlap] = (1.0 - w) * y_target[overlap] + w * y_interp

    assert_allclose(y_out, expected, atol=1e-12)
    assert_array_equal(y_out[x_target < x_source[0]], y_target[x_target < x_source[0]])


def test_smooth_overlap_middle_overlap_uses_both_transitions():
    x_target = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_target = np.zeros_like(x_target)
    x_source = np.array([1.0, 3.0, 5.0])
    y_source = np.array([10.0, 30.0, 50.0])

    y_out = smooth_overlap(
        x_target=x_target,
        y_target=y_target,
        x_source=x_source,
        y_source=y_source,
        transition_width=2.0,
        power=2.0,
    )

    overlap = (x_target >= x_source[0]) & (x_target <= x_source[-1])
    x_overlap = x_target[overlap]
    y_interp = np.interp(x_overlap, x_source, y_source)
    w = overlap_weights(
        x=x_overlap,
        transition_width_left=2.0,
        transition_width_right=2.0,
        power=2.0,
    )

    expected = y_target.copy()
    expected[overlap] = (1.0 - w) * y_target[overlap] + w * y_interp

    assert_allclose(y_out, expected, atol=1e-12)
    assert_array_equal(y_out[x_target < x_source[0]], y_target[x_target < x_source[0]])
    assert_array_equal(
        y_out[x_target > x_source[-1]], y_target[x_target > x_source[-1]]
    )


def test_smooth_overlap_none_transition_width_uses_half_target_span():
    x_target = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_target = np.zeros_like(x_target)
    x_source = np.array([1.0, 2.0, 3.0])
    y_source = np.array([10.0, 20.0, 30.0])

    y_out = smooth_overlap(
        x_target=x_target,
        y_target=y_target,
        x_source=x_source,
        y_source=y_source,
        transition_width=None,
        power=2.0,
    )

    default_width = 0.5 * (x_target[-1] - x_target[0])

    overlap = (x_target >= x_source[0]) & (x_target <= x_source[-1])
    x_overlap = x_target[overlap]
    y_interp = np.interp(x_overlap, x_source, y_source)
    w = overlap_weights(
        x=x_overlap,
        transition_width_left=default_width,
        transition_width_right=default_width,
        power=2.0,
    )

    expected = y_target.copy()
    expected[overlap] = (1.0 - w) * y_target[overlap] + w * y_interp

    assert_allclose(y_out, expected, atol=1e-12)


def test_smooth_overlap_unsorted_source_matches_sorted_source_case():
    x_target = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_target = np.zeros_like(x_target)

    x_source_sorted = np.array([1.0, 2.0, 3.0])
    y_source_sorted = np.array([10.0, 20.0, 30.0])

    x_source_unsorted = np.array([3.0, 1.0, 2.0])
    y_source_unsorted = np.array([30.0, 10.0, 20.0])

    out_sorted = smooth_overlap(
        x_target=x_target,
        y_target=y_target,
        x_source=x_source_sorted,
        y_source=y_source_sorted,
        transition_width=1.0,
        power=2.0,
    )
    out_unsorted = smooth_overlap(
        x_target=x_target,
        y_target=y_target,
        x_source=x_source_unsorted,
        y_source=y_source_unsorted,
        transition_width=1.0,
        power=2.0,
    )

    assert_allclose(out_unsorted, out_sorted, atol=1e-12)


def test_smooth_overlap_does_not_modify_inputs():
    x_target = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_target = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    x_source = np.array([1.0, 2.0, 3.0])
    y_source = np.array([10.0, 20.0, 30.0])

    x_target_before = x_target.copy()
    y_target_before = y_target.copy()
    x_source_before = x_source.copy()
    y_source_before = y_source.copy()

    _ = smooth_overlap(
        x_target=x_target,
        y_target=y_target,
        x_source=x_source,
        y_source=y_source,
        transition_width=1.0,
        power=2.0,
    )

    assert_array_equal(x_target, x_target_before)
    assert_array_equal(y_target, y_target_before)
    assert_array_equal(x_source, x_source_before)
    assert_array_equal(y_source, y_source_before)
