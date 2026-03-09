import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from phase_weaver.core.utils import (
    fwhm_highest_peak,
    safe_real,
    smooth_insert,
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

    width, x_left, x_right, half = fwhm_highest_peak(x, y)

    assert width == pytest.approx(2.0)
    assert x_left == pytest.approx(-1.0)
    assert x_right == pytest.approx(1.0)
    assert half == pytest.approx(1.0)


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

    width, x_left, x_right, half = fwhm_highest_peak(x, y)

    assert width == pytest.approx(2.0)
    assert x_left == pytest.approx(-5.0)
    assert x_right == pytest.approx(-3.0)
    assert half == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [0.5, 2.0, 10.0])
def test_consistency_under_positive_y_scaling(scale):
    x = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
    y = np.array([0, 0.5, 1, 2, 1, 0.5, 0], dtype=float)

    width1, x_left1, x_right1, half1 = fwhm_highest_peak(x, y)
    width2, x_left2, x_right2, half2 = fwhm_highest_peak(x, y * scale)

    assert width2 == pytest.approx(width1)
    assert x_left2 == pytest.approx(x_left1)
    assert x_right2 == pytest.approx(x_right1)
    assert half2 == pytest.approx(half1 * scale)


@pytest.mark.parametrize("shift", [-10.0, 3.5, 100.0])
def test_consistency_under_x_translation(shift):
    x = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
    y = np.array([0, 0.5, 1, 2, 1, 0.5, 0], dtype=float)

    width1, x_left1, x_right1, half1 = fwhm_highest_peak(x, y)
    width2, x_left2, x_right2, half2 = fwhm_highest_peak(x + shift, y)

    assert width2 == pytest.approx(width1)
    assert x_left2 == pytest.approx(x_left1 + shift)
    assert x_right2 == pytest.approx(x_right1 + shift)
    assert half2 == pytest.approx(half1)


@pytest.mark.parametrize("scale", [0.5, 2.0, 3.0])
def test_consistency_under_positive_x_scaling(scale):
    x = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
    y = np.array([0, 0.5, 1, 2, 1, 0.5, 0], dtype=float)

    width1, x_left1, x_right1, half1 = fwhm_highest_peak(x, y)
    width2, x_left2, x_right2, half2 = fwhm_highest_peak(x * scale, y)

    assert width2 == pytest.approx(width1 * scale)
    assert x_left2 == pytest.approx(x_left1 * scale)
    assert x_right2 == pytest.approx(x_right1 * scale)
    assert half2 == pytest.approx(half1)


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


def test_raises_for_non_1d_x1():
    x1 = np.array([[0, 1], [2, 3]])
    y1 = np.array([0, 1])
    x2 = np.array([0, 1])
    y2 = np.array([0, 1])

    with pytest.raises(ValueError, match="x1 and x2 must be 1D"):
        smooth_insert(x1, y1, x2, y2)


def test_raises_for_non_1d_x2():
    x1 = np.array([0, 1, 2])
    y1 = np.array([0, 1, 2])
    x2 = np.array([[0, 1], [2, 3]])
    y2 = np.array([0, 1])

    with pytest.raises(ValueError, match="x1 and x2 must be 1D"):
        smooth_insert(x1, y1, x2, y2)


def test_raises_for_length_mismatch_x1_y1():
    x1 = np.array([0, 1, 2])
    y1 = np.array([0, 1])  # wrong length
    x2 = np.array([0, 1])
    y2 = np.array([0, 1])

    with pytest.raises(ValueError, match="x/y lengths must match"):
        smooth_insert(x1, y1, x2, y2)


def test_raises_for_length_mismatch_x2_y2():
    x1 = np.array([0, 1, 2])
    y1 = np.array([0, 1, 2])
    x2 = np.array([0, 1])
    y2 = np.array([0])  # wrong length

    with pytest.raises(ValueError, match="x/y lengths must match"):
        smooth_insert(x1, y1, x2, y2)


def test_no_overlap_returns_copy_and_preserves_values():
    x1 = np.array([0, 1, 2, 3, 4], dtype=float)
    y1 = np.array([10, 20, 30, 40, 50], dtype=float)
    x2 = np.array([10, 11], dtype=float)
    y2 = np.array([1, 2], dtype=float)

    y_out = smooth_insert(x1, y1, x2, y2)

    assert_array_equal(y_out, y1)
    assert y_out is not y1  # function should return a copy


def test_full_domain_overlap_blends_everywhere():
    x1 = np.array([0, 1, 2], dtype=float)
    y1 = np.array([2, 2, 2], dtype=float)
    x2 = np.array([0, 1, 2], dtype=float)
    y2 = np.array([10, 10, 10], dtype=float)

    y_out = smooth_insert(x1, y1, x2, y2, favor=0.75)

    # (1 - 0.75)*2 + 0.75*10 = 8
    expected = np.array([8, 8, 8], dtype=float)
    assert_allclose(y_out, expected)


def test_beginning_overlap_fades_out_from_y2():
    x1 = np.array([0, 1, 2, 3, 4], dtype=float)
    y1 = np.zeros_like(x1)
    x2 = np.array([0, 1, 2], dtype=float)
    y2 = np.array([10, 10, 10], dtype=float)

    y_out = smooth_insert(x1, y1, x2, y2, favor=0.8, power=2.0)

    # overlap has n=3 -> t = [0, 0.5, 1]
    # w = 0.8 * (1 - t^2) = [0.8, 0.6, 0.0]
    expected = np.array([8.0, 6.0, 0.0, 0.0, 0.0])
    assert_allclose(y_out, expected)


def test_end_overlap_fades_in_to_y2():
    x1 = np.array([0, 1, 2, 3, 4], dtype=float)
    y1 = np.zeros_like(x1)
    x2 = np.array([2, 3, 4], dtype=float)
    y2 = np.array([10, 10, 10], dtype=float)

    y_out = smooth_insert(x1, y1, x2, y2, favor=0.8, power=2.0)

    # overlap has n=3 -> t = [0, 0.5, 1]
    # w = 0.8 * t^2 = [0.0, 0.2, 0.8]
    expected = np.array([0.0, 0.0, 0.0, 2.0, 8.0])
    assert_allclose(y_out, expected)


def test_middle_overlap_fades_in_and_out():
    x1 = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
    y1 = np.zeros_like(x1)
    x2 = np.array([1, 2, 3, 4, 5], dtype=float)
    y2 = np.array([10, 10, 10, 10, 10], dtype=float)

    y_out = smooth_insert(x1, y1, x2, y2, favor=0.8, power=2.0)

    # n=5 -> t = [0, 0.25, 0.5, 0.75, 1]
    # sin(pi*t)^2 = [0, 0.5, 1, 0.5, 0]
    # w = 0.8 * that = [0, 0.4, 0.8, 0.4, 0]
    expected = np.array([0.0, 0.0, 4.0, 8.0, 4.0, 0.0, 0.0])
    assert_allclose(y_out, expected, atol=1e-12)


def test_unsorted_x2_gives_same_result_as_sorted_x2():
    x1 = np.array([0, 1, 2, 3, 4], dtype=float)
    y1 = np.zeros_like(x1)

    x2_sorted = np.array([1, 2, 3], dtype=float)
    y2_sorted = np.array([10, 20, 30], dtype=float)

    x2_unsorted = np.array([3, 1, 2], dtype=float)
    y2_unsorted = np.array([30, 10, 20], dtype=float)

    out_sorted = smooth_insert(x1, y1, x2_sorted, y2_sorted, favor=1.0)
    out_unsorted = smooth_insert(x1, y1, x2_unsorted, y2_unsorted, favor=1.0)

    assert_allclose(out_unsorted, out_sorted)


def test_interpolates_y2_onto_x1_grid_when_spacing_differs():
    x1 = np.array([0, 1, 2, 3, 4], dtype=float)
    y1 = np.zeros_like(x1)

    # different spacing from x1
    x2 = np.array([0, 2, 4], dtype=float)
    y2 = np.array([0, 20, 40], dtype=float)

    y_out = smooth_insert(x1, y1, x2, y2, favor=1.0)

    # full overlap + favor=1 => output should be pure interpolated y2 on x1 grid
    expected = np.array([0, 10, 20, 30, 40], dtype=float)
    assert_allclose(y_out, expected)


def test_favor_zero_leaves_original_signal_unchanged():
    x1 = np.array([0, 1, 2, 3, 4], dtype=float)
    y1 = np.array([5, 6, 7, 8, 9], dtype=float)
    x2 = np.array([1, 2, 3], dtype=float)
    y2 = np.array([100, 200, 300], dtype=float)

    y_out = smooth_insert(x1, y1, x2, y2, favor=0.0)

    assert_allclose(y_out, y1)


def test_inputs_are_not_modified():
    x1 = np.array([0, 1, 2, 3, 4], dtype=float)
    y1 = np.array([1, 2, 3, 4, 5], dtype=float)
    x2 = np.array([1, 2, 3], dtype=float)
    y2 = np.array([10, 20, 30], dtype=float)

    x1_before = x1.copy()
    y1_before = y1.copy()
    x2_before = x2.copy()
    y2_before = y2.copy()

    _ = smooth_insert(x1, y1, x2, y2)

    assert_array_equal(x1, x1_before)
    assert_array_equal(y1, y1_before)
    assert_array_equal(x2, x2_before)
    assert_array_equal(y2, y2_before)
