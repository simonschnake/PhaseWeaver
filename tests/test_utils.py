import numpy as np
import pytest

from phase_weaver.core.utils import (
    fwhm_highest_peak,
    safe_real,
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
