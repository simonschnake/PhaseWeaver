from math import gamma

import numpy as np
import pytest

# Change "your_module" to the actual filename (without .py)
from phase_weaver.model.profiles import (
    AsymSuperGaussParams,
    asymmetric_super_gaussian,
    profile_two_asym_super_gaussians,
)


def trapz_manual(y: np.ndarray, t: np.ndarray) -> float:
    dt = float(t[1] - t[0])
    return dt * (y.sum() - 0.5 * (y[0] + y[-1]))


def test_params_defaults_and_slots():
    p = AsymSuperGaussParams()

    assert p.center == 0.0
    assert p.width == 100e-15
    assert p.skew == 0.0
    assert p.order == 2.0
    assert p.amplitude == 1.0

    # slots=True: no dynamic attributes
    with pytest.raises(AttributeError):
        p.extra = 123


def test_asymmetric_super_gaussian_raises_for_nonpositive_width():
    t = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="width must be > 0"):
        asymmetric_super_gaussian(
            t,
            center=0.0,
            width=0.0,
            skew=0.0,
            order=1.0,
        )


def test_asymmetric_super_gaussian_raises_for_nonpositive_order():
    t = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="order must be > 0"):
        asymmetric_super_gaussian(
            t,
            center=0.0,
            width=1.0,
            skew=0.0,
            order=0.0,
        )


def test_asymmetric_super_gaussian_no_normalization_symmetric_case():
    t = np.array([-1.0, 0.0, 1.0])
    y = asymmetric_super_gaussian(
        t,
        center=0.0,
        width=2.0,
        skew=0.0,
        order=1.0,
        amplitude=3.0,
        normalize_area=False,
    )

    expected = 3.0 * np.exp(-(np.abs(t / 2.0) ** 2.0))
    np.testing.assert_allclose(y, expected, rtol=1e-12, atol=1e-12)

    # peak amplitude semantics when normalize_area=False
    assert y[1] == pytest.approx(3.0)


def test_asymmetric_super_gaussian_piecewise_left_and_right_widths():
    t = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    center = 0.0
    width = 2.0
    skew = 0.25
    order = 1.0
    amplitude = 1.5

    y = asymmetric_super_gaussian(
        t,
        center=center,
        width=width,
        skew=skew,
        order=order,
        amplitude=amplitude,
        normalize_area=False,
    )

    w_left = width * (1.0 - skew)
    w_right = width * (1.0 + skew)

    expected = np.array(
        [
            amplitude * np.exp(-(abs((-2.0 - center) / w_left) ** 2.0)),
            amplitude * np.exp(-(abs((-1.0 - center) / w_left) ** 2.0)),
            amplitude
            * np.exp(-(abs((0.0 - center) / w_left) ** 2.0)),  # x == 0 goes left branch
            amplitude * np.exp(-(abs((1.0 - center) / w_right) ** 2.0)),
            amplitude * np.exp(-(abs((2.0 - center) / w_right) ** 2.0)),
        ]
    )

    np.testing.assert_allclose(y, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("skew", [10.0, -10.0])
def test_asymmetric_super_gaussian_clips_skew(skew):
    t = np.array([-1.0, 0.0, 1.0])
    width = 2.0
    order = 1.0
    amplitude = 1.0
    eps = 1e-12

    y = asymmetric_super_gaussian(
        t,
        center=0.0,
        width=width,
        skew=skew,
        order=order,
        amplitude=amplitude,
        normalize_area=False,
        eps=eps,
    )

    clipped = float(np.clip(skew, -1.0 + eps, 1.0 - eps))
    w_left = width * (1.0 - clipped)
    w_right = width * (1.0 + clipped)

    x = t
    expected = np.empty_like(t, dtype=float)
    left = x <= 0
    expected[left] = np.exp(-(np.abs(x[left] / w_left) ** (2.0 * order)))
    expected[~left] = np.exp(-(np.abs(x[~left] / w_right) ** (2.0 * order)))

    np.testing.assert_allclose(y, expected, rtol=1e-12, atol=1e-12)


def test_asymmetric_super_gaussian_normalize_area_matches_formula_and_integral():
    t = np.linspace(-10.0, 10.0, 200001)
    center = 0.5
    width = 1.7
    skew = 0.2
    order = 1.5
    amplitude = 2.3

    y = asymmetric_super_gaussian(
        t,
        center=center,
        width=width,
        skew=skew,
        order=order,
        amplitude=amplitude,
        normalize_area=True,
    )

    exponent = 2.0 * order
    w_left = width * (1.0 - skew)
    w_right = width * (1.0 + skew)
    area = (w_left + w_right) * (gamma(1.0 + 1.0 / exponent))

    # At t=center, base profile is 1, then scaled by amplitude/area
    center_idx = np.argmin(np.abs(t - center))
    assert y[center_idx] == pytest.approx(amplitude / area, rel=1e-4)

    # Continuous normalization means integrated area should be approximately "amplitude"
    numerical_area = trapz_manual(y, t)
    assert numerical_area == pytest.approx(amplitude, rel=1e-4)


def test_profile_two_asym_super_gaussians_returns_sum_and_components():
    t = np.linspace(-2.0, 2.0, 11)
    p1 = AsymSuperGaussParams(
        center=-0.5, width=0.7, skew=0.1, order=1.0, amplitude=2.0
    )
    p2 = AsymSuperGaussParams(
        center=0.8, width=0.4, skew=-0.2, order=2.0, amplitude=0.5
    )

    y, y1, y2 = profile_two_asym_super_gaussians(
        t,
        p1,
        p2,
        normalize_total_area=False,
    )

    expected_y1 = asymmetric_super_gaussian(
        t,
        center=p1.center,
        width=p1.width,
        skew=p1.skew,
        order=p1.order,
        amplitude=p1.amplitude,
    )
    expected_y2 = asymmetric_super_gaussian(
        t,
        center=p2.center,
        width=p2.width,
        skew=p2.skew,
        order=p2.order,
        amplitude=p2.amplitude,
    )

    np.testing.assert_allclose(y1, expected_y1)
    np.testing.assert_allclose(y2, expected_y2)
    np.testing.assert_allclose(y, expected_y1 + expected_y2)


def test_profile_two_asym_super_gaussians_normalizes_total_area():
    t = np.linspace(-10.0, 10.0, 200001)
    p1 = AsymSuperGaussParams(
        center=-1.0, width=0.8, skew=0.1, order=1.2, amplitude=1.5
    )
    p2 = AsymSuperGaussParams(
        center=1.5, width=1.1, skew=-0.15, order=2.0, amplitude=0.7
    )

    y, y1, y2 = profile_two_asym_super_gaussians(
        t,
        p1,
        p2,
        normalize_total_area=True,
    )

    total_area = trapz_manual(y, t)
    assert total_area == pytest.approx(1.0, rel=1e-5)

    np.testing.assert_allclose(y, y1 + y2, rtol=1e-12, atol=1e-12)


def test_profile_two_asym_super_gaussians_skips_normalization_when_area_is_zero():
    t = np.linspace(-5.0, 5.0, 1001)

    # Same shape, opposite amplitudes -> exact cancellation -> area == 0
    p1 = AsymSuperGaussParams(center=0.0, width=1.0, skew=0.0, order=1.0, amplitude=1.0)
    p2 = AsymSuperGaussParams(
        center=0.0, width=1.0, skew=0.0, order=1.0, amplitude=-1.0
    )

    y, y1, y2 = profile_two_asym_super_gaussians(
        t,
        p1,
        p2,
        normalize_total_area=True,
    )

    np.testing.assert_allclose(y, np.zeros_like(t), atol=1e-12)
    np.testing.assert_allclose(y, y1 + y2, atol=1e-12)


def test_profile_two_asym_super_gaussians_skips_normalization_when_area_is_not_finite():
    t = np.linspace(-1.0, 1.0, 11)

    p1 = AsymSuperGaussParams(
        center=0.0, width=1.0, skew=0.0, order=1.0, amplitude=np.inf
    )
    p2 = AsymSuperGaussParams(center=0.5, width=1.0, skew=0.0, order=1.0, amplitude=1.0)

    y, y1, y2 = profile_two_asym_super_gaussians(
        t,
        p1,
        p2,
        normalize_total_area=True,
    )

    # Since area is non-finite, function should leave arrays unnormalized
    assert np.isinf(y).any()
    np.testing.assert_allclose(y, y1 + y2, equal_nan=True)
