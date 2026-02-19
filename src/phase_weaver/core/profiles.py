from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from math import gamma


@dataclass(slots=True)
class AsymSuperGaussParams:
    center: float = 0.0          # s
    width: float = 100e-15       # s
    skew: float = 0.0            # (-1, 1)
    order: float = 2.0           # >= 0.5 typically; order=1 => exp(-|z|^2)
    amplitude: float = 1.0       # peak amplitude (e.g. A)


def asymmetric_super_gaussian(
    t: np.ndarray,
    *,
    center: float,
    width: float,
    skew: float,
    order: float,
    amplitude: float = 1.0,
    normalize_area: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Asymmetric super-Gaussian / generalized Gaussian (piecewise width).

    y(t) = A * exp(-|z|^(2*order)), where
      z = (t-center)/w_left  for t<=center
      z = (t-center)/w_right for t>center
      w_left  = width*(1 - skew)
      w_right = width*(1 + skew)

    Notes:
    - `amplitude` is peak amplitude at t=center (y(center)=amplitude) unless normalize_area=True.
    - `skew` is clamped slightly away from ±1 to keep widths positive.
    - If normalize_area=True, we rescale to make integral ~= 1 over continuous domain.
      For practical grids, you may also normalize numerically later if needed.
    """
    if width <= 0:
        raise ValueError("width must be > 0")
    if order <= 0:
        raise ValueError("order must be > 0")

    # keep widths positive and avoid division blow-ups
    skew = float(np.clip(skew, -1.0 + eps, 1.0 - eps))

    w_left = width * (1.0 - skew)
    w_right = width * (1.0 + skew)

    x = t - center
    z = np.empty_like(x, dtype=float)

    left = x <= 0
    z[left] = x[left] / w_left
    z[~left] = x[~left] / w_right

    exponent = 2.0 * order
    y = np.exp(-np.abs(z) ** exponent)

    if normalize_area:
        # Continuous integral of exp(-|u|^k) is 2*Gamma(1+1/k)/k
        # For our piecewise scaling:
        # ∫ y dt = w_left * ∫_{-∞}^0 exp(-|u|^k) du + w_right * ∫_0^∞ exp(-u^k) du
        # Each half is Gamma(1+1/k)/k, so total = (w_left + w_right) * Gamma(1+1/k)/k
        k = exponent
        area = (w_left + w_right) * (gamma(1.0 + 1.0 / k) / k)
        y = y / area
        # If normalized, amplitude no longer equals peak unless you re-scale; we apply amplitude as overall scale:
        y = amplitude * y
    else:
        # peak amplitude semantics
        y = amplitude * y

    return y


def profile_two_asym_super_gaussians(
    t: np.ndarray,
    p1: AsymSuperGaussParams,
    p2: AsymSuperGaussParams,
    *,
    normalize_total_area: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience: returns (y_total, y1, y2).
    """
    y1 = asymmetric_super_gaussian(
        t,
        center=p1.center, width=p1.width, skew=p1.skew, order=p1.order, amplitude=p1.amplitude
    )
    y2 = asymmetric_super_gaussian(
        t,
        center=p2.center, width=p2.width, skew=p2.skew, order=p2.order, amplitude=p2.amplitude
    )
    y = y1 + y2

    if normalize_total_area:
        # numerical normalization on the given grid
        dt = float(t[1] - t[0])
        area = dt * (y.sum() - 0.5 * (y[0] + y[-1]))
        if area != 0 and np.isfinite(area):
            y = y / area
            y1 = y1 / area
            y2 = y2 / area

    return y, y1, y2