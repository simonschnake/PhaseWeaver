import numpy as np


def trapz_uniform(dt: float, y: np.ndarray) -> float:
    """Fast trapezoid integration for uniform spacing."""
    y = np.asarray(y, dtype=float)
    return float(dt * (y.sum() - 0.5 * (y[0] + y[-1])))


def safe_real(x: np.ndarray) -> np.ndarray:
    """Return real part if complex (numerical leakage), else return float array."""
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return np.asarray(x.real, dtype=float)
    return np.asarray(x, dtype=float)


def fwhm_highest_peak(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    i_peak = np.argmax(y)
    y_peak = y[i_peak]
    half = 0.5 * y_peak

    # left crossing
    left_candidates = np.where(y[:i_peak] < half)[0]
    if len(left_candidates) == 0:
        raise ValueError("No left half-maximum crossing found.")

    i1 = left_candidates[-1]  # below half
    i2 = i1 + 1  # at/above half
    x_left = np.interp(half, [y[i1], y[i2]], [x[i1], x[i2]])

    # right crossing
    right_candidates = np.where(y[i_peak + 1 :] < half)[0]
    if len(right_candidates) == 0:
        raise ValueError("No right half-maximum crossing found.")

    i2 = i_peak + 1 + right_candidates[0]  # first point below half
    i1 = i2 - 1  # previous point at/above half

    # y-values must be increasing for np.interp
    x_right = np.interp(half, [y[i2], y[i1]], [x[i2], x[i1]])

    return x_right - x_left, x_left, x_right, half
