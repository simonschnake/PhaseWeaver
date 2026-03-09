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


def smooth_insert(x1, y1, x2, y2, favor=0.8, power=2.0):
    """
    Smoothly inserts (x2, y2) into (x1, y1).

    Parameters
    ----------
    x1, y1 : 1D arrays
        Base function.
    x2, y2 : 1D arrays
        Function to insert. x2 may have different spacing.
    favor : float in [0, 1]
        How much to favor y2 in the overlap.
        0.5 = equal blend, 0.8 = mostly y2.
    power : float
        Controls smoothness shape. >1 gives softer transition.

    Returns
    -------
    y_out : 1D array
        Modified y-values on x1 grid.
    """
    x1 = np.asarray(x1)
    y1 = np.asarray(y1)
    x2 = np.asarray(x2)
    y2 = np.asarray(y2)

    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("x1 and x2 must be 1D")
    if len(x1) != len(y1) or len(x2) != len(y2):
        raise ValueError("x/y lengths must match")

    # ensure x2 is sorted
    order = np.argsort(x2)
    x2 = x2[order]
    y2 = y2[order]

    # overlap region on x1
    mask = (x1 >= x2[0]) & (x1 <= x2[-1])
    if not np.any(mask):
        return y1.copy()  # no overlap

    y_out = y1.copy()

    # interpolate y2 onto x1 grid only in overlap
    x_overlap = x1[mask]
    y2_interp = np.interp(x_overlap, x2, y2)

    n = len(x_overlap)

    # detect position of overlap: beginning, middle, end, or whole domain
    at_start = np.isclose(x_overlap[0], x1[0])
    at_end = np.isclose(x_overlap[-1], x1[-1])

    if at_start and at_end:
        # x2 covers all of x1 -> just favor y2 everywhere
        w = np.full(n, favor)

    elif at_start:
        # x2 covers beginning of x1:
        # start strongly with y2, fade toward y1 at the right edge
        t = np.linspace(0, 1, n)
        w = favor * (1 - t**power)

    elif at_end:
        # x2 covers end of x1:
        # fade from y1 into y2
        t = np.linspace(0, 1, n)
        w = favor * (t**power)

    else:
        # x2 is in the middle:
        # fade in to y2, stay high, then fade out
        t = np.linspace(0, 1, n)
        w = np.sin(np.pi * t) ** power  # 0 at edges, 1 in middle
        w = favor * w

    # blend
    y_out[mask] = (1 - w) * y1[mask] + w * y2_interp
    return y_out
