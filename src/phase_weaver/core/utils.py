from dataclasses import dataclass

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


@dataclass(slots=True, frozen=True)
class FWHMResult:
    left_half_max: float
    right_half_max: float
    half_max_value: float

    @property
    def fwhm(self) -> float:
        return self.right_half_max - self.left_half_max


def fwhm_highest_peak(x: np.ndarray, y: np.ndarray) -> FWHMResult | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        print("Warning: x and y must be 1D. Returning None.")
        return None

    i_peak = np.argmax(y)
    y_peak = y[i_peak]
    half = 0.5 * y_peak

    # left crossing
    left_candidates = np.where(y[:i_peak] < half)[0]
    if len(left_candidates) == 0:
        print(
            "Warning: No left half-maximum crossing found. Returning NaN for left_half_max."
        )
        return None

    i1 = left_candidates[-1]  # below half
    i2 = i1 + 1  # at/above half
    x_left = np.interp(half, [y[i1], y[i2]], [x[i1], x[i2]])

    # right crossing
    right_candidates = np.where(y[i_peak + 1 :] < half)[0]
    if len(right_candidates) == 0:
        print(
            "Warning: No right half-maximum crossing found. Returning NaN for right_half_max."
        )
        return None

    i2 = i_peak + 1 + right_candidates[0]  # first point below half
    i1 = i2 - 1  # previous point at/above half

    # y-values must be increasing for np.interp
    x_right = np.interp(half, [y[i2], y[i1]], [x[i2], x[i1]])

    return FWHMResult(left_half_max=x_left, right_half_max=x_right, half_max_value=half)


def overlap_weights(
    x: np.ndarray,
    transition_width_left: float = 0.0,
    transition_width_right: float = 0.0,
    power: float = 2.0,
) -> np.ndarray:
    """
    Build smooth blend weights on a 1D target interval.

    The returned weights are always in [0, 1]. The function assumes that
    `x` describes the region where blending may happen.

    Parameters
    ----------
    x:
        1D x-values of the region to weight.
    transition_width_left:
        Width of the fade-in region from the left edge of `x`.
        - 0.0 means no left fade-in (weights start at 1 on the left).
    transition_width_right:
        Width of the fade-out region toward the right edge of `x`.
        - 0.0 means no right fade-out (weights stay at 1 on the right).
    power:
        Shape parameter for the ramps. Must be positive.
        Larger values make the transition softer / flatter near the plateau.

    Returns
    -------
    w:
        1D weights in [0, 1], same shape as `x`.

    Notes
    -----
    Behavior by parameter choice:

    - left=0, right=0:
        constant ones
    - left>0, right=0:
        fade in from the left, then plateau
    - left=0, right>0:
        plateau, then fade out to the right
    - left>0, right>0:
        fade in, optional plateau, fade out

    If the left and right transition widths overlap, the function still returns
    a smooth profile without a flat plateau.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("x_target must be 1D")
    if len(x) == 0:
        return np.empty(0, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("x_target must contain only finite values")
    if transition_width_left < 0:
        raise ValueError("transition_width_left must be non-negative")
    if transition_width_right < 0:
        raise ValueError("transition_width_right must be non-negative")
    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be positive")
    if len(x) > 1 and np.any(np.diff(x) < 0):
        raise ValueError("x_target must be sorted in non-decreasing order")

    if len(x) == 1:
        left_active = transition_width_left > 0
        right_active = transition_width_right > 0

        if left_active and right_active:
            return np.array([0.0], dtype=float)
        if left_active or right_active:
            return np.array([0.0], dtype=float)
        return np.array([1.0], dtype=float)

    x0 = x[0]
    x1 = x[-1]

    def smoothstep(t: np.ndarray) -> np.ndarray:
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    # Left ramp: 0 -> 1 over transition_width_left
    if transition_width_left == 0:
        w_left = np.ones_like(x)
    else:
        t_left = (x - x0) / transition_width_left
        w_left = smoothstep(t_left) ** power
        w_left[0] = 0.0

    # Right ramp: 1 -> 0 over transition_width_right
    if transition_width_right == 0:
        w_right = np.ones_like(x)
    else:
        t_right = (x1 - x) / transition_width_right
        w_right = smoothstep(t_right) ** power
        w_right[-1] = 0.0

    # Combine both tapers
    w = np.minimum(w_left, w_right)
    return np.clip(w, 0.0, 1.0)


def smooth_overlap(
    x_target: np.ndarray,
    y_target: np.ndarray,
    x_source: np.ndarray,
    y_source: np.ndarray,
    transition_width: float | None = None,
    power: float = 2.0,
) -> np.ndarray:
    """
    Smoothly blend source data into target data on the target grid.

    The source is interpolated onto the target grid over the overlapping region.
    Smooth transitions are applied at the overlap boundaries unless the source
    already contains the corresponding target boundary.

    Parameters
    ----------
    x_target, y_target:
        Target function on the target grid.
    x_source, y_source:
        Source function to blend in. `x_source` may have different spacing.
    transition_width:
        Width of the transition region in x-units.
        - If None, uses half of the target span.
        - If the source contains the left end of the target, no left transition
          is applied.
        - If the source contains the right end of the target, no right transition
          is applied.
    power:
        Shape parameter for the taper. Must be positive.

    Returns
    -------
    y_out:
        Smoothed result on the target grid.

    Notes
    -----
    This function performs a full replacement in the plateau region.
    If you later want partial trust in the source, use:

        y_out = (1 - favor * w) * y_target + (favor * w) * y_source_interp
    """
    x_target = np.asarray(x_target, dtype=float)
    y_target = np.asarray(y_target, dtype=float)
    x_source = np.asarray(x_source, dtype=float)
    y_source = np.asarray(y_source, dtype=float)

    if x_target.ndim != 1 or x_source.ndim != 1:
        raise ValueError("x_target and x_source must be 1D")
    if len(x_target) != len(y_target) or len(x_source) != len(y_source):
        raise ValueError("x/y lengths must match")
    if len(x_target) == 0:
        raise ValueError("x_target and y_target must not be empty")
    if len(x_source) == 0:
        raise ValueError("x_source and y_source must not be empty")
    if not np.all(np.isfinite(x_target)) or not np.all(np.isfinite(y_target)):
        raise ValueError("x_target and y_target must contain only finite values")
    if not np.all(np.isfinite(x_source)) or not np.all(np.isfinite(y_source)):
        raise ValueError("x_source and y_source must contain only finite values")
    if np.any(np.diff(x_target) < 0):
        raise ValueError("x_target must be sorted in non-decreasing order")
    if transition_width is not None and transition_width < 0:
        raise ValueError("transition_width must be non-negative")
    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be positive")

    # sort source if needed
    order = np.argsort(x_source)
    x_source = x_source[order]
    y_source = y_source[order]

    # overlap of source coverage with target grid
    overlap = (x_target >= x_source[0]) & (x_target <= x_source[-1])
    if not np.any(overlap):
        return y_target.copy()

    x_overlap = x_target[overlap]
    y_source_interp = np.interp(x_overlap, x_source, y_source)

    if transition_width is None:
        target_span = x_target[-1] - x_target[0]
        transition_width = 0.5 * target_span

    left_end_contained = x_source[0] <= x_target[0] <= x_source[-1]
    right_end_contained = x_source[0] <= x_target[-1] <= x_source[-1]

    transition_width_left = 0.0 if left_end_contained else transition_width
    transition_width_right = 0.0 if right_end_contained else transition_width

    w = overlap_weights(
        x=x_overlap,
        transition_width_left=transition_width_left,
        transition_width_right=transition_width_right,
        power=power,
    )

    y_out = y_target.copy()
    y_out[overlap] = (1.0 - w) * y_target[overlap] + w * y_source_interp
    return y_out


def exponential_extend(
    x_target: np.ndarray, x_source: np.ndarray, y_source: np.ndarray, power: float = 2.0
) -> np.ndarray:
    """
    Extend y_source outside x_source using Gaussian tails

        y = A * exp(-1/2 (sigma * x)^2 )

    fitted independently on the left and right with weighted least squares.

    Inside the source interval, values are linearly interpolated.

    Parameters
    ----------
    x_target : np.ndarray
        Points where the extended signal is evaluated.
    x_source : np.ndarray
        Strictly increasing source x values.
    y_source : np.ndarray
        Source y values.
    power : float, default=2.0
        Exponent applied to the linear edge-weight ramp.
        Larger values emphasize the boundary region more strongly.

    Returns
    -------
    np.ndarray
        Signal evaluated at x_target, interpolated inside the source interval
        and Gaussian-extended outside it.
    """
    x_target = np.asarray(x_target, dtype=float)
    x_source = np.asarray(x_source, dtype=float)
    y_source = np.asarray(y_source, dtype=float)

    if x_target.ndim != 1 or x_source.ndim != 1 or y_source.ndim != 1:
        raise ValueError("x_target, x_source, and y_source must be 1D arrays.")
    if len(x_source) != len(y_source):
        raise ValueError("x_source and y_source must have the same length.")
    if len(x_source) < 3:
        raise ValueError("Need at least 3 source points.")
    if not np.all(np.isfinite(x_source)) or not np.all(np.isfinite(y_source)):
        raise ValueError("x_source and y_source must contain only finite values.")
    if not np.all(np.diff(x_source) > 0):
        raise ValueError("x_source must be strictly increasing.")
    if power <= 0:
        raise ValueError("power must be positive.")

    n = len(x_source)
    w_left = np.linspace(100.0, 1.0, n) ** power
    w_right = np.linspace(1.0, 100.0, n) ** power

    yy = -2 * np.log(y_source)
    X = np.column_stack([x_source * x_source, np.ones_like(x_source)])

    sw_left = np.sqrt(w_left)

    Xw_left = X * sw_left[:, np.newaxis]
    yw_left = yy * sw_left

    sigma2_left, neg_two_ln_A_left = np.linalg.lstsq(Xw_left, yw_left, rcond=None)[0]
    A_left = np.exp(-0.5 * neg_two_ln_A_left)

    sw_right = np.sqrt(w_right)
    X_right = X * sw_right[:, np.newaxis]
    yw_right = yy * sw_right

    sigma2_right, neg_two_ln_A_right = np.linalg.lstsq(X_right, yw_right, rcond=None)[0]
    A_right = np.exp(-0.5 * neg_two_ln_A_right)

    y_target = np.interp(x_target, x_source, y_source)

    left_mask = x_target < x_source[0]
    right_mask = x_target > x_source[-1]

    if np.any(left_mask):
        x = x_target[left_mask]
        y_target[left_mask] = A_left * np.exp(-0.5 * sigma2_left * x**2)

    if np.any(right_mask):
        x = x_target[right_mask]
        y_target[right_mask] = A_right * np.exp(-0.5 * sigma2_right * x**2)

    return y_target


def quadratic_log_extend(
    x_target: np.ndarray,
    x_source: np.ndarray,
    y_source: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:
    """
    Extend y_source outside x_source by fitting quadratic models to log(y)
    on the left and right:

        log(y) = a*x^2 + b*x + c

    using weighted least squares.

    Inside the source interval, values are linearly interpolated.

    Parameters
    ----------
    x_target : np.ndarray
        Points where the extended signal is evaluated.
    x_source : np.ndarray
        Strictly increasing source x values.
    y_source : np.ndarray
        Source y values. Must be strictly positive.
    power : float, default=2.0
        Exponent applied to the edge-weight ramp.
        Larger values emphasize boundary regions more strongly.

    Returns
    -------
    np.ndarray
        Signal evaluated at x_target, interpolated inside the source interval
        and exponentially extended outside it.
    """
    x_target = np.asarray(x_target, dtype=float)
    x_source = np.asarray(x_source, dtype=float)
    y_source = np.asarray(y_source, dtype=float)

    if x_target.ndim != 1 or x_source.ndim != 1 or y_source.ndim != 1:
        raise ValueError("x_target, x_source, and y_source must be 1D arrays.")
    if len(x_source) != len(y_source):
        raise ValueError("x_source and y_source must have the same length.")
    if len(x_source) < 3:
        raise ValueError("Need at least 3 source points.")
    if not np.all(np.isfinite(x_source)) or not np.all(np.isfinite(y_source)):
        raise ValueError("x_source and y_source must contain only finite values.")
    if not np.all(np.diff(x_source) > 0):
        raise ValueError("x_source must be strictly increasing.")
    if np.any(y_source <= 0):
        raise ValueError("y_source must be strictly positive for log fitting.")
    if power <= 0:
        raise ValueError("power must be positive.")

    n = len(x_source)
    w_left = np.linspace(100.0, 1.0, n) ** power
    w_left = w_left / w_left.sum()  # normalize to sum to 1
    w_right = np.linspace(1.0, 100.0, n) ** power
    w_right = w_right / w_right.sum()  # normalize to sum to 1

    logy = np.log(y_source)
    X = np.column_stack([x_source**2, x_source, np.ones_like(x_source)])

    def fit_quadratic_log(weights):
        sw = np.sqrt(weights)
        coeffs, *_ = np.linalg.lstsq(X * sw[:, None], logy * sw, rcond=None)
        return coeffs  # a, b, c

    a_left, b_left, c_left = fit_quadratic_log(w_left)
    a_right, b_right, c_right = fit_quadratic_log(w_right)

    y_target = np.interp(x_target, x_source, y_source)

    left_mask = x_target <= x_source[0]
    right_mask = x_target >= x_source[-1]

    if np.any(left_mask):
        x = x_target[left_mask]
        y_target[left_mask] = np.exp(a_left * x**2 + b_left * x + c_left)

    if np.any(right_mask):
        x = x_target[right_mask]
        y_target[right_mask] = np.exp(a_right * x**2 + b_right * x + c_right)

    return y_target


def gaussian_extend(
    x_target: np.ndarray,
    x_source: np.ndarray,
    y_source: np.ndarray,
    *,
    n_tail: int = 5,
    lower_derivative_zero_at: float = 0.0,
    min_positive: float | None = None,
) -> np.ndarray:
    """
    Extend y_source outside x_source.

    Inside the source interval, values are linearly interpolated.

    Below the source range, the extension is fitted in log-space with the
    constraint that dy/dx = 0 at lower_derivative_zero_at.

        log(y) = a + b * (x - lower_derivative_zero_at)^2

    This guarantees:

        d/dx log(y) = 0 at x = lower_derivative_zero_at

    and therefore:

        dy/dx = 0 at x = lower_derivative_zero_at

    Above the source range, the extension uses a log-linear tail:

        log(y) = a + b * x

    Parameters
    ----------
    x_target : np.ndarray
        Points where the extended signal is evaluated.
    x_source : np.ndarray
        Strictly increasing source x values.
    y_source : np.ndarray
        Source y values. Must be strictly positive unless min_positive is given.
    n_tail : int, default 5
        Number of points from each end of x_source used to fit each tail.
    lower_derivative_zero_at : float, default 0.0
        Location where the lower-side extension should have derivative zero.
    min_positive : float or None, default None
        Optional lower clipping value for y before fitting in log-space.

    Returns
    -------
    np.ndarray
        Signal evaluated at x_target.
    """
    x_target = np.asarray(x_target, dtype=float)
    x_source = np.asarray(x_source, dtype=float)
    y_source = np.asarray(y_source, dtype=float)

    if x_target.ndim != 1 or x_source.ndim != 1 or y_source.ndim != 1:
        raise ValueError("x_target, x_source, and y_source must be 1D arrays.")
    if len(x_source) != len(y_source):
        raise ValueError("x_source and y_source must have the same length.")
    if len(x_source) < 3:
        raise ValueError("Need at least 3 source points.")
    if not np.all(np.isfinite(x_target)):
        raise ValueError("x_target must contain only finite values.")
    if not np.all(np.isfinite(x_source)) or not np.all(np.isfinite(y_source)):
        raise ValueError("x_source and y_source must contain only finite values.")
    if not np.all(np.diff(x_source) > 0):
        raise ValueError("x_source must be strictly increasing.")

    n = len(x_source)
    if not 2 <= n_tail <= n:
        raise ValueError("n_tail must satisfy 2 <= n_tail <= len(x_source).")

    if min_positive is None:
        if np.any(y_source <= 0):
            raise ValueError(
                "Log-space extension requires y_source to be strictly positive. "
                "Pass min_positive to clip non-positive values if appropriate."
            )
        y_fit = y_source
    else:
        if min_positive <= 0:
            raise ValueError("min_positive must be positive.")
        y_fit = np.maximum(y_source, min_positive)

    log_y_fit = np.log(y_fit)

    x_left = x_source[0]
    x_right = x_source[-1]

    # ------------------------------------------------------------------
    # Lower-side extension:
    #
    #     log(y) = a + b * (x - x0)^2
    #
    # This guarantees derivative zero at x0.
    # ------------------------------------------------------------------
    x0 = lower_derivative_zero_at

    x_lower_tail = x_source[:n_tail]
    log_y_lower_tail = log_y_fit[:n_tail]

    u_lower = (x_lower_tail - x0) ** 2

    # Linear least squares:
    # log(y) = a + b * u
    lower_coeff = np.polyfit(u_lower, log_y_lower_tail, deg=1)
    b_lower = lower_coeff[0]
    a_lower = lower_coeff[1]

    # ------------------------------------------------------------------
    # Upper-side extension:
    #
    #     log(y) = a + b * x
    #
    # You can change this to another constrained model if needed.
    # ------------------------------------------------------------------
    x_upper_tail = x_source[-n_tail:]
    log_y_upper_tail = log_y_fit[-n_tail:]

    upper_coeff = np.polyfit(x_upper_tail, log_y_upper_tail, deg=1)
    b_upper = upper_coeff[0]
    a_upper = upper_coeff[1]

    y_target = np.empty_like(x_target, dtype=float)

    mask_lower = x_target < x_left
    mask_inside = (x_target >= x_left) & (x_target <= x_right)
    mask_upper = x_target > x_right

    # Inside: ordinary interpolation in original y-space
    y_target[mask_inside] = np.interp(
        x_target[mask_inside],
        x_source,
        y_source,
    )

    # Below range: constrained log-space extension
    log_y_lower = a_lower + b_lower * (x_target[mask_lower] - x0) ** 2
    y_target[mask_lower] = np.exp(log_y_lower)

    # Above range: log-linear extension
    log_y_upper = a_upper + b_upper * x_target[mask_upper]
    y_target[mask_upper] = np.exp(log_y_upper)

    return y_target


def interpolate_between_functions(
    x: np.ndarray,
    f_1: np.ndarray,
    f_2: np.ndarray,
    x_1: float,
    x_2: float,
    power: float = 1.0,
) -> np.ndarray:
    """
    Piecewise interpolate between two functions defined on the same x-grid.

    Regions:
        x < x_1:       f_1
        x_1 -> x_2:    smooth interpolation from f_1 to f_2
        x > x_2:       f_2

    Parameters
    ----------
    x:
        1D sorted x-grid.
    f_1, f_2:
        Function values on the same grid.
    x_1:
        Start of transition.
    x_2:
        End of transition. Must be greater than x_1.
    power:
        Shape parameter. 1.0 gives normal smoothstep.
        Larger values delay the transition near x_1 and steepen it later.

    Returns
    -------
    f_out:
        Blended function on x.
    """
    x = np.asarray(x, dtype=float)
    f_1 = np.asarray(f_1, dtype=float)
    f_2 = np.asarray(f_2, dtype=float)

    if x.ndim != 1 or f_1.ndim != 1 or f_2.ndim != 1:
        raise ValueError("x, f_1, and f_2 must be 1D arrays")
    if len(x) != len(f_1) or len(x) != len(f_2):
        raise ValueError("x, f_1, and f_2 must have the same length")
    if len(x) == 0:
        raise ValueError("arrays must not be empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    if not np.all(np.isfinite(f_1)) or not np.all(np.isfinite(f_2)):
        raise ValueError("f_1 and f_2 must contain only finite values")
    if np.any(np.diff(x) < 0):
        raise ValueError("x must be sorted in non-decreasing order")
    if not np.isfinite(x_1) or not np.isfinite(x_2):
        raise ValueError("x_1 and x_2 must be finite")
    if x_2 <= x_1:
        raise ValueError("x_2 must be greater than x_1")
    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be positive")

    t = (x - x_1) / (x_2 - x_1)
    t = np.clip(t, 0.0, 1.0)

    # Smoothstep: 0 -> 1 with zero slope at both ends.
    w = t * t * (3.0 - 2.0 * t)

    # Optional shaping.
    w = w**power

    return (1.0 - w) * f_1 + w * f_2
