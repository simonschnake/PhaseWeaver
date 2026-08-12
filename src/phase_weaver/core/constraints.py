from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from .utils import smooth_overlap, trapz_uniform
from .measurement import MeasuredFormFactor

if TYPE_CHECKING:
    from .base import Profile, CurrentProfile, FormFactor, Grid


class TimeConstraint(ABC):
    def __init__(self, *constraints: TimeConstraint):
        self._constraints = constraints or (self,)

    @abstractmethod
    def apply(self, prof: Profile) -> None: ...

    def __add__(self, other: TimeConstraint) -> TimeConstraint:
        return CombinedTimeConstraint(*self._constraints, *other._constraints)


class CombinedTimeConstraint(TimeConstraint):
    def __init__(self, *constraints: TimeConstraint):
        self._constraints = constraints

    def apply(self, prof: Profile) -> None:
        for constraint in self._constraints:
            constraint.apply(prof)


class FrequencyConstraint(ABC):
    def __init__(self, *constraints: FrequencyConstraint):
        self._constraints = constraints or (self,)

    @abstractmethod
    def apply(self, ff: "FormFactor") -> None: ...

    def __add__(self, other: FrequencyConstraint) -> FrequencyConstraint:
        return CombinedFrequencyConstraint(*self._constraints, *other._constraints)


class CombinedFrequencyConstraint(FrequencyConstraint):
    def __init__(self, *constraints: FrequencyConstraint):
        self._constraints = constraints

    def apply(self, ff: "FormFactor") -> None:
        for constraint in self._constraints:
            constraint.apply(ff)


class NonNegativity(TimeConstraint):
    def apply(self, prof: "CurrentProfile") -> None:
        prof.values = np.maximum(prof.values, 0.0)


class NormalizeArea(TimeConstraint):
    def apply(self, prof: "CurrentProfile") -> None:
        area = trapz_uniform(prof.grid.dt, prof.values)
        if area == 0:
            print("Warning: area is zero, cannot normalize.")
            return
        prof.values /= area


class CenterFirstMoment(TimeConstraint):
    def apply(self, prof: "CurrentProfile") -> None:
        g = prof.grid
        mean_t = trapz_uniform(g.dt, prof.values * g.t)
        shift = int(np.round(mean_t / g.dt))
        prof.values = np.roll(prof.values, -shift)


class ClampMagnitude(FrequencyConstraint):
    def __init__(self, eps: float = np.finfo(float).eps):
        super().__init__()
        self.eps = eps

    def apply(self, ff: "FormFactor") -> None:
        ff.mag = np.maximum(ff.mag, self.eps)


class EnforceDCOne(FrequencyConstraint):
    def __init__(self, eps: float = np.finfo(float).eps):
        super().__init__()
        self.eps = eps

    def apply(self, ff: "FormFactor") -> None:
        ff.mag /= np.max([ff.mag[0], self.eps])


class HighFrequencyMagnitudeDecay(FrequencyConstraint):
    """
    Smoothly attenuate unmeasured high-frequency magnitudes with an exponential tail.

    Frequencies up to ``start_freq`` are left unchanged. Between ``start_freq``
    and ``end_freq`` the attenuation falls to ``attenuation_at_end``. It
    continues to decay beyond ``end_freq`` rather than
    imposing a hard cutoff, which suppresses unmeasured frequencies without a
    spectral edge that would cause time-domain ringing.
    """

    def __init__(
        self,
        start_freq: float,
        end_freq: float,
        floor: float = 0.0,
        attenuation_at_end: float = 1e-3,
    ):
        super().__init__()
        if not np.isfinite(start_freq) or start_freq < 0:
            raise ValueError("start_freq must be finite and non-negative")
        if not np.isfinite(end_freq):
            raise ValueError("end_freq must be finite")
        if end_freq <= start_freq:
            raise ValueError("end_freq must be greater than start_freq")
        if not np.isfinite(floor) or not 0.0 <= floor < 1.0:
            raise ValueError("floor must be finite and in [0, 1)")
        if not np.isfinite(attenuation_at_end) or not 0.0 < attenuation_at_end < 1.0:
            raise ValueError("attenuation_at_end must be finite and in (0, 1)")

        self.start_freq = float(start_freq)
        self.end_freq = float(end_freq)
        self.floor = float(floor)
        self.attenuation_at_end = float(attenuation_at_end)

    def apply(self, ff: "FormFactor") -> None:
        f_pos = ff.grid.f_pos
        tail_mask = f_pos > self.start_freq
        if not np.any(tail_mask):
            return

        normalized_distance = (f_pos[tail_mask] - self.start_freq) / (
            self.end_freq - self.start_freq
        )
        decay = np.exp(np.log(self.attenuation_at_end) * normalized_distance**2)
        envelope = self.floor + (1.0 - self.floor) * decay
        ff.mag[tail_mask] *= envelope


class ReplacePhaseEndLinear(FrequencyConstraint):
    def __init__(self, grid: "Grid", start_freq: float, alpha: float):
        super().__init__()
        self.grid = grid
        self.start_freq = start_freq
        self.alpha = alpha

        if not np.isfinite(self.start_freq) or self.start_freq < 0:
            raise ValueError("start_freq must be finite and non-negative")
        if start_freq > grid.f_pos[-1]:
            raise ValueError("start_freq is above the maximum frequency in the grid")

        self._replace_mask = self.grid.f_pos >= self.start_freq

        self._values = self.alpha * self.grid.f_pos[self._replace_mask]

    def apply(self, ff: "FormFactor") -> None:
        ff.phase[self._replace_mask] = self._values


class ReplacePhaseEndLinearShift(FrequencyConstraint):
    def __init__(self, grid: "Grid", start_freq: float, freq_x: float, freq_y: float):
        super().__init__()
        self.grid = grid
        self.start_freq = start_freq
        self.freq_x = freq_x
        self.freq_y = freq_y

        if self.start_freq < 0:
            raise ValueError("start_freq must be non-negative")

        self._replace_mask = self.grid.f_pos >= self.start_freq
        if not np.any(self._replace_mask):
            raise ValueError("start_freq is above the maximum frequency in the grid")

        if self.freq_x <= self.start_freq:
            raise ValueError("freq_x must be greater than start_freq")

        if self.freq_x > self.grid.f_pos[-1]:
            raise ValueError(
                "freq_x must be less than or equal to the maximum frequency in the grid"
            )

        self._freqs_mask = self.grid.f_pos[self._replace_mask]
        self._start_x = self._freqs_mask[0]

    def apply(self, ff: "FormFactor") -> None:
        self._start_y = ff.phase[self._replace_mask][0]
        slope = (self.freq_y - self._start_y) / (self.freq_x - self._start_x)
        values = slope * (self._freqs_mask - self._start_x) + self._start_y
        ff.phase[self._replace_mask] = values


class ReplacePhaseEndLinearSmooth(FrequencyConstraint):
    """
    Smoothly blend in a linear high-frequency phase tail.

    The inserted tail is the line through (0, 0) and (freq_x, freq_y), applied
    from `start_freq` onward and smoothly merged into the original phase.
    """

    def __init__(
        self,
        start_freq: float,
        freq_x: float,
        freq_y: float,
        power: float = 2.0,
        transition_width: float | None = None,
        *constraints: FrequencyConstraint,
    ):
        super().__init__()
        self.start_freq = start_freq
        self.freq_x = freq_x
        self.freq_y = freq_y
        self.power = power
        self.transition_width = transition_width

        if not np.isfinite(self.start_freq) or self.start_freq < 0:
            raise ValueError("start_freq must be finite and non-negative")
        if not np.isfinite(self.freq_x):
            raise ValueError("freq_x must be finite")
        if not np.isfinite(self.freq_y):
            raise ValueError("freq_y must be finite")
        if not np.isfinite(self.power) or self.power <= 0:
            raise ValueError("power must be positive")
        if self.transition_width is not None and (
            not np.isfinite(self.transition_width) or self.transition_width < 0
        ):
            raise ValueError("transition_width must be finite and non-negative")

        if self.freq_x <= self.start_freq:
            raise ValueError("freq_x must be greater than start_freq")

        self.slope = self.freq_y / self.freq_x

    def apply(self, ff: "FormFactor") -> None:
        f_pos = ff.grid.f_pos

        if self.freq_x > f_pos[-1]:
            raise ValueError(
                "freq_x must be less than or equal to the maximum frequency in the grid"
            )

        replace_mask = f_pos >= self.start_freq

        f_mask = np.asarray(f_pos[replace_mask], dtype=float)
        lin_phase = self.slope * f_mask

        ff.phase = smooth_overlap(
            x_target=f_pos,
            y_target=ff.phase,
            x_source=f_mask,
            y_source=lin_phase,
            power=self.power,
            transition_width=self.transition_width,
        )


class BlendMeasuredMagnitude(FrequencyConstraint):
    """
    Smoothly blend a measured magnitudes into ff.mag.

    Parameters
    ----------
    measured:
        Measured form factor.
    favor:
        Blend strength in [0, 1]. Higher means closer to measured data.
    transition_width:
        Width of the smooth transition at the overlap boundaries, in frequency
        units. If None, defaults to 20% of overlap span.
    """

    def __init__(
        self,
        measured: tuple[MeasuredFormFactor, ...],
        power: float = 2.0,
        transition_width: float | None = None,
        transition_width_left: tuple[float, ...] | None = None,
        transition_width_right: tuple[float, ...] | None = None,
        scale: bool = False,
    ):
        super().__init__()
        if not np.isfinite(power) or power <= 0:
            raise ValueError("power must be positive")
        if transition_width is not None and (
            not np.isfinite(transition_width) or transition_width < 0
        ):
            raise ValueError("transition_width must be finite and non-negative")

        self.measured = measured
        self.power = float(power)
        self.transition_width = (
            float(transition_width) if transition_width is not None else None
        )
        self.transition_width_left = self._validate_edge_widths(
            transition_width_left,
            "transition_width_left",
        )
        self.transition_width_right = self._validate_edge_widths(
            transition_width_right,
            "transition_width_right",
        )

        self.scale = scale
        if scale:
            self._avg_measured_mag = [mes.mag.mean() for mes in self.measured]

    def _validate_edge_widths(
        self,
        widths: tuple[float, ...] | None,
        name: str,
    ) -> tuple[float, ...] | None:
        if widths is None:
            return None
        if len(widths) != len(self.measured):
            raise ValueError(f"{name} must match the number of measurements")
        values = tuple(float(width) for width in widths)
        if any(not np.isfinite(width) or width < 0.0 for width in values):
            raise ValueError(f"{name} must contain finite non-negative widths")
        return values

    def apply(self, ff: "FormFactor") -> None:
        for i, meas in enumerate(self.measured):
            if self.scale:
                current_mag_at_measured_freq = np.interp(
                    meas.freq,
                    ff.grid.f_pos,
                    ff.mag,
                )
                avg_current_mag = current_mag_at_measured_freq.mean()
                y_source = meas.mag * avg_current_mag / self._avg_measured_mag[i]

            else:
                y_source = meas.mag

            ff.mag = smooth_overlap(
                x_target=ff.grid.f_pos,
                y_target=ff.mag,
                x_source=meas.freq,
                y_source=y_source,
                power=self.power,
                transition_width=self.transition_width,
                transition_width_left=(
                    None
                    if self.transition_width_left is None
                    else self.transition_width_left[i]
                ),
                transition_width_right=(
                    None
                    if self.transition_width_right is None
                    else self.transition_width_right[i]
                ),
            )


class BlendRelativeMeasuredShape(BlendMeasuredMagnitude):
    """Blend relative shapes after scaling, weighted by supplied uncertainty."""

    def __init__(
        self,
        measured: tuple[MeasuredFormFactor, ...],
        power: float = 2.0,
        transition_width: float | None = None,
        transition_width_left: tuple[float, ...] | None = None,
        transition_width_right: tuple[float, ...] | None = None,
        anchor_formfactor: "FormFactor | None" = None,
        fixed_scale: float | None = None,
    ):
        super().__init__(
            measured,
            power=power,
            transition_width=transition_width,
            transition_width_left=transition_width_left,
            transition_width_right=transition_width_right,
        )
        self.anchor_formfactor = anchor_formfactor
        if fixed_scale is not None and (
            not np.isfinite(fixed_scale) or fixed_scale < 0.0
        ):
            raise ValueError("fixed_scale must be finite and non-negative")
        self.fixed_scale = None if fixed_scale is None else float(fixed_scale)

    def apply(self, ff: "FormFactor") -> None:
        eps = np.finfo(float).eps
        for i, meas in enumerate(self.measured):
            current_mag = np.interp(
                meas.freq,
                ff.grid.f_pos,
                ff.mag,
            )
            confidence = np.ones_like(meas.mag)
            if meas.mag_std is not None:
                snr = meas.mag / np.maximum(meas.mag_std, eps)
                confidence = np.square(snr) / (1.0 + np.square(snr))
            if meas.detection_limit is not None:
                confidence = np.where(
                    meas.mag >= meas.detection_limit,
                    confidence,
                    0.0,
                )

            if self.fixed_scale is None:
                anchor_ff = self.anchor_formfactor or ff
                scale_anchor_mag = np.interp(
                    meas.freq,
                    anchor_ff.grid.f_pos,
                    anchor_ff.mag,
                )
                valid = (
                    np.isfinite(meas.mag)
                    & np.isfinite(scale_anchor_mag)
                    & (meas.mag > eps)
                    & (scale_anchor_mag >= 0.0)
                    & (confidence > 0.0)
                )
                if not np.any(valid):
                    continue

                if meas.mag_std is None:
                    measured_average = float(np.mean(meas.mag[valid]))
                    reconstruction_average = float(np.mean(scale_anchor_mag[valid]))
                    if measured_average <= eps:
                        continue
                    scale = reconstruction_average / measured_average
                else:
                    weights = confidence[valid] / np.maximum(
                        np.square(meas.mag_std[valid]), eps
                    )
                    denominator = float(np.sum(weights * np.square(meas.mag[valid])))
                    if denominator <= eps:
                        continue
                    scale = float(
                        np.sum(weights * meas.mag[valid] * scale_anchor_mag[valid])
                        / denominator
                    )
                if not np.isfinite(scale) or scale < 0.0:
                    continue
            else:
                scale = self.fixed_scale

            # Uncertain or undetected pixels retain the current reconstruction;
            # high-SNR pixels approach the measured relative shape.
            y_source = confidence * meas.mag * scale + (1.0 - confidence) * current_mag
            ff.mag = smooth_overlap(
                x_target=ff.grid.f_pos,
                y_target=ff.mag,
                x_source=meas.freq,
                y_source=y_source,
                power=self.power,
                transition_width=self.transition_width,
                transition_width_left=(
                    None
                    if self.transition_width_left is None
                    else self.transition_width_left[i]
                ),
                transition_width_right=(
                    None
                    if self.transition_width_right is None
                    else self.transition_width_right[i]
                ),
            )


class SplineInterpolateMeasurementGaps(FrequencyConstraint):
    """Bridge separated measured bands with a slope-matched cubic spline.

    The measurement constraints run first.  This constraint then uses several
    already-constrained FFT bins at each side of a detector gap to estimate the
    local slopes, and fills only the unmeasured interval.  The resulting bridge
    exactly meets both bands and has matching endpoint slopes, avoiding a step
    between CRISP and IR while leaving all measured bins intact.
    """

    def __init__(
        self,
        measured: tuple[MeasuredFormFactor, ...],
        slope_fit_points: int = 5,
    ) -> None:
        super().__init__()
        if slope_fit_points < 2:
            raise ValueError("slope_fit_points must be at least 2")
        self.measured = measured
        self.slope_fit_points = slope_fit_points

    @staticmethod
    def _fit_slope(x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 2 or np.ptp(x) <= 0.0:
            return 0.0
        return float(np.polyfit(x - x[-1], y, deg=1)[0])

    def apply(self, ff: "FormFactor") -> None:
        ordered = sorted(self.measured, key=lambda measurement: measurement.freq[0])
        freq = ff.grid.f_pos

        for left, right in zip(ordered, ordered[1:]):
            gap_mask = (freq > left.freq[-1]) & (freq < right.freq[0])
            if not np.any(gap_mask):
                continue

            left_bins = freq[(freq >= left.freq[0]) & (freq <= left.freq[-1])]
            right_bins = freq[(freq >= right.freq[0]) & (freq <= right.freq[-1])]
            if left_bins.size < 2 or right_bins.size < 2:
                continue

            left_bins = left_bins[-self.slope_fit_points :]
            right_bins = right_bins[: self.slope_fit_points]
            left_values = np.interp(left_bins, freq, ff.mag)
            right_values = np.interp(right_bins, freq, ff.mag)
            x0, x1 = left_bins[-1], right_bins[0]
            y0, y1 = left_values[-1], right_values[0]
            left_slope = self._fit_slope(left_bins, left_values)
            right_slope = self._fit_slope(right_bins[::-1], right_values[::-1])

            spline = CubicHermiteSpline(
                [x0, x1],
                [y0, y1],
                [left_slope, right_slope],
            )
            ff.mag[gap_mask] = np.maximum(spline(freq[gap_mask]), 0.0)


class CutAfterNthZeroFromPeak(TimeConstraint):
    def __init__(
        self, n: int = 10, threshold: float = 0.0, keep_crossing: bool = False
    ):
        super().__init__()

        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite")

        self.n = n
        self.threshold = threshold
        self.keep_crossing = keep_crossing

    def apply(self, prof: "CurrentProfile") -> None:
        x = np.asarray(prof.values, dtype=float)

        if x.ndim != 1:
            raise ValueError("CutAfterNthZeroFromPeak only supports 1D profiles")
        if x.size == 0:
            return

        imax = int(np.argmax(x))
        y = x.copy()

        # On the left, find threshold hits before the peak and trim at the
        # n-th hit when moving outward. If keep_crossing is enabled, keep the
        # full crossing lobe by trimming at the next farther hit instead.
        left_hits = np.where(x[:imax] <= self.threshold)[0]
        if left_hits.size >= self.n:
            left_cut = int(left_hits[-self.n])
            if self.keep_crossing and left_hits.size > self.n:
                left_cut = int(left_hits[-(self.n + 1)])
            if not self.keep_crossing or left_hits.size > self.n:
                y[: left_cut + 1] = 0.0

        # On the right, find threshold hits after the peak and trim at the
        # n-th hit when moving outward. If keep_crossing is enabled, keep the
        # full crossing lobe by trimming at the next farther hit instead.
        right_hits = np.where(x[imax + 1 :] <= self.threshold)[0]
        if right_hits.size >= self.n:
            right_cut = int(imax + 1 + right_hits[self.n - 1])
            if self.keep_crossing and right_hits.size > self.n:
                right_cut = int(imax + 1 + right_hits[self.n])
            if not self.keep_crossing or right_hits.size > self.n:
                y[right_cut:] = 0.0

        prof.values = y
