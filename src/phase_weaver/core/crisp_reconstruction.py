from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from scipy.signal import savgol_filter

from .base import DCPhysicalRFFT, FormFactor, Grid, Profile
from .measurement import SquaredMagnitudeMeasurement
from .utils import fwhm_highest_peak

THZ_TO_HZ = 1e12
HZ_TO_THZ = 1e-12


@dataclass(slots=True, frozen=True)
class CrispReconstructionConfig:
    num_output_points: int = 1024
    max_frequency_thz: float = 2048.0
    min_input_points: int = 100
    min_charge_c: float = 10e-12
    max_iterations: int = 20
    detection_limit_fudge: float = 1.0
    modulus_error_fraction: float = 0.2
    min_ffsq_std_fraction: float = 0.2
    use_input_uncertainty: bool = True
    enforce_extrapolated_modulus: bool = True
    channels_to_remove: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.num_output_points < 128:
            raise ValueError("num_output_points must be at least 128")
        if self.num_output_points & (self.num_output_points - 1):
            raise ValueError("num_output_points must be a power of two")
        if not 60.0 <= self.max_frequency_thz <= 100000.0:
            raise ValueError("max_frequency_thz must be between 60 and 100000 THz")
        if not 0.0 <= self.detection_limit_fudge <= 10.0:
            raise ValueError("detection_limit_fudge must be between 0 and 10")
        if not 0.0 <= self.modulus_error_fraction <= 10.0:
            raise ValueError("modulus_error_fraction must be between 0 and 10")
        if not 0.0 <= self.min_ffsq_std_fraction <= 10.0:
            raise ValueError("min_ffsq_std_fraction must be between 0 and 10")
        channels = tuple(int(channel) for channel in self.channels_to_remove)
        if any(channel < 0 for channel in channels):
            raise ValueError("channels_to_remove must contain non-negative indices")
        object.__setattr__(self, "channels_to_remove", channels)

    @property
    def dt_s(self) -> float:
        return 0.5 / (self.max_frequency_thz * THZ_TO_HZ)

    @property
    def dnu_thz(self) -> float:
        return 2.0 * self.max_frequency_thz / self.num_output_points


@dataclass(slots=True)
class CrispPreprocessedInput:
    freq_thz: np.ndarray
    ffsq: np.ndarray
    ffsq_std: np.ndarray
    detection_limit: np.ndarray
    num_input_points: int
    num_filtered_input_points: int
    max_input_frequency_thz: float


@dataclass(slots=True)
class CrispDiagnostics:
    kramers_kronig_profile: np.ndarray
    kramers_kronig_phase: np.ndarray
    intermediate_frequencies_thz: np.ndarray
    intermediate_ffsq: np.ndarray
    intermediate_ffsq_std: np.ndarray
    interpolated_frequencies_thz: np.ndarray
    interpolated_ffabs: np.ndarray
    interpolated_ffabs_error: np.ndarray
    iteration_profiles: list[np.ndarray] = field(default_factory=list)
    num_input_points: int = 0
    num_filtered_input_points: int = 0
    max_input_frequency_thz: float = 0.0
    num_iterations: int = 0
    peak_current_a: float = 0.0
    fwhm_fs: float = float("nan")
    rms_width_fs: float = float("nan")
    skewness: float = float("nan")


@dataclass(slots=True)
class CrispReconstructionResult:
    profile: Profile
    form_factor: FormFactor
    diagnostics: CrispDiagnostics
    stop_reason: str


def preprocess_crisp_input(
    raw: SquaredMagnitudeMeasurement,
    config: CrispReconstructionConfig = CrispReconstructionConfig(),
) -> CrispPreprocessedInput:
    num_input_points = len(raw.freq)
    keep = np.ones(num_input_points, dtype=bool)
    for channel in config.channels_to_remove:
        if channel >= num_input_points:
            raise ValueError(
                f"channel index {channel} is outside the CRISP input array"
            )
        keep[channel] = False

    assert raw.mag_std is not None
    assert raw.detection_limit is not None
    freq_hz = raw.freq[keep]
    ffsq_input = raw.mag[keep]
    ffsq_std_input = raw.mag_std[keep]
    detection_limit_input = raw.detection_limit[keep]
    if len(freq_hz) < config.min_input_points:
        raise ValueError(
            f"CRISP input contains fewer than {config.min_input_points} frequency points"
        )

    freq_thz = freq_hz * HZ_TO_THZ
    order = np.argsort(freq_thz)
    freq_thz = freq_thz[order]
    ffsq = ffsq_input[order]
    ffsq_std = ffsq_std_input[order]
    detection_limit = detection_limit_input[order]

    if not np.isfinite(freq_thz[0]) or freq_thz[0] <= 0.0:
        raise ValueError("first CRISP frequency must be finite and greater than 0 THz")

    valid = np.ones_like(freq_thz, dtype=bool)
    valid[:2] = False
    valid[-2:] = False
    for i in range(1, len(freq_thz) - 1):
        bad = (
            not np.isfinite(ffsq[i])
            or ffsq[i] < detection_limit[i] * config.detection_limit_fudge
        )
        if bad:
            valid[i - 1 : i + 2] = False

    bad_run = 0
    cut_index: int | None = None
    for i, is_valid in enumerate(valid):
        if is_valid:
            bad_run = 0
        else:
            bad_run += 1
            if bad_run >= 16:
                cut_index = i - bad_run + 1
                break
    if cut_index is not None:
        valid[cut_index:] = False

    freq_thz = freq_thz[valid]
    ffsq = np.clip(ffsq[valid], 0.0, 1.0)
    ffsq_std = ffsq_std[valid]
    minimum_std = config.min_ffsq_std_fraction * ffsq
    ffsq_std = np.where(np.isfinite(ffsq_std), ffsq_std, minimum_std)
    ffsq_std = np.maximum(ffsq_std, minimum_std)
    detection_limit = detection_limit[valid]

    if len(freq_thz) < 5:
        raise ValueError("CRISP input contains fewer than 5 valid points")

    return CrispPreprocessedInput(
        freq_thz=freq_thz,
        ffsq=ffsq,
        ffsq_std=ffsq_std,
        detection_limit=detection_limit,
        num_input_points=num_input_points,
        num_filtered_input_points=len(freq_thz),
        max_input_frequency_thz=float(freq_thz[-1]),
    )


def fit_low_frequency_sigma(freq_thz: np.ndarray, ffsq: np.ndarray) -> float:
    if len(freq_thz) < 5:
        raise ValueError("at least 5 points are required for low-frequency fit")
    fit_slice = slice(None) if len(freq_thz) < 20 else slice(5, 20)
    x = freq_thz[fit_slice]
    y = ffsq[fit_slice]

    def objective(sigma: float) -> float:
        model = np.exp(-0.5 * (x / sigma) ** 2)
        return float(np.sum((y - model) ** 2))

    result = minimize_scalar(
        objective,
        bounds=(1e-3, 1000.0),
        method="bounded",
        options={"xatol": 1e-4},
    )
    if not result.success:
        raise ValueError("low-frequency Gaussian fit failed")
    return float(result.x)


@dataclass(slots=True, frozen=True)
class LowFrequencyGaussianFit:
    sigma_thz: float
    sigma_std_thz: float
    replacement_start: int
    fit_points: int


def fit_ocelot_low_frequency_gaussian(
    freq_thz: np.ndarray,
    ffsq: np.ndarray,
    ffsq_std: np.ndarray,
) -> LowFrequencyGaussianFit:
    """Choose and fit the low-frequency Gaussian as in Ocelot's 2019 path.

    Ocelot locates the first run of three channels below an adaptive magnitude
    threshold and fits the preceding low-frequency channels.  This version keeps
    that handoff rule but weights the fit by the supplied ``|F|^2`` uncertainty.
    """
    freq_thz = np.asarray(freq_thz, dtype=float)
    ffsq = np.asarray(ffsq, dtype=float)
    ffsq_std = np.asarray(ffsq_std, dtype=float)
    if len(freq_thz) < 5:
        raise ValueError("at least 5 points are required for low-frequency fit")

    ffabs = np.sqrt(np.clip(ffsq, 0.0, None))
    ffabs_std = np.zeros_like(ffabs)
    positive = ffabs > np.finfo(float).tiny
    ffabs_std[positive] = ffsq_std[positive] / (2.0 * ffabs[positive])
    ffabs_std = np.maximum(ffabs_std, 1e-12)

    threshold = 0.2
    while threshold < 0.9 and np.count_nonzero(ffabs > threshold) > 8:
        threshold += 0.1

    below = np.flatnonzero(ffabs < threshold)
    handoff: int | None = None
    if len(below) >= 3:
        for index in range(2, len(below)):
            if below[index] == below[index - 1] + 1 == below[index - 2] + 2:
                handoff = int(below[index] + 1)
                break

    if handoff is None:
        fit_points = min(20, len(freq_thz))
        replacement_start = 0
    elif handoff < 4:
        fit_points = min(60, len(freq_thz))
        replacement_start = 0
    else:
        fit_points = handoff
        replacement_start = handoff

    def gaussian(freq: np.ndarray, sigma: float) -> np.ndarray:
        return np.exp(-0.5 * (freq / sigma) ** 2)

    sigma_guess = max(float(freq_thz[min(fit_points - 1, len(freq_thz) - 1)]), 1.0)
    try:
        parameters, covariance = curve_fit(
            gaussian,
            freq_thz[:fit_points],
            ffabs[:fit_points],
            p0=(sigma_guess,),
            sigma=ffabs_std[:fit_points],
            absolute_sigma=True,
            bounds=(1e-3, 1000.0),
            maxfev=10_000,
        )
    except (RuntimeError, ValueError) as exc:
        raise ValueError("low-frequency Gaussian fit failed") from exc

    sigma_thz = float(parameters[0])
    variance = float(covariance[0, 0])
    sigma_std_thz = (
        float(np.sqrt(variance))
        if np.isfinite(variance) and variance > 0.0
        else 0.2 * sigma_thz
    )
    return LowFrequencyGaussianFit(
        sigma_thz=sigma_thz,
        sigma_std_thz=sigma_std_thz,
        replacement_start=replacement_start,
        fit_points=fit_points,
    )


def high_frequency_sigma(freq_last_thz: float, ffsq_last: float) -> float:
    safe_ffsq = max(float(ffsq_last), 1e-30)
    return float(np.sqrt(-(freq_last_thz**2) / np.log(safe_ffsq)))


def extrapolate_crisp_ffsq(
    preprocessed: CrispPreprocessedInput,
    config: CrispReconstructionConfig = CrispReconstructionConfig(),
) -> tuple[np.ndarray, np.ndarray, int]:
    frequencies, ffsq, _ffsq_std, index = extrapolate_crisp_ffsq_with_uncertainty(
        preprocessed,
        config,
    )
    return frequencies, ffsq, index


def extrapolate_crisp_ffsq_with_uncertainty(
    preprocessed: CrispPreprocessedInput,
    config: CrispReconstructionConfig = CrispReconstructionConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    low_fit = fit_ocelot_low_frequency_gaussian(
        preprocessed.freq_thz,
        preprocessed.ffsq,
        preprocessed.ffsq_std,
    )
    sigma_high = high_frequency_sigma(
        float(preprocessed.freq_thz[-1]), float(preprocessed.ffsq[-1])
    )

    dnu = config.dnu_thz
    first_frequency = float(preprocessed.freq_thz[low_fit.replacement_start])
    dnu_front = dnu
    while first_frequency / dnu_front < 4.0:
        dnu_front *= 0.5

    front_freq = np.arange(0.0, first_frequency, dnu_front, dtype=float)
    front_ffabs = np.exp(-0.5 * (front_freq / low_fit.sigma_thz) ** 2)
    front_ffsq = front_ffabs**2
    front_derivative = (
        front_ffabs * front_freq**2 / low_fit.sigma_thz**3
    )
    front_ffsq_std = np.abs(2.0 * front_ffabs * front_derivative) * (
        low_fit.sigma_std_thz
    )

    preserved = slice(low_fit.replacement_start, None)
    freqs = [front_freq, preprocessed.freq_thz[preserved]]
    values = [front_ffsq, preprocessed.ffsq[preserved]]
    uncertainties = [front_ffsq_std, preprocessed.ffsq_std[preserved]]
    idx_extrap_start = len(front_freq) + len(preprocessed.freq_thz[preserved])

    tail_start = float(preprocessed.freq_thz[-1])
    tail_step = float(preprocessed.freq_thz[-1] - preprocessed.freq_thz[-2])
    tail_first = tail_start + tail_step * np.arange(1, 11, dtype=float)
    regular_start = tail_first[-1] + dnu
    tail_regular = np.arange(regular_start, config.max_frequency_thz, dnu, dtype=float)
    tail_freq = np.concatenate([tail_first, tail_regular])
    tail_ffsq = np.exp(-(tail_freq**2) / sigma_high**2)
    freqs.append(tail_freq)
    values.append(tail_ffsq)
    uncertainties.append(np.zeros_like(tail_ffsq))

    return (
        np.concatenate(freqs),
        np.concatenate(values),
        np.concatenate(uncertainties),
        idx_extrap_start,
    )


def smooth_intermediate_ffsq(ffsq: np.ndarray) -> np.ndarray:
    ffsq = np.asarray(ffsq, dtype=float)
    if len(ffsq) < 9:
        return np.clip(ffsq.copy(), 0.0, 1.0)
    smoothed = np.asarray(
        savgol_filter(ffsq, window_length=9, polyorder=3, mode="interp"),
        dtype=float,
    )
    smoothed[:4] = ffsq[:4]
    smoothed[-4:] = ffsq[-4:]
    return np.clip(smoothed, 0.0, 1.0)


def interpolate_crisp_ffabs(
    frequencies_thz: np.ndarray,
    ffsq: np.ndarray,
    max_input_frequency_thz: float,
    config: CrispReconstructionConfig = CrispReconstructionConfig(),
) -> tuple[np.ndarray, np.ndarray, int]:
    half = config.num_output_points // 2
    target_freq = config.dnu_thz * np.arange(half + 1, dtype=float)
    interp_ffsq = np.interp(target_freq, frequencies_thz, ffsq)
    ffabs = np.sqrt(np.clip(interp_ffsq, 0.0, None))
    upper = np.flatnonzero(target_freq > max_input_frequency_thz)
    idx_upper = int(upper[0]) if upper.size else len(target_freq)
    return target_freq, ffabs, idx_upper


def interpolate_crisp_ffabs_error(
    target_frequencies_thz: np.ndarray,
    ffabs: np.ndarray,
    preprocessed: CrispPreprocessedInput,
    *,
    intermediate_frequencies_thz: np.ndarray | None = None,
    intermediate_ffsq_std: np.ndarray | None = None,
) -> np.ndarray:
    """Propagate measured ``|F|^2`` standard deviations to ``|F|``.

    When an extrapolated uncertainty scale is supplied, it is interpolated along
    with the form factor.  Otherwise only measured points contribute.  In either
    case, first-order propagation gives ``sigma_|F| =
    sigma_|F|^2 / (2 |F|)``.
    """
    target_frequencies_thz = np.asarray(target_frequencies_thz, dtype=float)
    ffabs = np.asarray(ffabs, dtype=float)
    if target_frequencies_thz.shape != ffabs.shape:
        raise ValueError("target frequencies and |F| must have equal shape")

    if (intermediate_frequencies_thz is None) != (intermediate_ffsq_std is None):
        raise ValueError(
            "intermediate frequencies and uncertainties must be supplied together"
        )
    if intermediate_frequencies_thz is None:
        valid = (
            (target_frequencies_thz >= preprocessed.freq_thz[0])
            & (target_frequencies_thz <= preprocessed.freq_thz[-1])
        )
        interpolated_ffsq_std = np.interp(
            target_frequencies_thz[valid],
            preprocessed.freq_thz,
            preprocessed.ffsq_std,
        )
    else:
        intermediate_frequencies_thz = np.asarray(
            intermediate_frequencies_thz,
            dtype=float,
        )
        intermediate_ffsq_std = np.asarray(intermediate_ffsq_std, dtype=float)
        if intermediate_frequencies_thz.shape != intermediate_ffsq_std.shape:
            raise ValueError(
                "intermediate frequencies and uncertainties must have equal shape"
            )
        valid = (target_frequencies_thz >= intermediate_frequencies_thz[0]) & (
            target_frequencies_thz <= intermediate_frequencies_thz[-1]
        )
        interpolated_ffsq_std = np.interp(
            target_frequencies_thz[valid],
            intermediate_frequencies_thz,
            intermediate_ffsq_std,
        )

    error = np.zeros_like(ffabs)
    positive = ffabs[valid] > np.finfo(float).tiny
    measured_error = np.zeros_like(interpolated_ffsq_std)
    measured_error[positive] = (
        interpolated_ffsq_std[positive] / (2.0 * ffabs[valid][positive])
    )
    error[valid] = measured_error
    return error


def kramers_kronig_phase(
    frequencies_thz: np.ndarray,
    ffabs: np.ndarray,
    dnu_thz: float,
) -> np.ndarray:
    frequencies_thz = np.asarray(frequencies_thz, dtype=float)
    ffabs = np.asarray(ffabs, dtype=float)
    nonpositive = np.flatnonzero(ffabs <= 0.0)
    num_nonzero = int(nonpositive[0]) if nonpositive.size else len(ffabs)
    if num_nonzero < 8:
        raise ValueError("at least 8 non-zero form-factor points are required")

    phase = np.zeros_like(ffabs)
    f = frequencies_thz[:num_nonzero]
    mag = ffabs[:num_nonzero]
    for i in range(num_nonzero):
        denom = f[i] ** 2 - f**2
        ratio = mag / mag[i]
        mask = (np.arange(num_nonzero) != i) & (ratio > 0.0) & (denom != 0.0)
        phase[i] = (2.0 / np.pi) * f[i] * dnu_thz * np.sum(
            np.log(ratio[mask]) / denom[mask]
        )
    if num_nonzero < len(ffabs):
        phase[num_nonzero:] = phase[num_nonzero - 1]
    return phase


def center_maximum(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    target = (len(out) - 1) // 2
    peak = int(np.argmax(out))
    return np.roll(out, target - peak)


def isolate_positive_maximum(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.zeros_like(x)
    peak = int(np.argmax(x))
    if x[peak] <= 0.0:
        return out

    left = peak
    while left > 0 and x[left - 1] > 0.0:
        left -= 1
    right = peak
    while right < len(x) - 1 and x[right + 1] > 0.0:
        right += 1
    out[left : right + 1] = x[left : right + 1]
    return out


def build_crisp_full_spectrum(ffabs: np.ndarray, phase: np.ndarray) -> np.ndarray:
    ffabs = np.asarray(ffabs, dtype=float)
    phase = np.asarray(phase, dtype=float)
    if ffabs.shape != phase.shape or ffabs.ndim != 1:
        raise ValueError("|F| and phase must be equal-length 1D arrays")
    if len(ffabs) < 2:
        raise ValueError("at least DC and Nyquist bins are required")

    # ``ffabs`` contains the non-negative rFFT bins, including DC and Nyquist.
    # Both self-conjugate bins must be real; only the interior bins are mirrored.
    positive = ffabs * np.exp(1j * phase)
    positive[0] = complex(ffabs[0], 0.0)
    positive[-1] = complex(ffabs[-1], 0.0)
    return np.concatenate([positive, np.conj(positive[-2:0:-1])])


def _replace_modulus_outside_band(
    spectrum: np.ndarray,
    ref_ffabs: np.ndarray,
    ref_error: np.ndarray,
    idx_upper_extrapolation: int,
) -> int:
    replaced = 0
    upper = min(idx_upper_extrapolation, len(ref_ffabs))
    for i in range(upper):
        if abs(abs(spectrum[i]) - ref_ffabs[i]) > ref_error[i]:
            spectrum[i] = ref_ffabs[i] * np.exp(1j * np.angle(spectrum[i]))
            replaced += 1
        if i > 0:
            j = len(spectrum) - i
            if abs(abs(spectrum[j]) - ref_ffabs[i]) > ref_error[i]:
                spectrum[j] = ref_ffabs[i] * np.exp(1j * np.angle(spectrum[j]))
                replaced += 1
    return replaced


def _profile_stats(time_fs: np.ndarray, current_a: np.ndarray) -> tuple[float, float, float, float]:
    peak = float(np.max(current_a)) if current_a.size else 0.0
    fwhm = fwhm_highest_peak(time_fs, current_a)
    fwhm_fs = float(fwhm.fwhm) if fwhm is not None else float("nan")
    total = float(np.sum(current_a))
    if total <= 0.0:
        return peak, fwhm_fs, float("nan"), float("nan")
    mean = float(np.sum(time_fs * current_a) / total)
    rms = float(np.sqrt(np.sum((time_fs - mean) ** 2 * current_a) / total))
    skew = float(
        np.sum((time_fs - mean) ** 3 * current_a) / total / rms**3
    ) if rms > 0.0 else float("nan")
    return peak, fwhm_fs, rms, skew


class CrispReconstruction:
    name = "CRISP"

    def __init__(
        self,
        data: SquaredMagnitudeMeasurement,
        config: CrispReconstructionConfig = CrispReconstructionConfig(),
    ) -> None:
        self.data = data
        if data.max_frequency_thz is not None:
            config = replace(config, max_frequency_thz=data.max_frequency_thz)
        self.config = config
        self.charge_c = data.charge_c if data.charge_c is not None else 250e-12
        self.last_iterations = 0
        self.last_stop_reason = "not_run"
        self.diagnostics: CrispDiagnostics | None = None

    def run(self) -> CrispReconstructionResult:
        if self.charge_c < self.config.min_charge_c:
            raise ValueError("no_beam")

        pre = preprocess_crisp_input(self.data, self.config)
        inter_freq, inter_ffsq, inter_ffsq_std, _idx_extrap_start = (
            extrapolate_crisp_ffsq_with_uncertainty(
                pre,
                self.config,
            )
        )
        inter_ffsq = smooth_intermediate_ffsq(inter_ffsq)
        interp_freq, ffabs, idx_upper = interpolate_crisp_ffabs(
            inter_freq, inter_ffsq, pre.max_input_frequency_thz, self.config
        )
        if self.config.use_input_uncertainty:
            ref_error = interpolate_crisp_ffabs_error(
                interp_freq,
                ffabs,
                pre,
                intermediate_frequencies_thz=inter_freq,
                intermediate_ffsq_std=inter_ffsq_std,
            )
        else:
            ref_error = self.config.modulus_error_fraction * ffabs
        phase = kramers_kronig_phase(interp_freq, ffabs, self.config.dnu_thz)

        full_spectrum = build_crisp_full_spectrum(ffabs, phase)
        profile = center_maximum(np.fft.ifft(full_spectrum).real)
        kk_profile = profile.copy()
        iteration_profiles: list[np.ndarray] = []
        stop_reason = "max_iter"
        constraint_upper = (
            len(ffabs)
            if self.config.enforce_extrapolated_modulus
            else idx_upper
        )

        for iteration in range(1, self.config.max_iterations + 1):
            profile = isolate_positive_maximum(profile)
            if iteration <= 4:
                iteration_profiles.append(profile.copy())

            spectrum = np.fft.fft(profile)
            if spectrum[0] != 0.0:
                spectrum *= 1.0 / spectrum[0]
            replaced = _replace_modulus_outside_band(
                spectrum, ffabs, ref_error, constraint_upper
            )
            profile = center_maximum(np.fft.ifft(spectrum).real)
            self.last_iterations = iteration
            if replaced == 0:
                stop_reason = "converged_no_replacements"
                break

        profile = isolate_positive_maximum(profile)
        profile_sum = float(np.sum(profile))
        if profile_sum <= 0.0:
            raise ValueError("invalid_input: reconstructed profile has zero area")
        current_a = profile * ((self.charge_c / self.config.dt_s) / profile_sum)

        grid = Grid(N=self.config.num_output_points, dt=self.config.dt_s)
        density = current_a / self.charge_c
        prof = Profile(grid=grid, values=density, charge=self.charge_c)
        ff_recon = prof.to_form_factor(
            transform=DCPhysicalRFFT(unwrap_phase=True, dc_normalize=True)
        )
        time_fs = grid.t * 1e15
        peak, fwhm_fs, rms_width_fs, skewness = _profile_stats(time_fs, current_a)

        diagnostics = CrispDiagnostics(
            kramers_kronig_profile=kk_profile,
            kramers_kronig_phase=phase,
            intermediate_frequencies_thz=inter_freq,
            intermediate_ffsq=inter_ffsq,
            intermediate_ffsq_std=inter_ffsq_std,
            interpolated_frequencies_thz=interp_freq,
            interpolated_ffabs=ffabs,
            interpolated_ffabs_error=ref_error,
            iteration_profiles=iteration_profiles,
            num_input_points=pre.num_input_points,
            num_filtered_input_points=pre.num_filtered_input_points,
            max_input_frequency_thz=pre.max_input_frequency_thz,
            num_iterations=self.last_iterations,
            peak_current_a=peak,
            fwhm_fs=fwhm_fs,
            rms_width_fs=rms_width_fs,
            skewness=skewness,
        )
        self.last_stop_reason = stop_reason
        self.diagnostics = diagnostics
        return CrispReconstructionResult(
            profile=prof,
            form_factor=ff_recon,
            diagnostics=diagnostics,
            stop_reason=stop_reason,
        )
