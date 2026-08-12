from __future__ import annotations

from dataclasses import dataclass

import numpy as np


C_NM_THZ = 299792.458
OCEAN_NUM_PIXELS = 512
OCEAN_MIN_WAVELENGTH_NM = 896.0
OCEAN_MAX_WAVELENGTH_NM = 2515.0


@dataclass(frozen=True, slots=True)
class OceanSimulationConfig:
    """Configuration for the uncalibrated Ocean/NIR forward model.

    The count-domain defaults are deliberately generic.  They reproduce the
    important measurement effects without claiming an absolute calibration of
    the installed optical beamline.
    """

    n_shots: int = 1
    seed: int = 0
    peak_signal_counts: float = 30_000.0
    dark_counts: float = 500.0
    read_noise_std_counts: float = 15.0
    adc_max_counts: float = 65_535.0
    detection_sigma: float = 3.0
    normalization_percentile: float = 95.0

    def __post_init__(self) -> None:
        if self.n_shots < 1:
            raise ValueError("n_shots must be at least 1")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name in (
            "peak_signal_counts",
            "dark_counts",
            "read_noise_std_counts",
            "adc_max_counts",
            "detection_sigma",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.peak_signal_counts <= 0.0 or self.adc_max_counts <= 0.0:
            raise ValueError("peak_signal_counts and adc_max_counts must be positive")
        if not 0.0 < self.normalization_percentile <= 100.0:
            raise ValueError("normalization_percentile must be in (0, 100]")


@dataclass(frozen=True, slots=True)
class OceanSimulationResult:
    wavelength_nm: np.ndarray
    freq_hz: np.ndarray
    intensity_counts: np.ndarray
    intensity_std_counts: np.ndarray
    ffabs_relative: np.ndarray
    ffabs_std: np.ndarray
    ffabs_detection_limit: np.ndarray


def simulate_ocean_measurement(
    formfactor_freq_hz: np.ndarray,
    formfactor_mag: np.ndarray,
    config: OceanSimulationConfig | None = None,
) -> OceanSimulationResult:
    """Simulate an Ocean/NIR spectrum and recover a relative ``|F|`` shape.

    The model samples on 512 wavelength pixels, converts ``|F|^2`` to arbitrary
    detector counts, adds photon and electronic noise in the count domain,
    averages shots, subtracts the known model dark level, and applies the same
    square-root/95th-percentile normalization used for loaded Ocean data.
    """

    config = config or OceanSimulationConfig()
    source_freq = np.asarray(formfactor_freq_hz, dtype=float)
    source_mag = np.asarray(formfactor_mag, dtype=float)
    if source_freq.ndim != 1 or source_mag.ndim != 1:
        raise ValueError("form-factor frequency and magnitude must be 1D")
    if source_freq.shape != source_mag.shape or source_freq.size < 2:
        raise ValueError("form-factor arrays must have equal shape and at least 2 points")
    if np.any(~np.isfinite(source_freq)) or np.any(~np.isfinite(source_mag)):
        raise ValueError("form-factor arrays must contain only finite values")

    order = np.argsort(source_freq)
    source_freq = source_freq[order]
    source_mag = np.maximum(source_mag[order], 0.0)

    wavelength_nm = np.linspace(
        OCEAN_MIN_WAVELENGTH_NM,
        OCEAN_MAX_WAVELENGTH_NM,
        OCEAN_NUM_PIXELS,
    )
    freq_hz = (C_NM_THZ / wavelength_nm) * 1e12
    ideal_mag = np.interp(freq_hz, source_freq, source_mag, left=0.0, right=0.0)
    expected_signal = config.peak_signal_counts * np.square(ideal_mag)
    expected_total = np.clip(
        expected_signal + config.dark_counts,
        0.0,
        config.adc_max_counts,
    )

    rng = np.random.default_rng(config.seed)
    accumulated = np.zeros_like(expected_total)
    for _ in range(config.n_shots):
        photon_counts = rng.poisson(expected_total).astype(float)
        electronic_noise = rng.normal(
            0.0,
            config.read_noise_std_counts,
            size=expected_total.shape,
        )
        accumulated += np.clip(
            photon_counts + electronic_noise,
            0.0,
            config.adc_max_counts,
        )

    averaged_counts = accumulated / config.n_shots
    intensity_counts = np.maximum(averaged_counts - config.dark_counts, 0.0)
    intensity_std = np.sqrt(
        expected_total + config.read_noise_std_counts**2
    ) / np.sqrt(config.n_shots)
    intensity_detection_limit = config.detection_sigma * intensity_std

    mag_like = np.sqrt(intensity_counts)
    normalization = float(
        np.percentile(mag_like, config.normalization_percentile)
    )
    if not np.isfinite(normalization) or normalization <= 0.0:
        normalization = 1.0

    ffabs_relative = np.clip(mag_like / normalization, 0.0, 1.0)
    propagation_floor = np.maximum(intensity_counts, intensity_detection_limit)
    ffabs_std = (
        0.5
        * intensity_std
        / np.sqrt(np.maximum(propagation_floor, np.finfo(float).eps))
        / normalization
    )
    ffabs_detection_limit = (
        np.sqrt(np.maximum(intensity_detection_limit, 0.0)) / normalization
    )

    # Frequency is more convenient in increasing order throughout PhaseWeaver.
    reverse = slice(None, None, -1)
    return OceanSimulationResult(
        wavelength_nm=wavelength_nm[reverse].copy(),
        freq_hz=freq_hz[reverse].copy(),
        intensity_counts=intensity_counts[reverse].copy(),
        intensity_std_counts=intensity_std[reverse].copy(),
        ffabs_relative=ffabs_relative[reverse].copy(),
        ffabs_std=ffabs_std[reverse].copy(),
        ffabs_detection_limit=ffabs_detection_limit[reverse].copy(),
    )
