from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from phase_weaver.core.grid import Grid
from phase_weaver.core.reconstruction import reconstruct_current_from_mag
from phase_weaver.core.spectrum import mag_phase_from_time, normalize_mag_dc_to_one
from phase_weaver.core.time_constraints import normalize_center_nonneg_density


@dataclass(frozen=True, slots=True)
class MagnitudePhaseRepresentation:
    density: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray


def profile_to_magnitude_phase(
    profile: np.ndarray,
    g: Grid,
    *,
    normalize_density: bool = True,
    normalize_dc: bool = True,
    eps: float = 1e-30,
) -> MagnitudePhaseRepresentation:
    """
    Convert a time-domain profile on `g.t` into one-sided magnitude + phase.
    """
    profile_arr = np.asarray(profile, dtype=float)

    if normalize_density:
        density = normalize_center_nonneg_density(profile_arr, g, eps=eps)
    else:
        density = profile_arr

    magnitude, phase, _ = mag_phase_from_time(density, g, eps=eps)
    if normalize_dc:
        magnitude = normalize_mag_dc_to_one(magnitude)

    return MagnitudePhaseRepresentation(density=density, magnitude=magnitude, phase=phase)


def magnitude_phase_to_density(
    magnitude: np.ndarray,
    phase: np.ndarray,
    g: Grid,
    *,
    apply_constraints: bool = True,
    eps: float = 1e-30,
) -> np.ndarray:
    """
    Convert one-sided magnitude + phase back to a centered time-domain density.
    """
    magnitude_arr = np.asarray(magnitude, dtype=float)
    phase_arr = np.asarray(phase, dtype=float)
    expected_len = g.N // 2 + 1

    if magnitude_arr.shape != (expected_len,):
        raise ValueError(f"magnitude must have length N/2+1 = {expected_len}")
    if phase_arr.shape != (expected_len,):
        raise ValueError(f"phase must have length N/2+1 = {expected_len}")

    spectrum_pos = np.maximum(magnitude_arr, eps) * np.exp(1j * phase_arr)
    density_shifted = np.fft.irfft(spectrum_pos / g.dt, n=g.N)
    density = np.fft.fftshift(density_shifted)

    if apply_constraints:
        density = normalize_center_nonneg_density(density, g, eps=eps)
    return density


def magnitude_phase_to_current(
    magnitude: np.ndarray,
    phase: np.ndarray,
    g: Grid,
    *,
    charge_C: float,
    apply_constraints: bool = True,
    eps: float = 1e-30,
) -> np.ndarray:
    density = magnitude_phase_to_density(
        magnitude,
        phase,
        g,
        apply_constraints=apply_constraints,
        eps=eps,
    )
    return charge_C * density


def reconstruct_current_from_magnitude(
    g: Grid,
    magnitude: np.ndarray,
    *,
    charge_C: float = 250e-12,
    n_iter: int = 60,
    init_phase: str = "minphase",
    enforce_dc_one: bool = True,
    enforce_nonneg: bool = True,
    normalize_area_to_one: bool = True,
    center: bool = True,
    eps_mag: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Magnitude-only reconstruction wrapper to keep transfer logic centralized.
    """
    return reconstruct_current_from_mag(
        g,
        magnitude,
        charge_C=charge_C,
        n_iter=n_iter,
        init_phase=init_phase,
        enforce_dc_one=enforce_dc_one,
        enforce_nonneg=enforce_nonneg,
        normalize_area_to_one=normalize_area_to_one,
        center=center,
        eps_mag=eps_mag,
    )


def relative_l2_error(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-30) -> float:
    reference_arr = np.asarray(reference, dtype=float)
    estimate_arr = np.asarray(estimate, dtype=float)

    denom = float(np.linalg.norm(reference_arr))
    if denom <= eps:
        return float(np.linalg.norm(estimate_arr))
    return float(np.linalg.norm(reference_arr - estimate_arr) / denom)
