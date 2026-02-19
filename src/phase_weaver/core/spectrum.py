from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from phase_weaver.core.grid import Grid


def fft_pos(x_t: NDArray[np.floating] | NDArray[np.complexfloating], g: Grid) -> NDArray[np.complexfloating]:
    """
    One-sided FFT (0..Nyquist) consistent with centered time grid.

    Convention:
      - time signal x_t is on g.t (centered)
      - we shift it so t=0 is at index 0, then do rfft/fft
      - scale by dt to preserve units (continuous FT approx)

    Returns array length N/2+1 matching g.f_pos.
    """
    x = np.asarray(x_t)
    x_shift = np.fft.ifftshift(x)

    if np.iscomplexobj(x_shift):
        X = np.fft.fft(x_shift)[: (g.N // 2 + 1)]
    else:
        X = np.fft.rfft(x_shift)

    return X * g.dt


def unwrap_phase(phi: NDArray[np.floating]) -> NDArray[np.floating]:
    """Unwrap phase (radians). Thin wrapper for clarity."""
    return np.unwrap(phi)

def fft_pos_centered(x_centered: np.ndarray, g: Grid) -> np.ndarray:
    """
    One-sided FFT: X_pos = rfft(ifftshift(x)) * dt
    """
    x_shift = np.fft.ifftshift(np.asarray(x_centered))
    X = np.fft.rfft(x_shift) * g.dt
    return X


def mag_phase_from_time(x_centered: np.ndarray, g: Grid, eps: float = 1e-30):
    X = fft_pos_centered(x_centered, g)
    mag = np.abs(X)
    mag = np.maximum(mag, eps)
    phase = np.unwrap(np.angle(X))
    return mag, phase, X


def normalize_mag_dc_to_one(mag: np.ndarray) -> np.ndarray:
    mag = np.asarray(mag, dtype=float)
    if mag[0] <= 0:
        mag2 = mag.copy()
        mag2[0] = 1.0
        return mag2
    return mag / mag[0]