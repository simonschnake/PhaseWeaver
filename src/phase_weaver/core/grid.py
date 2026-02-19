from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
import numpy as np


def next_pow2(n: int) -> int:
    """Small helper: smallest power-of-two >= n (n>=1)."""
    if n <= 1:
        return 1
    return 1 << int(ceil(log2(n)))


@dataclass(frozen=True, slots=True)
class Grid:
    """
    Frequency/time grid with centered time axis.

    Stores:
      - df: frequency spacing (Hz)
      - f_max_req: requested / implied max positive frequency (Hz)
      - N: number of samples (even)
    """

    df: float
    f_max_req: float
    N: int

    # -----------------------------
    # Constructors
    # -----------------------------
    @classmethod
    def from_df_fmax(
        cls,
        df: float,
        f_max: float,
        *,
        snap_pow2: bool = True,
        min_N: int = 64,
    ) -> "Grid":
        """
        Construct grid from desired df and desired max positive frequency f_max.

        We want: (N/2)*df >= f_max  =>  N >= 2*f_max/df
        """
        if df <= 0 or f_max <= 0:
            raise ValueError("df and f_max must be positive.")
        if min_N < 2:
            raise ValueError("min_N must be >= 2.")

        N_raw = int(ceil(2.0 * f_max / df))
        N_raw = max(N_raw, min_N)

        N = next_pow2(N_raw) if snap_pow2 else N_raw
        if N % 2 == 1:
            N += 1

        # With df fixed, dt is determined by N: dt = 1/(N*df)
        # With dt fixed, Nyquist = 1/(2*dt) = N*df/2
        # Here we can define the achieved Nyquist as f_max_req:
        dt = 1.0 / (N * df)
        f_max_req = 1.0 / (2.0 * dt)  # == (N/2)*df

        return cls(float(df), float(f_max_req), int(N))

    @classmethod
    def from_dt_tmax(
        cls,
        *,
        dt: float,
        t_max: float,
        snap_pow2: bool = True,
        min_N: int = 64,
    ) -> "Grid":
        """
        Construct grid from desired time spacing dt and desired half-width t_max.

        Centered axis is: t = dt * [-(N/2) ... (N/2 - 1)]
        We enforce: (N/2)*dt >= t_max  =>  N >= 2*t_max/dt
        """
        if dt <= 0 or t_max <= 0:
            raise ValueError("dt and t_max must be positive.")
        if min_N < 2:
            raise ValueError("min_N must be >= 2.")

        N_raw = int(ceil(2.0 * t_max / dt))
        N_raw = max(N_raw, min_N)

        N = next_pow2(N_raw) if snap_pow2 else N_raw
        if N % 2 == 1:
            N += 1

        df = 1.0 / (N * dt)
        f_max_req = 1.0 / (2.0 * dt)  # Nyquist implied by dt

        return cls(float(df), float(f_max_req), int(N))

    # -----------------------------
    # Derived properties
    # -----------------------------
    @property
    def dt(self) -> float:
        return 1.0 / (self.N * self.df)

    @property
    def t(self) -> np.ndarray:
        """
        Centered time axis: length N, covers approximately [-N/2*dt, (N/2-1)*dt]
        """
        half = self.N // 2
        return self.dt * np.arange(-half, half, dtype=float)

    @property
    def f_pos(self) -> np.ndarray:
        """
        One-sided frequency axis: [0 .. Nyquist], length N/2 + 1
        """
        half = self.N // 2
        return self.df * np.arange(0, half + 1, dtype=float)

    @property
    def f_shift(self) -> np.ndarray:
        """
        Centered frequency axis (FFT-shifted ordering): length N
        """
        half = self.N // 2
        return self.df * np.arange(-half, half, dtype=float)