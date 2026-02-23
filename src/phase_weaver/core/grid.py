from __future__ import annotations
from dataclasses import dataclass
from math import ceil, log2
import numpy as np

def next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << int(ceil(log2(n)))

@dataclass(frozen=True, slots=True)
class Grid:
    N: int
    dt: float

    def __post_init__(self):
        if self.N <= 0:
            raise ValueError("N must be positive.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")


    @classmethod
    def from_dt_tmax(cls, *, dt: float, t_max: float, snap_pow2: bool=True, min_N: int=64) -> "Grid":
        if dt <= 0 or t_max <= 0:
            raise ValueError("dt and t_max must be positive.")
        N_raw = int(ceil(2.0 * t_max / dt))
        N_raw = max(N_raw, min_N)
        N = next_pow2(N_raw) if snap_pow2 else N_raw
        if N % 2 == 1:
            N += 1
        return cls(N=int(N), dt=float(dt))

    @classmethod
    def from_df_fmax(cls, df: float, f_max: float, *, snap_pow2: bool=True, min_N: int=64) -> "Grid":
        if df <= 0 or f_max <= 0:
            raise ValueError("df and f_max must be positive.")
        N_raw = int(ceil(2.0 * f_max / df))
        N_raw = max(N_raw, min_N)
        N = next_pow2(N_raw) if snap_pow2 else N_raw
        if N % 2 == 1:
            N += 1
        dt = 1.0 / (N * df)
        return cls(N=int(N), dt=float(dt))

    @property
    def df(self) -> float:
        return 1.0 / (self.N * self.dt)

    @property
    def f_nyq(self) -> float:
        return 1.0 / (2.0 * self.dt)

    @property
    def t(self) -> np.ndarray:
        half = self.N // 2
        return self.dt * np.arange(-half, half, dtype=float)

    @property
    def f_pos(self) -> np.ndarray:
        half = self.N // 2
        return self.df * np.arange(0, half + 1, dtype=float)
            
    
    @property
    def f_shift(self) -> np.ndarray:
        half = self.N // 2
        return self.df * np.arange(-half, half, dtype=float)