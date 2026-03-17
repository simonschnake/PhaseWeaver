from dataclasses import dataclass
import numpy as np


@dataclass
class MeasuredFormFactor:
    freq: np.ndarray
    mag: np.ndarray

    def __post_init__(self):
        self.freq = np.asarray(self.freq, dtype=float)
        self.mag = np.asarray(self.mag, dtype=float)

        if self.freq.ndim != 1 or self.mag.ndim != 1:
            raise ValueError("freq and mag must be 1D arrays")
        if len(self.freq) != len(self.mag):
            raise ValueError("freq and mag must have the same length")
        if len(self.freq) == 0:
            raise ValueError("freq and mag must not be empty")

        order = np.argsort(self.freq)
        self.freq = self.freq[order]
        self.mag = self.mag[order]

        if np.any(~np.isfinite(self.freq)) or np.any(~np.isfinite(self.mag)):
            raise ValueError("freq and mag must contain only finite values")
