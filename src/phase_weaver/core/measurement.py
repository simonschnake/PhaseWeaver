from dataclasses import dataclass
import numpy as np


@dataclass
class MeasuredFormFactor:
    freq: np.ndarray
    mag: np.ndarray
    mag_std: np.ndarray | None = None
    detection_limit: np.ndarray | None = None

    def __post_init__(self):
        self.freq = np.asarray(self.freq, dtype=float)
        self.mag = np.asarray(self.mag, dtype=float)

        optional = {}
        for name in ("mag_std", "detection_limit"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=float)
                if array.ndim != 1 or array.shape != self.mag.shape:
                    raise ValueError(f"{name} must be a 1D array matching mag")
                if np.any(~np.isfinite(array)) or np.any(array < 0.0):
                    raise ValueError(f"{name} must be finite and non-negative")
                optional[name] = array

        if self.freq.ndim != 1 or self.mag.ndim != 1:
            raise ValueError("freq and mag must be 1D arrays")
        if len(self.freq) != len(self.mag):
            raise ValueError("freq and mag must have the same length")
        if len(self.freq) == 0:
            raise ValueError("freq and mag must not be empty")

        order = np.argsort(self.freq)
        self.freq = self.freq[order]
        self.mag = self.mag[order]
        for name, value in optional.items():
            setattr(self, name, value[order])

        if np.any(~np.isfinite(self.freq)) or np.any(~np.isfinite(self.mag)):
            raise ValueError("freq and mag must contain only finite values")
