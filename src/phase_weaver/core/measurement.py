from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, TypeVar

import numpy as np

_EnumT = TypeVar("_EnumT", bound=Enum)


class MeasurementKind(Enum):
    """Which instrument/service produced a measurement."""

    UNKNOWN = "unknown"
    CRISP = "crisp"
    OCEAN_NIR = "ocean_nir"
    INFRARED = "infrared"


class CalibrationStatus(Enum):
    """Calibration status of a measurement.

    Structural honesty guarantee: ``RELATIVE`` data is never promoted to
    ``ABSOLUTE`` form-factor semantics by the reconstruction layer.
    """

    UNKNOWN = "unknown"
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


def _coerce_enum(value: object, enum_cls: type[_EnumT]) -> _EnumT:
    if isinstance(value, enum_cls):
        return value
    return enum_cls(str(value))


def _validated_arrays(
    freq: np.ndarray,
    mag: np.ndarray,
    mag_std: np.ndarray | None,
    detection_limit: np.ndarray | None,
    *,
    allow_raw: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Validate and normalise the (freq, mag, mag_std, detection_limit) arrays.

    Returns arrays with ``freq`` sorted ascending and optional arrays shaped to
    match ``mag``. When ``allow_raw`` is true the ``mag`` values are permitted
    to be non-finite or negative (raw ``|F|^2`` that the preprocessing layer is
    responsible for masking); ``freq`` must always be finite.
    """
    freq = np.asarray(freq, dtype=float)
    mag = np.asarray(mag, dtype=float)

    if freq.ndim != 1 or mag.ndim != 1:
        raise ValueError("freq and mag must be 1D arrays")
    if len(freq) != len(mag):
        raise ValueError("freq and mag must have the same length")
    if len(freq) == 0:
        raise ValueError("freq and mag must not be empty")
    if np.any(~np.isfinite(freq)):
        raise ValueError("freq must contain only finite values")
    if not allow_raw:
        if np.any(~np.isfinite(mag)):
            raise ValueError("mag must contain only finite values")
        if np.any(mag < 0.0):
            raise ValueError("mag must be non-negative")

    optional: dict[str, np.ndarray | None] = {"mag_std": None, "detection_limit": None}
    for name, value in (("mag_std", mag_std), ("detection_limit", detection_limit)):
        if value is None:
            optional[name] = None
            continue
        array = np.asarray(value, dtype=float)
        if array.ndim != 1 or array.shape != mag.shape:
            raise ValueError(f"{name} must be a 1D array matching mag")
        if np.any(~np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
        optional[name] = array

    order = np.argsort(freq)
    freq = freq[order]
    mag = mag[order]
    mag_std = optional["mag_std"]
    detection_limit = optional["detection_limit"]
    if mag_std is not None:
        mag_std = mag_std[order]
    if detection_limit is not None:
        detection_limit = detection_limit[order]
    return freq, mag, mag_std, detection_limit


@dataclass
class Measurement:
    """A frequency-domain measurement with explicit calibration provenance.

    Semantics: :attr:`mag` is the **magnitude** ``|F|``. Calibration status and
    instrument ``kind`` are first-class so relative-shape data can never be
    silently promoted to absolute form-factor semantics.

    The squared-magnitude representation (``|F|^2``) is carried by the
    :class:`SquaredMagnitudeMeasurement` subtype with an explicit conversion
    (``.to_squared()`` / ``.as_magnitude()``) rather than ad hoc sqrt/squares.
    """

    freq: np.ndarray
    mag: np.ndarray
    mag_std: np.ndarray | None = None
    detection_limit: np.ndarray | None = None
    kind: MeasurementKind = MeasurementKind.UNKNOWN
    calibration: CalibrationStatus = CalibrationStatus.UNKNOWN
    label: str = ""
    source: str = ""
    charge_c: float | None = None
    max_frequency_thz: float | None = None
    use_in_absolute_constraint: bool = True
    use_in_relative_constraint: bool = False

    # Squared/raw carriers override this to allow non-finite/negative mag values
    # (raw |F|² masked later by preprocessing); see SquaredMagnitudeMeasurement.
    _allow_raw_values: ClassVar[bool] = False

    def __post_init__(self) -> None:
        self.freq, self.mag, self.mag_std, self.detection_limit = _validated_arrays(
            self.freq,
            self.mag,
            self.mag_std,
            self.detection_limit,
            allow_raw=self._allow_raw_values,
        )
        self.kind = _coerce_enum(self.kind, MeasurementKind)
        self.calibration = _coerce_enum(self.calibration, CalibrationStatus)
        self.label = str(self.label)
        self.source = str(self.source)
        if self.charge_c is not None:
            charge = float(self.charge_c)
            if not np.isfinite(charge) or charge <= 0.0:
                raise ValueError("charge_c must be finite and positive")
            self.charge_c = charge
        if self.max_frequency_thz is not None:
            max_f = float(self.max_frequency_thz)
            if not np.isfinite(max_f) or max_f <= 0.0:
                raise ValueError("max_frequency_thz must be finite and positive")
            self.max_frequency_thz = max_f

    @property
    def is_squared(self) -> bool:
        return False

    @property
    def is_absolute(self) -> bool:
        return self.calibration is CalibrationStatus.ABSOLUTE

    def to_squared(self) -> "SquaredMagnitudeMeasurement":
        """Convert to the squared-magnitude (``|F|^2``) representation.

        Uncertainty and detection-limit are propagated by error propagation
        (``d(|F|^2) = 2 |F| d|F|``). Never performed silently.
        """
        mag_sq = self.mag * self.mag
        mag_std_sq = None
        if self.mag_std is not None:
            mag_std_sq = 2.0 * self.mag * self.mag_std
        detection_limit_sq = None
        if self.detection_limit is not None:
            detection_limit_sq = self.detection_limit * self.detection_limit
        return SquaredMagnitudeMeasurement(
            freq=self.freq,
            mag=mag_sq,
            mag_std=mag_std_sq,
            detection_limit=detection_limit_sq,
            kind=self.kind,
            calibration=self.calibration,
            label=self.label,
            source=self.source,
            charge_c=self.charge_c,
            max_frequency_thz=self.max_frequency_thz,
            use_in_absolute_constraint=self.use_in_absolute_constraint,
            use_in_relative_constraint=self.use_in_relative_constraint,
        )


@dataclass
class SquaredMagnitudeMeasurement(Measurement):
    """A measurement whose :attr:`mag` holds ``|F|^2`` (e.g. raw CRISP data).

    The ``mag``/``mag_std``/``detection_limit`` fields carry the squared
    quantities; read them here as ``|F|^2`` (convenience aliases ``ffsq``,
    ``ffsq_std``). Use :meth:`as_magnitude` for the magnitude representation.
    """

    _allow_raw_values: ClassVar[bool] = True

    @property
    def is_squared(self) -> bool:
        return True

    @property
    def ffsq(self) -> np.ndarray:
        return self.mag

    @property
    def ffsq_std(self) -> np.ndarray | None:
        return self.mag_std

    @property
    def ffsq_detection_limit(self) -> np.ndarray | None:
        return self.detection_limit

    def as_magnitude(self) -> Measurement:
        """Convert to the magnitude (``|F|``) representation.

        Uncertainty propagated by error propagation
        (``d(|F|) = d(|F|^2) / (2 |F|)``). Never performed silently.
        """
        mag = np.sqrt(self.mag)
        mag_std = None
        if self.mag_std is not None:
            denominator = 2.0 * mag
            mag_std = np.divide(
                self.mag_std,
                denominator,
                out=np.zeros_like(mag),
                where=denominator > 0.0,
            )
        detection_limit = None
        if self.detection_limit is not None:
            detection_limit = np.sqrt(np.maximum(self.detection_limit, 0.0))
        return Measurement(
            freq=self.freq,
            mag=mag,
            mag_std=mag_std,
            detection_limit=detection_limit,
            kind=self.kind,
            calibration=self.calibration,
            label=self.label,
            source=self.source,
            charge_c=self.charge_c,
            max_frequency_thz=self.max_frequency_thz,
            use_in_absolute_constraint=self.use_in_absolute_constraint,
            use_in_relative_constraint=self.use_in_relative_constraint,
        )
