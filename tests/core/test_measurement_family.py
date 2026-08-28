"""Tests for the unified Measurement type family (core/measurement.py)."""
import numpy as np
import pytest

from phase_weaver.core.measurement import (
    CalibrationStatus,
    Measurement,
    MeasurementKind,
    SquaredMagnitudeMeasurement,
)

rng = np.random.default_rng(0)
FREQ = np.array([1.0, 2.0, 3.0, 4.0])
MAG = np.array([0.5, 0.3, 0.2, 0.1])
STD = np.array([0.05, 0.04, 0.02, 0.01])
DET = np.array([0.01, 0.01, 0.01, 0.01])


def make_measurement(**overrides) -> Measurement:
    params = dict(
        freq=FREQ,
        mag=MAG,
        mag_std=STD,
        detection_limit=DET,
        kind=MeasurementKind.CRISP,
        calibration=CalibrationStatus.ABSOLUTE,
        label="crisp",
        source="file.h5:shot=3",
    )
    params.update(overrides)
    return Measurement(**params)


# --- construction & validation -------------------------------------------


def test_freq_is_sorted_and_arrays_normalised():
    measurement = Measurement(freq=np.array([3.0, 1.0, 2.0]), mag=np.array([9.0, 1.0, 4.0]))
    assert np.array_equal(measurement.freq, [1.0, 2.0, 3.0])
    assert np.array_equal(measurement.mag, [1.0, 4.0, 9.0])


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        Measurement(freq=np.array([1.0, 2.0]), mag=np.array([1.0, 2.0, 3.0]))


def test_optional_arrays_must_match_mag_shape():
    with pytest.raises(ValueError):
        Measurement(freq=FREQ, mag=MAG, mag_std=np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        Measurement(freq=FREQ, mag=MAG, detection_limit=np.array([0.1, 0.2]))


def test_negative_mag_rejected():
    with pytest.raises(ValueError):
        Measurement(freq=FREQ, mag=np.array([1.0, -1.0, 1.0, 1.0]))


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        Measurement(freq=FREQ, mag=np.array([1.0, np.nan, 1.0, 1.0]))


def test_empty_rejected():
    with pytest.raises(ValueError):
        Measurement(freq=np.array([]), mag=np.array([]))


def test_optional_arrays_follow_sort_order():
    measurement = Measurement(
        freq=np.array([3.0, 1.0, 2.0]),
        mag=np.array([9.0, 1.0, 4.0]),
        mag_std=np.array([0.9, 0.1, 0.4]),
    )
    assert np.array_equal(measurement.freq, [1.0, 2.0, 3.0])
    assert np.array_equal(measurement.mag_std, [0.1, 0.4, 0.9])


# --- provenance -----------------------------------------------------------


def test_kind_and_calibration_coerced_from_strings():
    measurement = Measurement(
        freq=FREQ, mag=MAG, kind="ocean_nir", calibration="relative"
    )
    assert measurement.kind is MeasurementKind.OCEAN_NIR
    assert measurement.calibration is CalibrationStatus.RELATIVE


def test_defaults_are_unknown():
    measurement = Measurement(freq=FREQ, mag=MAG)
    assert measurement.kind is MeasurementKind.UNKNOWN
    assert measurement.calibration is CalibrationStatus.UNKNOWN


def test_is_absolute_reflects_calibration():
    assert make_measurement().is_absolute is True
    assert (
        make_measurement(calibration=CalibrationStatus.RELATIVE).is_absolute is False
    )


# --- squared semantics ----------------------------------------------------


def test_magnitude_measurement_is_not_squared():
    assert make_measurement().is_squared is False
    assert make_measurement().to_squared().is_squared is True


def test_squared_measurement_ffsq_aliases():
    squared = make_measurement().to_squared()
    assert np.array_equal(squared.ffsq, squared.mag)
    assert np.array_equal(squared.ffsq_std, squared.mag_std)
    assert np.array_equal(squared.ffsq_detection_limit, squared.detection_limit)


# --- conversions ----------------------------------------------------------


def test_to_squared_squares_values_and_propagates_uncertainty():
    squared = make_measurement().to_squared()
    assert np.allclose(squared.mag, MAG**2)
    assert np.allclose(squared.mag_std, 2.0 * MAG * STD)
    assert np.allclose(squared.detection_limit, DET**2)
    # provenance and flags preserved
    assert squared.kind is MeasurementKind.CRISP
    assert squared.calibration is CalibrationStatus.ABSOLUTE
    assert squared.label == "crisp"
    assert squared.source == "file.h5:shot=3"
    assert squared.use_in_absolute_constraint is True
    assert squared.use_in_relative_constraint is False


def test_as_magnitude_recovers_values_and_uncertainty():
    magnitude = make_measurement().to_squared().as_magnitude()
    assert np.allclose(magnitude.mag, MAG)
    assert np.allclose(magnitude.mag_std, STD)
    assert np.allclose(magnitude.detection_limit, DET)


def test_to_squared_round_trip_preserves_magnitude():
    round_trip = make_measurement().to_squared().as_magnitude()
    assert np.allclose(round_trip.mag, MAG)
    assert np.allclose(round_trip.mag_std, STD)
    assert np.allclose(round_trip.detection_limit, DET)


def test_conversion_without_uncertainty_leaves_none():
    measurement = Measurement(freq=FREQ, mag=MAG)
    assert measurement.mag_std is None
    squared = measurement.to_squared()
    assert squared.mag_std is None
    assert squared.detection_limit is None
    back = squared.as_magnitude()
    assert back.mag_std is None
    assert np.allclose(back.mag, MAG)


def test_constructing_squared_measurement_directly():
    squared = SquaredMagnitudeMeasurement(
        freq=FREQ,
        mag=MAG**2,
        mag_std=STD,
        detection_limit=DET,
        kind=MeasurementKind.CRISP,
        calibration=CalibrationStatus.ABSOLUTE,
    )
    assert squared.is_squared is True
    assert squared.is_absolute is True
    assert np.allclose(squared.as_magnitude().mag, MAG)


def test_squared_measurement_allows_raw_values():
    # Raw |F|^2 may carry non-finite/negative entries (below the detection
    # limit); the preprocessing layer is responsible for masking them.
    squared = SquaredMagnitudeMeasurement(
        freq=np.array([1.0, 2.0, 3.0, 4.0]),
        mag=np.array([0.5, np.nan, -0.2, 0.9]),
    )
    assert squared.is_squared is True
    assert np.isnan(squared.mag[1])
    assert squared.mag[2] == -0.2


def test_magnitude_measurement_still_rejects_raw_values():
    # The magnitude carrier remains strict (finite, non-negative).
    with pytest.raises(ValueError):
        Measurement(freq=np.array([1.0, 2.0]), mag=np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        Measurement(freq=np.array([1.0, 2.0]), mag=np.array([1.0, -1.0]))
