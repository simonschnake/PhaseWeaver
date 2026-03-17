import numpy as np
import pytest
from numpy.testing import assert_allclose

from phase_weaver.core.measurement import MeasuredFormFactor


def test_constructs_from_lists_and_converts_to_float_arrays():
    measured = MeasuredFormFactor(
        freq=[3, 1, 2],
        mag=[30, 10, 20],
    )

    assert isinstance(measured.freq, np.ndarray)
    assert isinstance(measured.mag, np.ndarray)
    assert measured.freq.dtype == float
    assert measured.mag.dtype == float

    # also checks sorting happens
    assert_allclose(measured.freq, [1.0, 2.0, 3.0])
    assert_allclose(measured.mag, [10.0, 20.0, 30.0])


def test_preserves_pairing_when_sorting_by_frequency():
    measured = MeasuredFormFactor(
        freq=np.array([5.0, 1.0, 3.0]),
        mag=np.array([50.0, 10.0, 30.0]),
    )

    assert_allclose(measured.freq, [1.0, 3.0, 5.0])
    assert_allclose(measured.mag, [10.0, 30.0, 50.0])


def test_raises_for_non_1d_freq():
    with pytest.raises(ValueError, match="freq and mag must be 1D"):
        MeasuredFormFactor(
            freq=np.array([[1.0, 2.0], [3.0, 4.0]]),
            mag=np.array([10.0, 20.0]),
        )


def test_raises_for_non_1d_mag():
    with pytest.raises(ValueError, match="freq and mag must be 1D"):
        MeasuredFormFactor(
            freq=np.array([1.0, 2.0]),
            mag=np.array([[10.0, 20.0], [30.0, 40.0]]),
        )


def test_raises_for_length_mismatch():
    with pytest.raises(ValueError, match="freq and mag must have the same length"):
        MeasuredFormFactor(
            freq=np.array([1.0, 2.0, 3.0]),
            mag=np.array([10.0, 20.0]),
        )


def test_raises_for_empty_input():
    with pytest.raises(ValueError, match="freq and mag must not be empty"):
        MeasuredFormFactor(
            freq=np.array([]),
            mag=np.array([]),
        )


def test_raises_for_non_finite_frequency():
    with pytest.raises(ValueError, match="freq and mag must contain only finite values"):
        MeasuredFormFactor(
            freq=np.array([1.0, np.nan, 3.0]),
            mag=np.array([10.0, 20.0, 30.0]),
        )


def test_raises_for_non_finite_magnitude():
    with pytest.raises(ValueError, match="freq and mag must contain only finite values"):
        MeasuredFormFactor(
            freq=np.array([1.0, 2.0, 3.0]),
            mag=np.array([10.0, np.inf, 30.0]),
        )