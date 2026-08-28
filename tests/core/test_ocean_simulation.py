import numpy as np
import pytest
from numpy.testing import assert_allclose

from phase_weaver.core.measurement import CalibrationStatus, MeasurementKind
from phase_weaver.core.ocean_simulation import (
    OCEAN_NUM_PIXELS,
    OceanSimulationConfig,
    simulate_ocean_measurement,
)


def _source_formfactor() -> tuple[np.ndarray, np.ndarray]:
    freq_hz = np.linspace(0.0, 400e12, 4001)
    mag = np.exp(-0.5 * (freq_hz / 180e12) ** 2)
    return freq_hz, mag


def test_ocean_simulator_uses_nirquest_pixel_grid_and_finite_outputs():
    freq_hz, mag = _source_formfactor()

    result = simulate_ocean_measurement(freq_hz, mag)

    assert result.freq.shape == (OCEAN_NUM_PIXELS,)
    assert np.all(np.diff(result.freq) > 0.0)
    for values in (result.mag, result.mag_std, result.detection_limit):
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)
    assert np.all(result.mag <= 1.0)

    # Provenance: Ocean/NIR is a relative shape only, never absolute.
    assert result.kind is MeasurementKind.OCEAN_NIR
    assert result.calibration is CalibrationStatus.RELATIVE
    assert result.is_squared is False
    assert result.is_absolute is False


def test_ocean_simulator_is_deterministic_for_fixed_seed():
    freq_hz, mag = _source_formfactor()
    config = OceanSimulationConfig(n_shots=7, seed=19)

    first = simulate_ocean_measurement(freq_hz, mag, config)
    second = simulate_ocean_measurement(freq_hz, mag, config)

    assert_allclose(first.mag, second.mag)
    assert_allclose(first.mag_std, second.mag_std)
    assert_allclose(first.detection_limit, second.detection_limit)


def test_ocean_shot_averaging_reduces_reported_uncertainty():
    freq_hz, mag = _source_formfactor()

    one = simulate_ocean_measurement(
        freq_hz,
        mag,
        OceanSimulationConfig(n_shots=1, seed=4),
    )
    many = simulate_ocean_measurement(
        freq_hz,
        mag,
        OceanSimulationConfig(n_shots=25, seed=4),
    )

    assert np.mean(many.mag_std) < np.mean(one.mag_std)


def test_ocean_simulator_validates_configuration_and_input():
    with pytest.raises(ValueError, match="n_shots"):
        OceanSimulationConfig(n_shots=0)
    with pytest.raises(ValueError, match="equal shape"):
        simulate_ocean_measurement(np.ones(3), np.ones(2))
