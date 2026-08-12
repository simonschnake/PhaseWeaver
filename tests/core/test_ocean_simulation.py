import numpy as np
import pytest
from numpy.testing import assert_allclose

from phase_weaver.core.ocean_simulation import (
    OCEAN_MAX_WAVELENGTH_NM,
    OCEAN_MIN_WAVELENGTH_NM,
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

    assert result.freq_hz.shape == (OCEAN_NUM_PIXELS,)
    assert result.wavelength_nm.shape == (OCEAN_NUM_PIXELS,)
    assert result.wavelength_nm[0] == pytest.approx(OCEAN_MAX_WAVELENGTH_NM)
    assert result.wavelength_nm[-1] == pytest.approx(OCEAN_MIN_WAVELENGTH_NM)
    assert np.all(np.diff(result.freq_hz) > 0.0)
    for values in (
        result.intensity_counts,
        result.intensity_std_counts,
        result.ffabs_relative,
        result.ffabs_std,
        result.ffabs_detection_limit,
    ):
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)
    assert np.all(result.ffabs_relative <= 1.0)


def test_ocean_simulator_is_deterministic_for_fixed_seed():
    freq_hz, mag = _source_formfactor()
    config = OceanSimulationConfig(n_shots=7, seed=19)

    first = simulate_ocean_measurement(freq_hz, mag, config)
    second = simulate_ocean_measurement(freq_hz, mag, config)

    assert_allclose(first.intensity_counts, second.intensity_counts)
    assert_allclose(first.ffabs_relative, second.ffabs_relative)
    assert_allclose(first.ffabs_std, second.ffabs_std)


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

    assert np.mean(many.intensity_std_counts) == pytest.approx(
        np.mean(one.intensity_std_counts) / 5.0
    )
    assert np.mean(many.ffabs_std) < np.mean(one.ffabs_std)


def test_ocean_simulator_validates_configuration_and_input():
    with pytest.raises(ValueError, match="n_shots"):
        OceanSimulationConfig(n_shots=0)
    with pytest.raises(ValueError, match="equal shape"):
        simulate_ocean_measurement(np.ones(3), np.ones(2))
