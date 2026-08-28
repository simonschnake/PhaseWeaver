import numpy as np
from numpy.testing import assert_allclose

from phase_weaver.core.crisp_simulation import (
    CrispSimulationConfig,
    simulate_crisp_measurement,
)
from phase_weaver.core.measurement import CalibrationStatus, MeasurementKind


def _gaussian_profile() -> tuple[np.ndarray, np.ndarray, float]:
    time_s = np.arange(-250e-15, 251e-15, 1e-15)
    current_a = np.exp(-0.5 * (time_s / 18e-15) ** 2)
    charge_c = 250e-12
    current_a *= charge_c / np.trapezoid(current_a, x=time_s)
    return time_s, current_a, charge_c


def test_crisp_simulator_uses_calibrated_channel_sets_and_finite_outputs():
    time_s, current_a, charge_c = _gaussian_profile()

    low = simulate_crisp_measurement(
        time_s,
        current_a,
        charge_c,
        CrispSimulationConfig(channel_set="low"),
    )
    high = simulate_crisp_measurement(
        time_s,
        current_a,
        charge_c,
        CrispSimulationConfig(channel_set="high"),
    )
    both = simulate_crisp_measurement(time_s, current_a, charge_c)

    assert low.freq.shape == high.freq.shape == (120,)
    assert both.freq.shape == (240,)
    assert_allclose(both.freq[:120], low.freq)
    assert_allclose(both.freq[120:], high.freq)

    magnitude = both.as_magnitude()
    for values in (
        both.mag,
        both.mag_std,
        both.detection_limit,
        magnitude.mag,
        magnitude.mag_std,
        magnitude.detection_limit,
    ):
        assert np.all(np.isfinite(values))

    # Provenance: CRISP is calibrated, absolute |F|^2.
    assert both.kind is MeasurementKind.CRISP
    assert both.calibration is CalibrationStatus.ABSOLUTE
    assert both.is_squared is True
    assert both.is_absolute is True


def test_crisp_simulator_is_repeatable_and_converts_uncertainties_to_ffsq():
    time_s, current_a, charge_c = _gaussian_profile()
    config = CrispSimulationConfig(n_shots=3, seed=42)

    first = simulate_crisp_measurement(time_s, current_a, charge_c, config)
    second = simulate_crisp_measurement(time_s, current_a, charge_c, config)
    magnitude = first.as_magnitude()

    # Repeatability.
    assert_allclose(magnitude.mag, second.as_magnitude().mag)
    # |F|^2 values derived from the magnitude view, with error propagation.
    assert_allclose(first.mag, magnitude.mag**2)
    assert_allclose(first.mag_std, 2.0 * magnitude.mag * magnitude.mag_std)
    assert_allclose(
        first.detection_limit,
        magnitude.detection_limit**2,
    )


def test_crisp_simulator_shot_averaging_reduces_electronic_noise_floor():
    time_s, current_a, charge_c = _gaussian_profile()
    single = simulate_crisp_measurement(
        time_s,
        current_a,
        charge_c,
        CrispSimulationConfig(n_shots=1, seed=5),
    )
    averaged = simulate_crisp_measurement(
        time_s,
        current_a,
        charge_c,
        CrispSimulationConfig(n_shots=16, seed=5),
    )

    assert_allclose(
        averaged.as_magnitude().detection_limit,
        single.as_magnitude().detection_limit / 2.0,
    )
