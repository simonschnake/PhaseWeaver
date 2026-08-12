import numpy as np
from numpy.testing import assert_allclose

from phase_weaver.core.crisp_simulation import (
    CrispSimulationConfig,
    simulate_crisp_measurement,
)


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

    assert low.freq_hz.shape == high.freq_hz.shape == (120,)
    assert both.freq_hz.shape == (240,)
    assert_allclose(both.freq_hz[:120], low.freq_hz)
    assert_allclose(both.freq_hz[120:], high.freq_hz)
    for values in (
        both.ffabs,
        both.ffabs_std,
        both.detection_limit,
        both.ffsq,
        both.ffsq_std,
        both.ffsq_detection_limit,
    ):
        assert np.all(np.isfinite(values))


def test_crisp_simulator_is_repeatable_and_converts_uncertainties_to_ffsq():
    time_s, current_a, charge_c = _gaussian_profile()
    config = CrispSimulationConfig(n_shots=3, seed=42)

    first = simulate_crisp_measurement(time_s, current_a, charge_c, config)
    second = simulate_crisp_measurement(time_s, current_a, charge_c, config)

    assert_allclose(first.ffabs, second.ffabs)
    assert_allclose(first.ffsq, first.ffabs**2)
    assert_allclose(first.ffsq_std, 2.0 * first.ffabs * first.ffabs_std)
    assert_allclose(
        first.ffsq_detection_limit,
        first.detection_limit**2,
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

    assert_allclose(averaged.detection_limit, single.detection_limit / 2.0)
