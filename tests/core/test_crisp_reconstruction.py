from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from phase_weaver.app.logic import load_measurements_h5
from phase_weaver.core.crisp_reconstruction import (
    CrispReconstruction,
    CrispReconstructionConfig,
    _replace_modulus_outside_band,
    build_crisp_full_spectrum,
    center_maximum,
    extrapolate_crisp_ffsq,
    extrapolate_crisp_ffsq_with_uncertainty,
    fit_low_frequency_sigma,
    fit_ocelot_low_frequency_gaussian,
    high_frequency_sigma,
    interpolate_crisp_ffabs,
    interpolate_crisp_ffabs_error,
    isolate_positive_maximum,
    kramers_kronig_phase,
    preprocess_crisp_input,
    smooth_intermediate_ffsq,
)
from phase_weaver.core.measurement import SquaredMagnitudeMeasurement


def _crisp_input(freq_hz, ffsq, *, ffsq_std=None, detection_limit=None, charge_c=None):
    """Build a SquaredMagnitudeMeasurement, zero-filling missing std/det."""
    ffsq = np.asarray(ffsq, dtype=float)
    return SquaredMagnitudeMeasurement(
        freq=freq_hz,
        mag=ffsq,
        mag_std=ffsq_std if ffsq_std is not None else np.zeros_like(ffsq),
        detection_limit=(
            detection_limit
            if detection_limit is not None
            else np.zeros_like(ffsq)
        ),
        charge_c=charge_c if charge_c is not None else 250e-12,
    )


def test_preprocess_sorts_masks_and_applies_bad_run_cutoff():
    config = CrispReconstructionConfig(
        min_input_points=20,
        detection_limit_fudge=1.0,
    )
    freq_hz = np.array([5, 1, 4, 2, 3, *range(6, 31)], dtype=float) * 1e12
    ffsq = np.full_like(freq_hz, 0.5, dtype=float)
    detection = np.zeros_like(ffsq)
    ffsq[4] = np.nan
    ffsq[-16:] = -1.0
    detection[-16:] = 0.0

    pre = preprocess_crisp_input(
        _crisp_input(
            freq_hz=freq_hz,
            ffsq=ffsq,
            detection_limit=detection,
            charge_c=250e-12,
        ),
        config,
    )

    assert pre.num_input_points == len(freq_hz)
    assert np.all(np.diff(pre.freq_thz) > 0.0)
    assert pre.max_input_frequency_thz < 15.0
    assert np.all((pre.ffsq >= 0.0) & (pre.ffsq <= 1.0))
    assert_allclose(pre.ffsq_std, 0.2 * pre.ffsq)


def test_preprocess_can_remove_explicit_detector_channels():
    freq_hz = np.arange(1.0, 31.0) * 1e12
    ffsq = np.linspace(1.0, 0.2, len(freq_hz))
    config = CrispReconstructionConfig(
        min_input_points=20,
        channels_to_remove=(8, 12),
    )

    pre = preprocess_crisp_input(
        _crisp_input(freq_hz=freq_hz, ffsq=ffsq),
        config,
    )

    assert pre.num_input_points == 30
    assert 9.0 not in pre.freq_thz
    assert 13.0 not in pre.freq_thz


def test_gaussian_fits_and_extrapolation_are_finite():
    freq = np.linspace(1.0, 50.0, 120)
    sigma = 12.0
    ffsq = np.exp(-0.5 * (freq / sigma) ** 2)
    pre = preprocess_crisp_input(
        _crisp_input(freq_hz=freq * 1e12, ffsq=ffsq),
        CrispReconstructionConfig(min_input_points=100),
    )

    low_sigma = fit_low_frequency_sigma(pre.freq_thz, pre.ffsq)
    high_sigma = high_frequency_sigma(pre.freq_thz[-1], pre.ffsq[-1])
    inter_freq, inter_ffsq, idx_tail = extrapolate_crisp_ffsq(pre)

    assert low_sigma == pytest.approx(sigma, rel=0.1)
    assert np.isfinite(high_sigma)
    assert inter_freq[0] == 0.0
    assert idx_tail > len(pre.freq_thz)
    assert np.all(np.isfinite(inter_ffsq))


def test_ocelot_low_frequency_fit_uses_weighted_magnitude_handoff():
    freq = np.linspace(0.5, 60.0, 120)
    sigma = 8.0
    ffabs = np.exp(-0.5 * (freq / sigma) ** 2)
    ffsq = ffabs**2
    ffsq[1] = 0.01
    ffsq_std = 0.05 * ffabs**2
    ffsq_std[1] = 10.0
    config = CrispReconstructionConfig(min_input_points=100)
    pre = preprocess_crisp_input(
        _crisp_input(
            freq_hz=freq * 1e12,
            ffsq=ffsq,
            ffsq_std=ffsq_std,
        ),
        config,
    )

    low_fit = fit_ocelot_low_frequency_gaussian(
        pre.freq_thz,
        pre.ffsq,
        pre.ffsq_std,
    )
    inter_freq, inter_ffsq, inter_ffsq_std, _ = (
        extrapolate_crisp_ffsq_with_uncertainty(pre, config)
    )

    assert low_fit.sigma_thz == pytest.approx(sigma, rel=0.05)
    assert low_fit.replacement_start > 0
    assert inter_freq[0] == 0.0
    low_extension = inter_freq < pre.freq_thz[low_fit.replacement_start]
    assert np.all(inter_ffsq_std[low_extension] >= 0.0)
    assert np.any(inter_ffsq_std[low_extension] > 0.0)
    assert_allclose(inter_ffsq_std[-10:], 0.0)


def test_smooth_intermediate_ffsq_keeps_four_edge_samples():
    ffsq = np.linspace(0.0, 1.0, 21)
    ffsq[10] = 0.0

    smoothed = smooth_intermediate_ffsq(ffsq)

    assert_allclose(smoothed[:4], ffsq[:4])
    assert_allclose(smoothed[-4:], ffsq[-4:])
    assert np.all((smoothed >= 0.0) & (smoothed <= 1.0))


def test_interpolated_error_uses_measurements_and_exact_extrapolation():
    freq = np.linspace(1.0, 50.0, 120)
    ffsq = np.exp(-0.5 * (freq / 12.0) ** 2)
    supplied_std = 0.4 * ffsq
    config = CrispReconstructionConfig(min_input_points=100)
    pre = preprocess_crisp_input(
        _crisp_input(
            freq_hz=freq * 1e12,
            ffsq=ffsq,
            ffsq_std=supplied_std,
        ),
        config,
    )
    inter_freq, inter_ffsq, _ = extrapolate_crisp_ffsq(pre, config)
    target_freq, ffabs, _ = interpolate_crisp_ffabs(
        inter_freq,
        inter_ffsq,
        pre.max_input_frequency_thz,
        config,
    )

    error = interpolate_crisp_ffabs_error(target_freq, ffabs, pre)
    measured = (target_freq >= pre.freq_thz[0]) & (
        target_freq <= pre.freq_thz[-1]
    )

    assert np.all(error[measured] > 0.0)
    assert_allclose(error[~measured], 0.0)


def test_interpolated_error_includes_low_frequency_fit_uncertainty():
    freq = np.linspace(0.5, 60.0, 120)
    ffabs = np.exp(-0.5 * (freq / 8.0) ** 2)
    config = CrispReconstructionConfig(
        min_input_points=100,
        num_output_points=1024,
        max_frequency_thz=60.0,
    )
    pre = preprocess_crisp_input(
        _crisp_input(
            freq_hz=freq * 1e12,
            ffsq=ffabs**2,
            ffsq_std=0.1 * ffabs**2,
        ),
        config,
    )
    inter_freq, inter_ffsq, inter_ffsq_std, _ = (
        extrapolate_crisp_ffsq_with_uncertainty(pre, config)
    )
    target_freq, interp_ffabs, _ = interpolate_crisp_ffabs(
        inter_freq,
        inter_ffsq,
        pre.max_input_frequency_thz,
        config,
    )

    error = interpolate_crisp_ffabs_error(
        target_freq,
        interp_ffabs,
        pre,
        intermediate_frequencies_thz=inter_freq,
        intermediate_ffsq_std=inter_ffsq_std,
    )

    low_extension = target_freq < pre.freq_thz[0]
    assert np.any(error[low_extension] > 0.0)


def test_kramers_kronig_phase_fills_nonpositive_tail():
    freq = np.arange(12, dtype=float)
    mag = np.linspace(1.0, 0.2, 12)
    mag[9:] = 0.0

    phase = kramers_kronig_phase(freq, mag, dnu_thz=1.0)

    assert phase.shape == mag.shape
    assert np.isfinite(phase).all()
    assert_allclose(phase[9:], phase[8])


def test_full_spectrum_is_hermitian_and_has_real_inverse():
    ffabs = np.linspace(1.0, 0.1, 9)
    phase = np.linspace(0.0, 0.7, 9)

    spectrum = build_crisp_full_spectrum(ffabs, phase)

    assert spectrum.shape == (16,)
    assert spectrum[0].imag == 0.0
    assert spectrum[8].imag == 0.0
    for i in range(1, len(spectrum)):
        assert spectrum[i] == pytest.approx(np.conj(spectrum[-i]))
    assert_allclose(np.fft.ifft(spectrum).imag, 0.0, atol=1e-15)


def test_positive_peak_isolation_and_centering():
    values = np.array([-1.0, 0.0, 2.0, 3.0, 1.0, 0.0, 2.0])

    isolated = isolate_positive_maximum(values)
    centered = center_maximum(isolated)

    assert_allclose(isolated, [0.0, 0.0, 2.0, 3.0, 1.0, 0.0, 0.0])
    assert int(np.argmax(centered)) == (len(centered) - 1) // 2


def test_modulus_replacement_reports_convergence_when_in_band():
    spectrum = np.array([1.0 + 0j, 0.9 + 0j, 0.8 + 0j, 0.9 + 0j])
    ref = np.array([1.0, 0.9])
    err = np.array([0.2, 0.2])

    replaced = _replace_modulus_outside_band(spectrum, ref, err, 2)

    assert replaced == 0


def test_crisp_reconstruction_runs_on_synthetic_input():
    freq = np.linspace(0.5, 60.0, 160)
    ffsq = np.exp(-0.5 * (freq / 18.0) ** 2)
    alg = CrispReconstruction(
        _crisp_input(
            freq_hz=freq * 1e12,
            ffsq=ffsq,
            charge_c=250e-12,
        )
    )

    result = alg.run()

    assert result.profile.grid.N == 1024
    assert result.form_factor.mag.shape == (513,)
    assert result.diagnostics.interpolated_ffabs.shape == (513,)
    assert result.diagnostics.interpolated_ffabs_error.shape == (513,)
    assert result.diagnostics.num_iterations <= 20
    assert np.all(np.isfinite(result.profile.values))
    assert result.diagnostics.peak_current_a > 0.0


def test_gaussian_reconstruction_does_not_grow_long_artificial_tails():
    freq = np.geomspace(0.7, 58.0, 240)
    sigma_frequency_thz = 15.0
    ffabs = np.exp(-0.5 * (freq / sigma_frequency_thz) ** 2)
    config = CrispReconstructionConfig(
        num_output_points=2048,
        max_frequency_thz=500.0,
    )
    result = CrispReconstruction(
        _crisp_input(
            freq_hz=freq * 1e12,
            ffsq=ffabs**2,
            ffsq_std=0.4 * ffabs**2,
            detection_limit=np.full_like(ffabs, 1e-12),
            charge_c=250e-12,
        ),
        config,
    ).run()

    expected_rms_fs = 1e15 / (2.0 * np.pi * sigma_frequency_thz * 1e12)
    current_a = result.profile.values * result.profile.charge
    outside = np.abs(result.profile.grid.t * 1e15) > 50.0

    assert result.diagnostics.rms_width_fs == pytest.approx(
        expected_rms_fs, rel=0.02
    )
    assert np.sum(current_a[outside]) / np.sum(current_a) < 1e-3


def test_crisp_reconstruction_runs_on_reference_h5_when_available():
    path = Path("2026.06.17_10.48.04.h5")
    if not path.exists():
        pytest.skip("reference HDF5 recording is not available")

    loaded = load_measurements_h5(path)
    assert loaded[0].crisp_input is not None

    result = CrispReconstruction(loaded[0].crisp_input).run()
    current_a = result.profile.values * result.profile.charge

    with h5py.File(path, "r") as data:
        shot_index = int(loaded[0].crisp_input.source.rsplit("=", 1)[1])
        reference = np.asarray(
            data[
                "XFEL.SDIAG__THZ_SPECTROMETER.RECONSTRUCTION__CRD.1934.TL.SA1__CURRENT_PROFILE"
            ][shot_index]
        )

    assert current_a.shape == reference.shape == (1024,)
    assert np.all(np.isfinite(current_a))
    assert np.max(current_a) == pytest.approx(np.max(reference), rel=0.25)
