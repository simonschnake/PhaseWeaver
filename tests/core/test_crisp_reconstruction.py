from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from phase_weaver.app.logic import load_measurements_h5
from phase_weaver.core.crisp_reconstruction import (
    CrispReconstruction,
    CrispReconstructionConfig,
    CrispReconstructionInput,
    _replace_modulus_outside_band,
    center_maximum,
    extrapolate_crisp_ffsq,
    fit_low_frequency_sigma,
    high_frequency_sigma,
    isolate_positive_maximum,
    kramers_kronig_phase,
    preprocess_crisp_input,
    smooth_intermediate_ffsq,
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
        CrispReconstructionInput(
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


def test_gaussian_fits_and_extrapolation_are_finite():
    freq = np.linspace(1.0, 50.0, 120)
    sigma = 12.0
    ffsq = np.exp(-0.5 * (freq / sigma) ** 2)
    pre = preprocess_crisp_input(
        CrispReconstructionInput(freq_hz=freq * 1e12, ffsq=ffsq),
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


def test_smooth_intermediate_ffsq_keeps_four_edge_samples():
    ffsq = np.linspace(0.0, 1.0, 21)
    ffsq[10] = 0.0

    smoothed = smooth_intermediate_ffsq(ffsq)

    assert_allclose(smoothed[:4], ffsq[:4])
    assert_allclose(smoothed[-4:], ffsq[-4:])
    assert np.all((smoothed >= 0.0) & (smoothed <= 1.0))


def test_kramers_kronig_phase_fills_nonpositive_tail():
    freq = np.arange(12, dtype=float)
    mag = np.linspace(1.0, 0.2, 12)
    mag[9:] = 0.0

    phase = kramers_kronig_phase(freq, mag, dnu_thz=1.0)

    assert phase.shape == mag.shape
    assert np.isfinite(phase).all()
    assert_allclose(phase[9:], phase[8])


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
        CrispReconstructionInput(
            freq_hz=freq * 1e12,
            ffsq=ffsq,
            charge_c=250e-12,
        )
    )

    result = alg.run()

    assert result.profile.grid.N == 1024
    assert result.form_factor.mag.shape == (513,)
    assert result.diagnostics.num_iterations <= 20
    assert np.all(np.isfinite(result.profile.values))
    assert result.diagnostics.peak_current_a > 0.0


def test_crisp_reconstruction_runs_on_reference_h5_when_available():
    path = Path("2026.06.17_10.48.04.h5")
    if not path.exists():
        pytest.skip("reference HDF5 recording is not available")

    loaded = load_measurements_h5(path)
    assert loaded[0].crisp_input is not None

    result = CrispReconstruction(loaded[0].crisp_input).run()
    current_a = result.profile.values * result.profile.charge

    with h5py.File(path, "r") as data:
        reference = np.asarray(
            data[
                "XFEL.SDIAG__THZ_SPECTROMETER.RECONSTRUCTION__CRD.1934.TL.SA1__CURRENT_PROFILE"
            ][loaded[0].crisp_input.shot_index]
        )

    assert current_a.shape == reference.shape == (1024,)
    assert np.all(np.isfinite(current_a))
    assert np.max(current_a) == pytest.approx(np.max(reference), rel=0.25)
