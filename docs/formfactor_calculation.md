# CRISP Form Factor Calculation

This note summarizes how the upstream `crispformfactor` service calculates the
published form factor arrays. The source inspected here is
`/Users/schnakes/Downloads/crispformfactor-main.zip`.

The most important point: this code does not calculate a form factor by taking
an FFT of a time-domain current profile. It converts CRISP detector channel
signals into form factor squared values using pile-up correction, channel
calibration, bunch charge, and detector response.

## Key Outputs

The service works with 120 CRISP signal channels in each grating mode:

- `MAX_SIG = 120`
- `FORMFACTOR_SIZE = 240`
- low-frequency bins occupy indices `0..119`
- high-frequency bins occupy indices `120..239`

Main output properties include:

- `FORMFACTOR.XY`: form factor squared for the first bunch, sorted by frequency.
- `FORMFACTOR`: the same first-bunch values as a plain float array.
- `FORMFACTOR_MEAN.ARRAY`: sliding mean form factor squared for each bunch.
- `FORMFACTOR_MEAN_NTH.XY`: mean form factor squared for a selected bunch.
- `FORMFACTOR_STD`: sliding standard deviation of the mean array values.
- `FORMFACTOR_MEAN_DETECTLIMIT.XY`: detection-limit estimate from baseline
  noise.

Although several variables are named `formfactor`, the stored values are
form factor squared. The header names the float arrays as `FormFactorSquare`,
and the calculation divides intensity-like detector signal by charge squared.

## Startup Calibration

During `post_init()`, the service reads:

```text
respMatrix/Channel_sorting_and_response.csv
```

Each row maps a raw shaper-server channel to calibrated metadata:

- channel position in the sorted output array
- low-frequency response
- high-frequency response
- low-frequency bin frequency
- high-frequency bin frequency
- low/high channel validity flags

The values are installed on each `D_CRISPshortarray`:

```text
setChannelPos(index)
setResponse(resLow, resHigh)
setFrequency(freqLow, freqHigh)
setOk(okLow, okHigh)
```

This is why the final arrays are sorted by physical frequency rather than by raw
input channel number.

The service also loads response-vector files through
`D_CRISPshortarray::LoadFiles()` and builds an inverse response vector with
`create_b_vector()`. That vector is used for pile-up correction.

## Pile-Up Correction

Every `SIGNAL.N.ZMQ` input is represented by a `D_CRISPshortarray`. On each
measurement interrupt, the service runs `calcCorrected()` for all 120 channels
in a thread pool.

For each raw short-array sample `data[k]`, the corrected signal is calculated as
a convolution with the inverse response vector `b_`:

```text
corrected[k] = sum over j from s to k of b_[k - j] * data[j]
```

where `s` limits the loop to the useful response length for the active grating
mode. The corrected value is converted with the `INTENSITY.TD` evaluator and
stored in `SIGNAL.N.CORRECTED.TD`.

The active grating mode matters:

- `LOWFREQ` uses the low-frequency response vector and calibration.
- `HIGHFREQ` uses the high-frequency response vector and calibration.
- when the grating changes, `update()` rebuilds the active `b_` vector for every
  channel.

## Per-Pulse Sequence

The main calculation is triggered from `interrupt_usr1()`. After detector state
checks and pile-up correction, the service runs:

1. `fillIntensity(macroPulse)`
2. `fillFormfactorArray(macroPulse)`
3. `calcFormfactorMeanArray(macroPulse)`
4. `fillFormfactorFromArray(macroPulse)`
5. `calcFormfactor(macroPulse)`
6. `fillFormfactorImage(macroPulse)`
7. `calcBaselineSigma()`

The service refuses to publish fresh valid data when the spectrometer is moving,
the detector is off, the hole is out, the last mirror is in the beam path, or
the bunch pattern is missing/stale.

## First-Bunch Intensity Array

`fillIntensity()` creates `INTENSITY.TD` for the first bunch only.

For each of the 120 channels:

1. read the corrected signal at `pre_samples`
2. look up the calibrated channel position
3. write into the low or high half of the 240-element intensity array

In simplified form:

```text
idx = channel_position
if active mode is HIGHFREQ:
    idx += 120

INTENSITY.TD[idx] = SIGNAL[channel].CORRECTED.TD[pre_samples]
```

## First-Bunch Form Factor Squared

`calcFormfactor()` computes `FORMFACTOR.XY` and `FORMFACTOR` for the first bunch.
It reads the first bunch charge from `CHARGE.TD` at `firstIdxWithBeam_`.

For each low/high output bin:

```text
signal = INTENSITY.TD[l]
response = channel_response_for_low_or_high_mode
frequency = channel_frequency_for_low_or_high_mode

ffsq = signal / (charge * charge) / response
```

The result is written as:

```text
FORMFACTOR.XY[l] = (frequency, ffsq)
FORMFACTOR[l] = ffsq
```

So the first-bunch form factor product is a charge-normalized and
response-normalized detector intensity:

```text
ffsq(f) = corrected_detector_signal(f) / charge^2 / detector_response(f)
```

No square root is taken in the CRISP service for this output.

## Bunch-Train Form Factor Array

`fillFormfactorArray()` calculates form factor squared for every bunch that
passes timing-pattern and charge checks.

The service scans the bunch pattern and skips:

- destination `0`
- bunch indices that are not sampled by the detector rate
- bunches below 10 percent of the timing-pattern lower charge limit

For each accepted bunch and each of the 120 active-mode channels:

```text
signal = SIGNAL[channel].CORRECTED.TD[presample + bunchIdx / rate_divider]
response = channel_response_for_active_mode
charge = CHARGE.TD[bunchIdx]

ffsq = signal / (charge * charge) / response
pixel = channel_position
if active mode is HIGHFREQ:
    pixel += 120

FORMFACTOR.ARRAY[pixel, bunchIdx] = ffsq
```

This array uses the same bunch indexing as `CHARGE.TD`.

## Sliding Mean And Standard Deviation

`calcFormfactorMeanArray()` builds smoothed values from `FORMFACTOR.ARRAY`.
For each valid bunch, each pixel's latest `ffsq` value is pushed into a
16-sample sliding buffer:

```text
ffMeanStd_[bunchIdx].push_back(ffsq, pixel)
```

Then `formfactorMeanStd::calcMean()` calculates the mean and standard deviation
for the active half of the 240-bin array:

```text
FORMFACTOR_MEAN.ARRAY[pixel, bunchIdx] = sliding_mean(ffsq[pixel, bunchIdx])
FORMFACTOR_STD[pixel] = sliding_sigma(ffsq[pixel, bunchIdx])
```

The same routine also fills per-subtrain mean outputs. For the first bunch found
in each configured subtrain, `fill_subtrain_formfactor()` copies the mean array
into that subtrain's `FORMFACTOR_MEAN.<subtrain>.XY` and
`FORMFACTOR_MEAN.<subtrain>` float array.

## Selected-Bunch Mean Output

`fillFormfactorFromArray()` publishes the selected bunch from
`FORMFACTOR_MEAN.ARRAY` as `FORMFACTOR_MEAN_NTH.XY` and
`FORMFACTOR_MEAN_NTH`.

The selected bunch is controlled by `NTH_BUNCH`. If the charge array has a
sample increment greater than one, the bunch index is adjusted with that
increment before reading from the mean array.

## Detection Limit

`calcBaselineSigma()` estimates baseline noise for each corrected signal channel.
It uses the first half of the pre-sample region:

```text
baseline_mean = mean(corrected[0:pre_samples / 2])
baseline_sigma = sqrt(mean((baseline_mean - corrected[i])^2))
```

`fillFormfactorDetectLimit()` converts that baseline noise into a form factor
squared detection-limit estimate using the same charge-squared and response
normalization:

```text
limit_ffsq = baseline_sigma / (chargeFirstBunch * chargeFirstBunch) / response
```

Those values are stored in `FORMFACTOR_MEAN_DETECTLIMIT.XY` and
`FORMFACTOR_MEAN_DETECTLIMIT`.

## Frequency And Grating Handling

The active grating selects which calibration half is being filled:

- low-frequency mode fills pixels `0..119`
- high-frequency mode fills pixels `120..239`

The frequency written to each `XY` output comes from the channel metadata loaded
from `Channel_sorting_and_response.csv`. The code does not derive frequencies
from a sampling interval or FFT bin spacing.

`check_grating()` compares the tail of the low-frequency mean curve with the
start of the high-frequency mean curve. If their means disagree too much, it
clears `GRATING.ALLOWED_TRANSITION`.

## Formula Summary

For a channel/bin `i` and bunch `b`, the core CRISP calculation is:

```text
corrected_signal_i,b = pileup_correct(raw_signal_i,b, inverse_response_i)

ffsq_i,b = corrected_signal_i,b / charge_b^2 / response_i
```

For first-bunch outputs, `b` is `firstIdxWithBeam_`.

For mean outputs, `ffsq_i,b` is additionally averaged over a 16-event sliding
buffer for the same bunch index and frequency pixel.

For display or downstream tools that expect magnitude rather than squared
magnitude, use:

```text
|F_i,b| = sqrt(ffsq_i,b)
```

PhaseWeaver does this square-root conversion when loading CRISP HDF5 data for
`MeasuredFormFactor`, while preserving the original squared values for the CRISP
reconstruction input.

## Source Map

- `src/EqFctCRISPformfactor.cc`: main service flow, normalization formula,
  output properties, baseline/detection-limit calculation.
- `src/EqFctCRISPformfactor.h`: property declarations and comments describing
  the 240-bin form factor arrays.
- `src/D_CRISPshortarray.cc`: pile-up correction and low/high response handling.
- `src/D_CRISPshortarray.h`: per-channel metadata and corrected signal property.
- `src/formfactorMeanStd.cc`: 16-event sliding mean and standard deviation.
- `src/formfactorMean.cc`: 16-event sliding mean for detection-limit values.
- `src/CRISPdefines.h`: constants such as `MAX_SIG = 120` and
  `FORMFACTOR_SIZE = 240`.
