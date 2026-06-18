# Ocean Insight Relative Form Factor Path

This note documents a practical path for using the Ocean Insight / NIRQuest512
spectrometer data in PhaseWeaver when no full absolute calibration is available.

The short version: without a measured transfer function or absolute response
calibration, the Ocean spectrum should not be treated as an absolute CRISP-like
form factor. It can still be useful as a relative near-infrared shape constraint.

## Available Measurement

The Ocean Insight server publishes spectra as intensity versus wavelength:

```text
wavelength_nm -> intensity_arb
```

For the NIRQuest512 setup discussed here, the wavelength range is approximately:

```text
896 nm ... 2515 nm
```

This corresponds to:

```text
334.6 THz ... 119.2 THz
```

The frequency range therefore matches the current PhaseWeaver infrared band
well:

```text
IR_MIN_HZ = 120 THz
IR_MAX_HZ = 333 THz
```

## What Is Missing For An Absolute CRISP-Like Form Factor

CRISP publishes a calibrated form-factor-squared quantity:

```text
|F(f)|^2 = corrected_detector_signal(f) / charge^2 / detector_response(f)
```

For Ocean Insight data, the following pieces are missing:

- dark/background subtraction, unless a matching no-beam spectrum is recorded
- detector and optical response versus wavelength or frequency
- absolute conversion from spectrometer counts to spectral intensity
- full beamline transfer function, including windows, mirrors, filters, fibers,
  apertures, and alignment effects
- a trusted constant mapping corrected Ocean intensity to `charge^2 * |F|^2`
- per-frequency detection limit or noise estimate

Online/vendor information can help with wavelength calibration and general
spectrometer behavior, but it cannot provide the full response of the installed
beamline. Even a vendor irradiance calibration would only apply to the calibrated
spectrometer input configuration, not automatically to the full experiment.

## Plausible Interpretation

The plausible and honest interpretation is:

```text
Ocean NIR data = relative high-frequency shape information
```

It should be used as a relative `|F|`-like curve, not as an absolute form factor
measurement. This can still constrain:

- where the near-infrared spectrum has support
- where it falls below noise
- relative peaks or dips inside the Ocean band
- high-frequency cutoff behavior
- shot-to-shot changes, if the setup is stable

The unknown detector/optical response means the broad spectral slope and
absolute amplitude should be treated carefully.

## Recommended First Processing

Given a beam spectrum and, ideally, a dark or no-beam spectrum recorded with the
same integration time and settings:

```text
signal(lambda) = intensity_beam(lambda) - intensity_dark(lambda)
signal(lambda) = max(signal(lambda), 0)
```

Convert wavelength to frequency:

```text
frequency_thz = 299792.458 / wavelength_nm
frequency_hz = frequency_thz * 1e12
```

Because frequency decreases as wavelength increases, sort the result by
increasing frequency.

Convert intensity-like data to a provisional magnitude:

```text
mag_like(f) = sqrt(signal(f))
```

Normalize robustly:

```text
scale = percentile(mag_like, 95)
mag_relative(f) = clip(mag_like(f) / scale, 0, 1)
```

This gives:

```text
frequency_hz, relative |F|
```

The result should be labeled clearly as:

```text
Ocean NIR relative |F|
```

## Optional Processing Steps

### Charge Normalization

If matching bunch charge is available, it may help with shot-to-shot comparison:

```text
ffsq_like(f) = signal(f) / charge^2
mag_like(f) = sqrt(ffsq_like(f))
```

This does not correct the wavelength-dependent response, so the final curve
should still be treated as relative unless a response calibration is added.

### Slow-Trend Removal

If the broad spectral envelope is dominated by unknown detector or optical
response, an optional high-pass style correction can be explored:

```text
shape_only(f) = signal(f) / smooth_large_scale(signal(f))
```

This may preserve local structure while suppressing slow response variations.
Use this carefully: a real bunch form factor can also contain broad spectral
structure, so this step should remain optional and visible in diagnostics.

### Noise And Detection Limit

Repeated dark spectra or repeated no-beam measurements can provide a
frequency-dependent noise floor:

```text
noise(f) = std(dark_spectra(f))
detection_limit(f) = k * noise(f)
```

Points below this limit should be down-weighted or masked.

## Combination With CRISP

The most defensible combined measurement model is:

```text
CRISP: calibrated low/mid-frequency |F| or |F|^2 constraint
Ocean: relative high-frequency shape constraint
```

If CRISP and Ocean overlap in frequency, Ocean can be scaled to match CRISP in
the overlap region. If there is no overlap, Ocean should be normalized
independently and used with lower confidence.

In reconstruction terms:

```text
CRISP -> hard or high-weight amplitude constraint
Ocean -> soft, relative high-frequency shape constraint
```

## Use In Reconstruction

The Ocean curve can later enter reconstruction as another measured magnitude
curve:

```text
MeasuredFormFactor(freq=ocean_frequency_hz, mag=ocean_relative_mag)
```

This fits the existing PhaseWeaver measurement abstraction, but the meaning is
different from CRISP. CRISP is a calibrated or near-calibrated form-factor
magnitude. Ocean is only a relative shape measurement.

During a Gerchberg-Saxton-style iteration, the combined flow is:

```text
1. transform current profile to form factor
2. apply measured CRISP magnitude in the low/mid-frequency band
3. softly apply Ocean relative magnitude shape in the NIR band
4. transform back to time domain
5. apply time-domain constraints, such as non-negativity and normalization
```

The important point is scaling. Since the Ocean curve is relative, it should not
force the absolute high-frequency amplitude. It should guide the shape of the
NIR band and the high-frequency cutoff.

### First Implementation

The simplest implementation can reuse the existing measured-magnitude blending
logic with scaling enabled:

```text
measurements = (
    crisp_measured,
    ocean_measured,
)

BlendMeasuredMagnitude(
    measurements,
    scale=True,
)
```

With scaling enabled, each measured curve is scaled to the current reconstructed
magnitude level in its own frequency band before blending. This makes the Ocean
curve act more like a local shape constraint than an absolute amplitude
constraint.

This is a reasonable first step, especially for visualization and algorithm
experiments.

### Preferred Longer-Term Implementation

Longer term, Ocean should probably use a dedicated relative-shape constraint,
separate from the CRISP amplitude constraint:

```text
CRISP constraint:
    enforce or strongly blend absolute |F|

Ocean constraint:
    compare normalized NIR shapes
    preserve the current overall NIR amplitude scale
    softly nudge the reconstruction toward the Ocean shape
```

One possible shape-only comparison is:

```text
recon_band_norm = recon_band / percentile(recon_band, 95)
ocean_norm = ocean_mag / percentile(ocean_mag, 95)
shape_error = recon_band_norm - ocean_norm
```

The correction should then adjust the reconstructed magnitude in the Ocean band
without importing Ocean's arbitrary absolute scale. A possible constraint name
would be:

```text
BlendRelativeMeasuredShape
```

Useful parameters would be:

```text
weight
normalization_percentile
transition_width
frequency_mask or valid/noise mask
```

This would let PhaseWeaver treat CRISP and Ocean differently:

```text
CRISP: strong calibrated amplitude information
Ocean: weaker relative NIR shape information
```

### What Ocean Can Help With

Used this way, Ocean can help the reconstruction prefer solutions with:

- a plausible high-frequency cutoff
- NIR structure consistent with the spectrometer measurement
- stable shot-to-shot changes in the high-frequency band
- fewer arbitrary high-frequency tails when CRISP alone has limited coverage

It should not be used to determine the absolute high-frequency amplitude unless
a real response calibration is added later.

## Data To Collect

The smallest useful bundle is:

```text
ocean_signal: wavelength_nm, intensity
ocean_dark: wavelength_nm, intensity
timestamp or macropulse ID
integration time
trigger mode
```

Helpful additions:

```text
matching bunch charge
several repeated dark spectra
several repeated beam spectra
CRISP shot from the same or nearby machine state
notes about filters, fibers, windows, and optical alignment
```

## Proposed PhaseWeaver Implementation

Add an Ocean/NIR measurement loader that:

1. reads wavelength and intensity arrays
2. optionally reads a matching dark spectrum
3. subtracts dark and clips negative values
4. converts wavelength to frequency
5. sorts by increasing frequency
6. computes `sqrt(signal)`
7. normalizes by a robust percentile
8. returns a `MeasuredFormFactor(freq=frequency_hz, mag=mag_relative)`

The loader should preserve metadata that records:

- source file
- wavelength range
- frequency range
- whether dark subtraction was applied
- normalization percentile
- whether charge normalization was applied
- whether slow-trend removal was applied

The UI and exports should avoid calling this an absolute form factor. Preferred
labels are:

```text
Ocean NIR relative |F|
Ocean NIR shape constraint
```

## Working Assumption

Until a real response calibration is available, the Ocean data is best used to
shape the high-frequency reconstruction band rather than to set the absolute
form-factor amplitude.
