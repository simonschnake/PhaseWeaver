# Ocean Insight Spectrometer Server

This note summarizes the Ocean Insight DOOCS server used for the infrared
spectrometer spectrum. The source inspected here is
`/Users/schnakes/Downloads/oceaninsightspectrometer-main.zip`.

The server connects to Ocean Insight spectrometers through the OceanDirect API,
reads wavelength-calibrated spectra, publishes them through DOOCS, and sends an
interpolated spectrum to ZMQ/DAQ.

## What The Server Provides

Each spectrometer location exposes configuration, device metadata, live spectrum
data, timing information, status/errors, and debug counters.

The most useful measurement outputs are:

- `SPECTRUM`: measured spectrum as `D_xy`, with wavelength on `x_data` in nm and
  intensity on `y_data` in arbitrary units.
- `SPECTRUM.INTERPOLATED`: same spectrum interpolated onto an equidistant
  wavelength axis, published as `D_spectrum` for ZMQ and DAQ.

The README describes the DAQ sub-channel as:

```text
SPECTRUM.INTERPOLATED, arb. units vs. nm, spectrum interpolated to equi-distant wavelengths
```

So for IR data consumers, the main physical information is:

```text
wavelength_nm -> intensity_arb
```

The server does not directly calculate form factors. It provides the infrared
spectrum that later code can use for analysis or reconstruction.

## Device Discovery And Connection

The server supports runtime-created locations of type:

```text
eq_fct_type: 10
```

Each location is tied to a physical spectrometer by:

- `SERIALNUMBER`: required serial number of the spectrometer.
- `OCEANFX.HOSTNAME_OR_IP`: optional host/IP for OceanFX Ethernet/TCP devices.

`OceanInsight` is a singleton wrapper around the OceanDirect API. It probes
devices when requested, tracks connected device IDs, and exposes choices through:

- `SERIALNUMBER.OPTIONS`: serial/model pairs suitable for a dropdown.

The probe loop can discover USB devices and explicitly configured OceanFX TCP
devices. TCP OceanFX devices are added with port `57357`.

A location refuses readout if another location is configured with the same
serial number. In that case it reports an `open_error` saying the device is in
use by another location.

## Configuration Inputs

Important configurable properties are:

- `SERIALNUMBER`: selected spectrometer serial number.
- `OCEANFX.HOSTNAME_OR_IP`: optional Ethernet host/IP for OceanFX.
- `INTEGRATIONTIME [microseconds]`: integration time.
- `AQUISITIONDELAY [microseconds]`: acquisition delay.
- `TRIGGERMODE`: Ocean Insight trigger mode. `0` means free-running; other modes
  are model-specific.
- `CONF.TIMERADDR`: base x2Timer address for macropulse/timestamp annotation.
- `CONF.POWERCYCLE.ADDR`: optional DOOCS address for power cycling a stuck
  device.
- `LINK.ANALYSIS_SERVER`: helper link for analysis tooling/jDDD navigation.
- `COMMENT`: operator-facing free text.

If the spectrometer is already connected when integration time, acquisition
delay, or trigger mode is changed, the property setter immediately forwards the
new value to the device.

Integration time and acquisition delay are clamped by the device wrapper to the
hardware-reported min/max limits before being applied.

## Device Metadata

When a readout thread successfully opens a device, the location publishes
metadata and limits:

- `SPECTROMETER.MODEL`: model type reported by OceanDirect.
- `SPECTROMETER.FIRMWARE_REVISION`: declared in the interface, but the current
  code reads firmware into a local buffer and does not publish it.
- `SPECTROMETER.WAVELENGTH.MIN`: first wavelength in nm.
- `SPECTROMETER.WAVELENGTH.MAX`: last wavelength in nm.
- `SPECTROMETER.INTEGRATIONTIME.MIN`: minimum integration time in microseconds.
- `SPECTROMETER.INTEGRATIONTIME.MAX`: maximum integration time in microseconds.
- `SPECTROMETER.ACQUISITIONDELAY.MIN`: minimum acquisition delay in microseconds.
- `SPECTROMETER.ACQUISITIONDELAY.MAX`: maximum acquisition delay in microseconds.
- `SPECTROMETER.SPECTRUM_LENGTH`: number of pixels/samples in a spectrum.
- `SPECTROMETER.MAX_INTENSITY`: maximum intensity reported by the device.

The wavelength axis is read once on connect with `odapi_get_wavelengths()` and
used to initialize `SPECTRUM`.

## Readout Flow

The location's `update()` runs once per second. If the location is online, it
starts a readout thread unless one is already running.

The readout thread:

1. checks that no other location uses the same serial number
2. finds the selected device by serial number
3. opens the spectrometer
4. publishes model, limits, spectrum length, max intensity, and wavelength range
5. reads the wavelength axis
6. applies configured integration time, acquisition delay, and trigger mode
7. repeatedly reads spectra with `odapi_get_formatted_spectrum()`
8. hands each spectrum to an asynchronous sender worker

In free-running trigger mode (`TRIGGERMODE = 0`), the loop is software-limited
to about 20 Hz by enforcing a minimum cycle time of 0.05 s.

In triggered modes, the README says up to 1 kHz has been tested. If no new
spectrum arrives for more than 5 s while trigger mode is nonzero, `update()`
sets a `not_responding` error with message `no trigger`; the readout thread then
aborts.

## `SPECTRUM`

`SPECTRUM` is the direct wavelength-calibrated spectrum from the device.

On connect:

```text
SPECTRUM.length = number of wavelengths
SPECTRUM.x_data[i] = wavelength_nm[i]
SPECTRUM.y_data[i] = 0
```

For each read spectrum:

```text
SPECTRUM.y_data[i] = formatted_spectrum[i]
```

The `x_data` axis comes from OceanDirect's wavelength calibration. It may be
non-equidistant, which is why a separate interpolated output exists.

The units are:

- x-axis: nm
- y-axis: arbitrary intensity units

## `SPECTRUM.INTERPOLATED`

`SPECTRUM.INTERPOLATED` is the DAQ/ZMQ-friendly copy. It is a `D_spectrum`,
which expects an equidistant x-axis, so the server linearly interpolates the
`D_xy` spectrum.

For every acquired spectrum, the server:

1. chooses the current DAQ/ZMQ ring buffer from `event_id % 16`
2. writes macropulse/event ID and timestamp metadata
3. sets the spectrum length to `spec.size()`
4. defines an equidistant wavelength grid
5. linearly interpolates the intensity from `SPECTRUM`
6. sends ZMQ and prepares DAQ data

The wavelength grid is:

```text
wavelenMin = SPECTROMETER.WAVELENGTH.MIN
wavelenMax = SPECTROMETER.WAVELENGTH.MAX
wavelenIncr = (wavelenMax - wavelenMin) / spec.size()
wavelen[i] = wavelenMin + i * wavelenIncr
```

The interpolation is linear between the neighboring original `SPECTRUM.x_data`
points:

```text
I_interp = I_left + (I_right - I_left) * ((wavelen - wl_left) / (wl_right - wl_left))
```

This means consumers can get an evenly spaced spectrum suitable for DAQ and
streaming, while `SPECTRUM` remains the direct calibrated XY output.

## Timing And Macropulse Information

If `CONF.TIMERADDR` is set, `update()` derives two subscription addresses:

- `CONF.MACROPULSEADDR = CONF.TIMERADDR ///MACRO_PULSE_NUMBER`
- `CONF.TRIGMASKADDR = CONF.TIMERADDR ///TRIG_MASK`

Callbacks store:

- `SYSTEM.MACRO_PULSE_NUMBER`: latest macropulse number.
- `SYSTEM.TRIG_MASK`: latest trigger mask.
- `SYSTEM.STAT`: status from the timer message.

The latest macropulse and timestamp are attached to both `SPECTRUM` and
`SPECTRUM.INTERPOLATED`. If no macropulse event ID has been received, the server
uses the current wall-clock time as fallback for the timestamp.

DAQ data is prepared with subchannel `SUBCHAN_SPECTRUM = 0` and the latest
trigger mask.

## Status And Recovery

The server reports useful status through DOOCS errors:

- offline location: `device_offline`
- selected serial not found: `open_error`, with the serial number in the message
- duplicate serial in another location: `open_error`
- triggered mode without data for more than 5 s: `not_responding`, `no trigger`
- no running readout thread: `SPECTRUM` and `SPECTRUM.INTERPOLATED` are marked
  `stale_data`

If stopping the readout thread takes more than 5 s, the server assumes the
readout may be stuck. If `CONF.POWERCYCLE.ADDR` is configured, it writes:

```text
0  -> power down
1  -> power up
```

This is intended to recover from a stuck triggered readout, for example when the
device is waiting for a trigger that never comes.

## Debug Information

The server exposes counters useful for monitoring:

- `DEBUG.PROBE_THREAD.CYCLE`: OceanDirect probe cycle count.
- `DEBUG.READOUT_THREAD.CYCLE`: readout loop cycle count.
- `SPECTRUM_SENDER.CYCLE`: asynchronous sender worker cycle count.
- `SPECTRUM_SENDER.CYLCE_TIME [microseconds]`: sender handling time. The
  property name contains the typo `CYLCE` in the source.

These are operational health indicators rather than scientific data.

## What Information We Can Use

For PhaseWeaver or other spectrum consumers, the server can provide:

- wavelength axis in nm
- infrared intensity spectrum in arbitrary units
- interpolated equidistant spectrum for DAQ/ZMQ
- raw calibrated XY spectrum with the hardware wavelength axis
- acquisition timestamp
- macropulse/event ID, when x2Timer is configured
- trigger mask, when x2Timer is configured
- spectrometer model
- spectrum length
- wavelength range
- integration-time and acquisition-delay limits
- max device intensity
- live integration time, acquisition delay, and trigger mode
- device status/error state

The server does not provide:

- frequency-domain form factor values directly
- a square-root or charge-normalized form factor
- background subtraction
- wavelength-to-frequency conversion
- detector-response correction beyond what OceanDirect returns as the formatted
  spectrum

Any conversion from IR spectrum to form-factor-like data must happen downstream.

## Source Map

- `README.md`: operational overview and public property list.
- `oceaninsight_CONF.TEMPLATE`: example DOOCS configuration.
- `src/EqSpectrometer.h`: DOOCS properties exposed by each spectrometer
  location.
- `src/EqSpectrometer.cc`: readout loop, spectrum publishing, interpolation,
  timing subscription, DAQ/ZMQ sending, and power cycling.
- `src/OceanInsight.h` and `src/OceanInsight.cc`: OceanDirect API wrapper,
  device probing, connection, limits, wavelength reads, and spectrum reads.
- `src/D_worker.h`: asynchronous sender worker used for spectrum publishing.
- `src/main.cc`: server setup, dynamic location support, update period, DAQ
  startup, and version string.
