# PhaseWeaver Project State

Last updated: 2026-06-21

## Executive Summary

PhaseWeaver is now moving from a generic magnitude-only reconstruction toy
workbench toward a CRISP-aware current-profile reconstruction tool. The PySide6
app still supports the original toy-model workflow, `.npz` measurement bundles,
and Gerchberg-Saxton reconstruction, but the current implementation also has a
first CRISP path:

- CRISP `.h5`/`.hdf5` measurement import from the XFEL HDF5 keys used by the
  available recording.
- Latest-shot selection by timestamp, with explicit shot-index support in the
  loader.
- GUI shot selection for multi-shot HDF5 files, defaulting to the latest shot.
- Conversion of CRISP form-factor-squared values to `MeasuredFormFactor`
  magnitudes for plotting and generic reconstruction.
- Preservation of the original CRISP `|F|^2` values, standard deviation,
  detection limit, shot index, timestamp, and charge for CRISP-style
  reconstruction.
- Use of CRISP reconstruction input arrays when present: `INPUT_FFSQ`,
  `INPUT_FFSQ_STD`, and `INPUT_FFSQ_DETECTION_LIMIT`.
- Loading of the upstream SA1 `CURRENT_PROFILE` from HDF5 as a reference current
  overlay for direct comparison with PhaseWeaver's CRISP reconstruction.
- Loading of Ocean/NIR `SPECTRUM` data from HDF5 as a relative high-frequency
  `|F|` overlay, excluded from normal reconstruction by default.
- Optional extended reconstruction mode where loaded Ocean/NIR data is used as a
  relative high-frequency shape constraint while CRISP remains the calibrated
  measurement and error source. This works for Gerchberg-Saxton directly and as
  a CRISP-seeded `CRISP + IR` extension.
- IR relative-shape scaling can be automatic, matching the average
  reconstructed `|F|` over the IR band, or fixed by a user-controlled scale
  factor with a live log slider.
- A `CRISP` reconstruction algorithm option beside `Gerchberg-Saxton`.
- A Python implementation of the upstream CRISP reconstruction flow, including
  input filtering, low/high-frequency Gaussian extrapolation, Savitzky-Golay
  smoothing, Kramers-Kronig phase, positive-peak isolation, modulus-band
  iteration, current scaling, and diagnostics.
- Export of CRISP diagnostics alongside the ordinary plot arrays and
  reconstruction summary.

The new docs capture the physics/software sources behind this direction:

- `docs/formfactor_calculation.md`: how the CRISP form-factor service creates
  calibrated `|F|^2` from detector channels, response, and charge.
- `docs/crisp_reconstruction_algorithm.md`: how the upstream CRISP
  reconstruction server turns `|F|^2` into a current profile.
- `docs/ocean_insight_server.md`: what the Ocean Insight/NIRQuest server
  publishes: wavelength-calibrated intensity spectra, not form factors.
- `docs/ocean_relative_formfactor_path.md`: how Ocean/NIR spectra can be used
  honestly as a relative high-frequency shape constraint until an absolute
  response calibration exists.

## Where We Are Heading

The near-term target is a trustworthy reconstruction workbench for comparing
toy/generated profiles, CRISP measurements, Ocean/NIR spectra, and reconstructed
current profiles without hiding which data are calibrated and which are only
relative.

The intended data interpretation is:

- CRISP form-factor service output is calibrated form-factor squared:
  `|F(f)|^2 = corrected_signal / charge^2 / detector_response`.
- PhaseWeaver plots and generic reconstruction use `|F|`, so CRISP `|F|^2`
  becomes `sqrt(|F|^2)` for `MeasuredFormFactor`.
- The CRISP reconstruction algorithm should keep using the original `|F|^2`
  input because that is what the upstream algorithm expects.
- Ocean/NIR spectra are wavelength/intensity data in arbitrary units. They
  should not be presented as absolute form factor without a measured response
  calibration. Their current reconstruction role is opt-in relative
  high-frequency shape guidance, either average-matched to the reconstructed IR
  band level or multiplied by a fixed user-selected scale.

## What The App Does Today

The active GUI remains a focused plotting workbench:

- Main view: time-domain current/profile plot and frequency-domain
  magnitude/phase plot.
- `File > Load Measurements...`: loads `.npz`, `.h5`, and `.hdf5` measurement
  files, including CRISP form-factor data and Ocean/NIR spectra from HDF5 when
  available.
- `File > Export Data...`: writes plot arrays, loaded measurement arrays,
  profile/reconstruction settings, reconstruction summary, and CRISP diagnostics
  when available.
- `View > Toy Model`: opens the configurable background/spike/spike-2 profile
  generator.
- `View > Reconstruction Setup`: opens algorithm selection, phase init,
  simulated CRISP/IR fallback selectors, constraints, stop conditions, auto
  reconstruction, IR relative-constraint controls, and run controls.
- Status summary: source, algorithm, calibrated measurement count, relative IR
  constraint state, run state, iterations, stop reason, and final
  Gerchberg-Saxton measurement error when applicable.

The reconstruction workflow is now:

1. Build or update the toy-model input profile.
2. Compute its reference form factor for visualization and simulated fallback
   measurements.
3. Optionally load `.npz` or CRISP HDF5 measurements.
4. Pick `Gerchberg-Saxton` or `CRISP`.
5. Optionally enable `Use IR as relative constraint` for Gerchberg-Saxton or
   CRISP.
6. Run reconstruction from the active calibrated measurement set, with optional
   relative IR shape guidance.
7. Render the reconstructed current/profile and form factor.
8. Export the run with enough metadata to inspect it later.

## Current Repository Snapshot

- Branch: `main`
- Package version: `0.6.0`
- Python package layout: `src/phase_weaver`
- GUI framework: PySide6
- Plotting: pyqtgraph
- Core numerical dependencies: NumPy, SciPy
- HDF5 dependency: h5py
- Test runner: Pytest
- Type checker: basedpyright
- Linter: Ruff

Current notable local changes before this state update:

- New docs under `docs/`.
- New `src/phase_weaver/core/crisp_reconstruction.py`.
- New CRISP reconstruction tests in `tests/core/test_crisp_reconstruction.py`.
- App logic updates for `.h5/.hdf5` loading, CRISP algorithm dispatch, and CRISP
  export diagnostics.
- UI/state updates for algorithm selection and persistence.
- Plot model/panel fixes so reconstruction data may live on a different grid
  than the input model, and manual axis ranges survive refreshes.
- GUI shot selection for multi-shot CRISP HDF5 recordings.
- CRISP HDF5 reconstruction inputs now use `INPUT_FFSQ`, `INPUT_FFSQ_STD`, and
  `INPUT_FFSQ_DETECTION_LIMIT` when those datasets are available, while keeping
  older files compatible.
- CRISP HDF5 SA1 current profiles now load as reference overlays and export with
  the measurement metadata when available. Their time axis is inferred from
  `charge = sum(current) * dt`, which also recovers the upstream reconstruction
  maximum frequency for loaded CRISP reconstructions.
- Ocean/NIR HDF5 spectra now load as `Ocean NIR relative |F|` overlays with
  relative-shape metadata. They remain excluded from normal reconstruction by
  default, but can be used as an explicit opt-in relative-shape constraint.
- Waterflow NPZ dumps now load `average` as an Ocean/NIR relative `|F|` overlay.
  The file's `background` is treated as contextual raw baseline data; it is not
  subtracted because `average` is already background-corrected in the current
  test dump.
- Cor2d NPZ dumps now load the mean of `spec_hist` over the `phen_scale`
  wavelength axis as an Ocean/NIR relative `|F|` overlay.
- `File > Load IR Measurement...` can replace only loaded Ocean/NIR overlays,
  preserving the loaded CRISP HDF5 data, CRISP reconstruction input, charge,
  shot metadata, and SA1 reference current profile.
- `Use IR as relative constraint` in the reconstruction setup enables an
  opt-in relative-shape constraint from loaded Ocean/NIR data. For
  Gerchberg-Saxton it is part of the normal iteration. For the CRISP algorithm,
  the app first runs the upstream-style CRISP reconstruction and then runs a
  CRISP-seeded `CRISP + IR` extension. Normal CRISP-only reconstruction remains
  unchanged when the toggle is off.
- `Use fixed IR scale` switches IR scaling from automatic band-average matching
  to a fixed user-selected multiplier. The companion `IR scale` control has a
  log slider for interactive sweeps and a numeric spinbox for exact values.
- `notebooks/scratch.ipynb` removed from the working tree.
- Optional reference-HDF5 comparison test that uses
  local CRISP HDF5 recordings only when available; the
  recording itself is sample data and is not part of the committed project
  state.

## Implemented CRISP Path

The current CRISP implementation is intentionally close to the documented
upstream server:

- Input validation and sorting on positive THz frequencies.
- Neighbor masking around NaNs and points below detection limit.
- Cutoff after a run of bad points.
- Clamping `|F|^2` to `[0, 1]`.
- Low-frequency Gaussian fit from early valid points.
- High-frequency Gaussian from the last valid measured point.
- Extrapolation toward 0 THz and up to the configured maximum frequency.
- 9-point, order-3 Savitzky-Golay smoothing with edge preservation.
- Interpolation to the positive half-grid and conversion to `|F|`.
- Kramers-Kronig phase calculation.
- Symmetric complex spectrum construction and iFFT start profile.
- Up to 20 iterations of positive-main-peak isolation, FFT/DC normalization,
  measured modulus replacement outside the 20 percent band, iFFT, and recentering.
- Final current scaling by charge and time step.
- Diagnostics for intermediate arrays, first iteration profiles, iterations,
  peak current, FWHM, RMS width, and skewness.

Known differences and caveats:

- HDF5 loading falls back to zero standard deviation and zero detection limit
  for older recordings that do not contain the CRISP reconstruction input
  datasets.
- The Python port is covered by synthetic tests and one optional reference-HDF5
  comparison, but it should still be validated against more upstream/reference
  cases.
- The generic app measurement abstraction stores magnitudes, while CRISP
  diagnostics and the CRISP algorithm need squared magnitudes. The current code
  handles this split, but future loader/export schemas should make it explicit.
- Ocean/NIR data is still relative, not calibrated. The extended reconstruction
  path either scales its shape to the current reconstructed magnitude average in
  the IR band or applies a fixed user scale. It never treats Ocean/NIR
  magnitudes as absolute calibrated form factor.

## Verification Results

Latest verified before this commit:

```bash
uv run pytest tests
```

Result: pass.

- 290 tests collected.
- 289 passed.
- 1 skipped because the optional reference HDF5 recording was not present in
  the checked path.

```bash
uv run ruff check src/phase_weaver/app/logic.py src/phase_weaver/core/reconstruction.py src/phase_weaver/core/constraints.py src/phase_weaver/app/ui/reconstruction_panel.py src/phase_weaver/app/ui/main_window.py tests/app/test_logic.py tests/app/test_controls_panel.py tests/core/test_reconstruction.py
```

Result: pass.

```bash
QT_QPA_PLATFORM=offscreen uv run python -c "<construct QApplication and MainWindow, quit after 200 ms>"
```

Result: pass.

The app window constructs and enters/exits the Qt event loop in offscreen mode.

Static tooling status has not been made a release gate yet:

- `ruff check .` still needs a focused cleanup pass before it should be treated
  as a required green check.
- `basedpyright` still has known PySide/test typing noise and architecture
  cleanup work before it can be used as a hard gate.

## Steps Forward

1. Validate the CRISP Python reconstruction against more real/reference data.
   Compare peak current, FWHM, RMS width, skewness, intermediate arrays, and
   iteration count against the upstream server where possible.

2. Extend CRISP HDF5 loading with richer metadata.
   Standard deviation and detection limit are now loaded when present. Next add
   explicit measurement kind, source-file identity, calibration state, event
   identity, and processing choices. Keep both `|F|^2` and `|F|` semantics
   explicit.

3. Validate and tune the CRISP+IR extension on more real shots.
   Compare automatic average matching and fixed IR scaling, and inspect whether
   the resulting high-frequency shape is physically plausible against known
   bunch profiles.

4. Clarify export/import schema.
   Store measurement kind, squared-vs-magnitude semantics, source file, shot
   index, timestamp, charge, calibration state, and processing choices.

5. Clean static tooling drift.
   Bring Ruff close to green first. Treat basedpyright as a later architecture
   pass because several issues are PySide stub friction or broader type design
   questions.

6. Keep the main UI focused.
   Preserve the two-plot main window. Put data selection, shot selection,
   reconstruction settings, and future Ocean processing controls into explicit
   menus or tool windows.

## Useful Commands

```bash
uv run pytest tests
```

Runs the active unit suite.

```bash
QT_QPA_PLATFORM=offscreen uv run python -c "import sys; from PySide6.QtCore import QTimer; from PySide6.QtWidgets import QApplication; from phase_weaver.qt_theme import set_dark_theme; from phase_weaver.app.ui.main_window import MainWindow; app=QApplication(sys.argv); set_dark_theme(app); w=MainWindow(); w.show(); QTimer.singleShot(200, app.quit); raise SystemExit(app.exec())"
```

Runs a headless GUI construction smoke test.

```bash
uv run ruff check .
```

Finds lint/static cleanup items.

```bash
uv run basedpyright
```

Runs the current type checker.

```bash
uv run phase_weaver
```

Runs the GUI app in a normal desktop session.

## Bottom Line

PhaseWeaver now has the first real bridge from documented CRISP server behavior
into the app: import CRISP HDF5, choose a shot, compare against the upstream SA1
current profile, reconstruct with a CRISP-like algorithm using upstream-style
reconstruction input arrays when available, inspect diagnostics, and export the
run. It also loads Ocean/NIR spectra as clearly labeled relative `|F|` data and
can use them as opt-in high-frequency relative-shape constraints, with automatic
band-average scaling or a fixed interactive IR scale. The next step is
calibration-aware validation and schema work rather than hiding relative IR data
behind absolute-form-factor semantics.
