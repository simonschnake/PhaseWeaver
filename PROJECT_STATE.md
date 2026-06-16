# PhaseWeaver Project State

Last updated: 2026-06-16

## Executive Summary

PhaseWeaver is currently a small Python/PySide6 scientific workbench for exploring current-profile reconstruction from form-factor magnitude data. The app can generate a configurable time-domain current profile, compute its spectrum magnitude and phase, simulate partial measurements, load `.npz` measurement bundles, and run a Gerchberg-Saxton-style magnitude-only reconstruction.

The project is functional enough that the full pytest suite passes and the Qt main window can be constructed in an offscreen smoke test. The last-phase reconstruction bug has been fixed, the first experiment-use workflow is in place, and the plots now render correctly on initial load instead of only after running reconstruction once. Recent work also added theme switching, reconstruction history capture, a colorblind-friendly plot palette, and some layout/DPI polish.

The next work should shift the app toward a clearer reconstruction workbench: a large two-plot main view for time-domain current/profile and frequency-domain form factor, with model generation, reconstruction configuration, data loading, saving, and export moved out of the crowded central controls.

## Product Goal And Workflow Direction

The target workflow is:

1. Establish source data.
   - Use the current toy model to generate a simulated current profile for now.
   - Later, also load simulated current profiles from files, including Ocelot-style simulation output once the concrete file structure has been inspected.
   - Exact simulation-file schemas are intentionally deferred until real example files are available.

2. Derive or load measurements.
   - From a toy-model or loaded current profile, derive synthetic CRISP and infrared spectrometer measurement data.
   - Independently load real measurement bundles that contain CRISP and infrared spectrometer data.
   - Measurement loading and profile loading must remain independent; loaded measurements do not necessarily need to correspond to the currently generated or loaded profile.

3. Compare visually first.
   - The near-term comparison should be primarily visual: overlay or otherwise compare generated/simulated measurements with loaded measurements.
   - Residuals, numerical metrics, and richer diagnostics are useful later but should not block the next UI restructuring.

4. Reconstruct from the active measurement set.
   - Reconstruction should use the selected loaded or generated measurements.
   - Reconstruction configuration should remain part of a shared app/session data model, but its controls should move out of the main view.

5. Keep the main view focused.
   - The main window should be centered on two large plots:
     - left: current/profile in time,
     - right: Fourier/form-factor view in frequency.
   - Secondary controls should be available through menus and optional view windows, not permanently crowding the main plotting workspace.

Important deferred decisions:

- The project will keep `.npz` as the first measurement-bundle format, but the exact schema should be revised only after real CRISP/IR files and simulation files are in hand.
- Simulated profile loading is not the first implementation priority.
- When file-based simulation data exists, the toy model may need an explicit enabled/disabled state so it no longer competes with loaded profiles.

## Operational Readiness

If the goal is to keep PhaseWeaver useful while the UI is being reworked, the important path is now:

1. Keep measurement loading available.
   - Supported input is currently one `.npz` file.
   - Single-measurement schema: `freq_hz`, `mag`, optional `label`.
   - Multi-measurement schema: `freq_hz_0`, `mag_0`, `label_0`, `freq_hz_1`, `mag_1`, `label_1`, etc.
   - Frequencies are in Hz; magnitudes are dimensionless `|F|`.

2. Keep reconstruction deliberate.
   - Parameter changes redraw the input/model only.
   - GS runs only when the operator presses `Run Reconstruction`.
   - Loaded measurements override simulated CRISP/IR measurements; simulated measurements remain as a demo fallback.

3. Compare model, measurements, and reconstruction.
   - The super-Gaussian model `|F|` stays visible.
   - Loaded measurements render as spectrum markers.
   - Reconstructed `|F|` and phase render after a successful reconstruction run.

4. Move data actions toward the menu bar.
   - Export uses a save dialog instead of always overwriting `export.npz`.
   - The `.npz` export includes plot arrays, loaded measurement arrays/labels, profile parameters, phase init mode, and reconstruction summary fields.
   - Loading, saving, and export should migrate under a `File` menu rather than remain as large persistent controls.

5. Preserve the reconstruction overview.
   - The GUI shows measurement source, measurement count, reconstruction state, iterations, stop reason, and final measurement error.

6. Keep the UI responsive enough.
   - The current synchronous reconstruction is acceptable for now if it stays fast.
   - If experiment-sized data makes the window stall, move reconstruction off the UI thread next.

## Current Repository Snapshot

- Branch: `main`
- Tracking: `origin/main`
- Local status: `main` is ahead of `origin/main` by 14 commits
- Latest commit: `9912018`
- Worktree state: modified
- Package version: `0.4.0`
- Python package layout: `src/phase_weaver`
- Main executable: `phase_weaver = phase_weaver.app:main`
- Main app entrypoint: `src/phase_weaver/app/main.py`
- GUI framework: PySide6
- Plotting: Matplotlib Qt backend
- Core numerical dependencies: NumPy, SciPy
- Test runner: Pytest
- Type checker: basedpyright
- Linter: Ruff

Current notable uncommitted paths:

- `PROJECT_STATE.md` is modified with the current product/workflow direction.

## What The App Does Today

The app currently presents an interactive current-profile model directly in the main window:

- A background asymmetric super-Gaussian.
- One primary spike.
- Optional second spike.
- Configurable profile parameters through Qt controls.
- Immediate plots for:
  - time-domain current/profile,
  - spectrum magnitude,
  - spectrum phase,
  - reconstructed current/profile.

The reconstruction path currently works like this:

1. UI state is collected in `ControlsState`.
2. `AppLogic.compute_input_profile()` builds a `CurrentProfile`.
3. `Profile.to_form_factor()` computes the reference form factor.
4. The operator may load one `.npz` measurement bundle with one or more measured form-factor magnitudes.
5. The operator presses `Run Reconstruction`.
6. `AppLogic.active_measurements()` uses loaded measurements when available, otherwise simulated CRISP/IR measurements.
7. `GerchbergSaxton` builds an initial form factor by Gaussian-extending measured magnitude data.
8. Iteration alternates between time-domain constraints and frequency-domain constraints.
9. The reconstructed profile, reconstructed spectrum, loaded measurement markers, and reconstruction summary are rendered in the GUI.

This works, but it leaves too much model and reconstruction machinery visible at once. The intended next UI shape is a main plotting workspace plus optional tool windows.

## Core Architecture

Important modules:

- `src/phase_weaver/core/base.py`
  - Defines `Grid`, `Profile`, `CurrentProfile`, `FormFactor`, and FFT transforms.
  - `DCPhysicalRFFT` is the main transform used by app and reconstruction.
  - `BandLimitedDCPhysicalRFFT` exists for cutoff-limited experiments.

- `src/phase_weaver/core/constraints.py`
  - Time constraints include non-negativity, area normalization, centering, and cutting after zeros.
  - Frequency constraints include magnitude clamping, DC normalization, measurement blending, and phase-tail replacement helpers.

- `src/phase_weaver/core/reconstruction.py`
  - Contains the current reconstruction framework.
  - `GerchbergSaxton` is now the main algorithm.
  - Stopping criteria include max iterations, min iterations, phase stability, and measured-magnitude distance.
  - Older classes such as `GSMagnitudeOnly`, initializer classes, and app service wrappers appear to have been removed.

- `src/phase_weaver/app/state.py`
  - Holds UI state dataclasses and the profile model.
  - The profile model currently lives here, while tests previously expected it under `phase_weaver.model.profile_model`.

- `src/phase_weaver/app/logic.py`
  - Bridges UI state to profile generation, measurement loading/simulation, reconstruction, summary reporting, and NPZ export.

- `src/phase_weaver/app/ui/`
  - Contains Qt widgets for controls and plots.
  - The current controls panel uses generic option selector widgets for phase init and measurement source selection.
  - `plot_controls_box.py` keeps only line-visibility controls; normalized time plotting is not exposed.
  - Reconstruction history is preserved in-memory and exported alongside the main arrays, which makes the latest reconstruction runs easier to inspect later.

## Verification Results

Commands run on 2026-06-15:

```bash
uv run pytest
```

Result: pass.

- 158 tests collected.
- 158 passed.

```bash
uv run pytest tests
```

Result: pass.

- 158 tests collected.
- 158 passed.

```bash
uv run ruff check .
```

Result: fail.

Main causes:

- Stale notebook imports.
- Unused imports in stale/dev code and older support modules.
- Broad workspace drift outside the focused fix area.

```bash
uv run basedpyright
```

Result: fail.

There are many errors, but the highest-signal groups are:

- Stale `scripts/reconstruction_dev.py` imports and constructor calls.
- Removed measurement config constants still referenced by `measurement_box.py`.
- Generic enum selector widgets return `Enum`, while callers expect specific enum types.
- PySide6 enum/type-stub mismatches.
- Core `Profile` vs `CurrentProfile` typing is inconsistent.

GUI smoke check:

```bash
QT_QPA_PLATFORM=offscreen uv run python -c "<construct QApplication and MainWindow, quit after 200 ms>"
```

Result: pass.

The app window constructs and enters/exits the Qt event loop in offscreen mode. Qt emitted non-fatal warnings about missing `Sans Serif` font alias and `propagateSizeHints()`.

## Current Problems

### 1. The Test Layout Is Mostly Stable

`pytest` now runs cleanly. The project still needs one canonical documented command for day-to-day contributors, but this is no longer blocking app usage.

### 2. The Main UI Is Overcrowded

The current main window mixes plotting, toy-model definition, measurement controls, reconstruction controls, loading, running, and exporting in one place. This was useful while making the first workflow run, but it is no longer the desired operating shape.

Desired direction:

- Make the main window primarily a two-plot workspace.
- Move the three-super-Gaussian toy model into an optional window opened from the `View` menu.
- Move reconstruction phase/configuration controls into an optional window opened from the `View` menu.
- Move data loading, saving, and export actions into a `File` menu.
- Keep small, high-signal status/run controls in the main window only where they support the plotting workflow directly.

### 3. Measurement UI Still Has Some Drift

The active operator workflow now supports loading real `.npz` measurement bundles. There is still older measurement UI code that is not part of the active flow.

- Current active UI: `ControlsPanel` has load, run, export controls plus simulated CRISP/IR fallback selectors.
- Stale/unused UI: `measurement_box.py` still expects removed config constants and a richer `MeasurementState` with `mode`, `f_min_hz`, `f_max_hz`, `overlap_width_hz`, and `scale`.

Impact:

- `measurement_box.py` is stale and fails type checking.
- Simulated CRISP and infrared ranges are currently hard-coded in config.

Likely next decision:

- Delete the old `MeasurementBox` unless it can become part of the new reconstruction or measurement tool window.
- Keep external `.npz` measurements as the first real measurement path, while deferring exact schema finalization until real files are available.

### 4. Plot Controls Are Now Simpler

`plot_controls_box.py` now controls line visibility only, and it sits to the left of the plots. The normalized time mode is no longer exposed in the active UI.

### 5. Recent Plot Polish Landed

The latest local commits focused on plot readability and layout stability:

- theme switching no longer disturbs the plot layout
- embedded plot DPI is fixed
- plot colors are adjusted for better colorblind friendliness

Remaining polish:

- The old `TIME_PLOT_MODE.NORMALIZED` enum still exists in config but is not reachable from the GUI.
- A later cleanup can remove the unused enum and normalized plot-model helpers.

### 6. Lint And Type Checking Are Not Clean

The project has working runtime tests, but the static tooling reports a lot of noise. Some issues are harmless typing friction around PySide6 stubs; others are real drift from refactors.

High-priority static issues:

- Stale config references in `measurement_box.py`.
- Stale development script imports.
- Generic enum type mismatch in option selector widgets.

Lower-priority/static-noise issues:

- PySide6 enum stub problems such as `Qt.Horizontal` and `QPalette.Window`.
- Tests that intentionally instantiate abstract classes inside `pytest.raises`.
- Tests that intentionally assign to slotted dataclasses to verify slot behavior.

### 7. App/Core Boundaries Are Blurred

`core/reconstruction.py` imports `PHASE_INIT_MODE` from `phase_weaver.app.state` inside a method, so the reconstruction layer still reaches into app state for one of its mode enums.

Impact:

- The core reconstruction module depends on app UI state.
- It is harder to reuse reconstruction outside the GUI.

Better direction:

- Move reconstruction configuration types into `core` or use a small protocol/value object.
- Keep app-specific widgets and config conversion in `app`.
- Let `core` accept primitive or core-owned enums.

### 8. Developer Artifacts Need Sorting

Current artifacts that should be intentionally handled:

- `.tmp-mpl/` should probably be gitignored.
- `notebooks/scratch.ipynb` has large uncommitted changes and lint noise.
- `scripts/reconstruction_dev.py` references removed APIs.
- `Makefile` is useful, but should be committed only after deciding the canonical test commands.

## Recommended Direction

### Near-Term Stabilization

1. Reshape the main window around the two central plots.
   - Keep time-domain current/profile large on the left.
   - Keep Fourier/form-factor frequency view large on the right.
   - Remove persistent toy-model and detailed reconstruction controls from the central layout.

2. Move the toy model into an optional View window.
   - Preserve the existing three-super-Gaussian model as the first profile generator.
   - Let the user open, close, and use that window to generate/update the active simulated profile.
   - Later, add a clear state transition for disabling toy-model generation when a file-based simulated profile is loaded.

3. Move reconstruction setup into an optional View window.
   - Put phase initialization and reconstruction parameters there.
   - Keep room for more detailed settings without bloating the main view.
   - Continue backing this with the shared app/session data model.

4. Move file operations into menus.
   - Add or consolidate `File > Load ...`, `File > Save ...`, and `File > Export ...`.
   - Keep `.npz` as the first measurement-bundle format.
   - Defer precise schema work for CRISP/IR bundles and simulation/Ocelot files until concrete input examples are available.

5. Clean stale code from the current refactor.
   - Delete or revive `measurement_box.py`.
   - Update or remove `scripts/reconstruction_dev.py`.
   - Add `.tmp-mpl/` to `.gitignore`.

6. Bring Ruff close to green.
   - Auto-fix unused imports where safe.
   - Exclude notebooks if they are exploratory.
   - Keep lint focused on package code and tests.

### Medium-Term Architecture

1. Separate core reconstruction from app state.
   - Put reconstruction config in `core`.
   - Make the GUI adapt its controls into core config.
   - Avoid importing app enums from core modules.

2. Clarify simulated measurement modeling.
   - Decide how generated CRISP and infrared spectrometer measurements are represented in the shared data model.
   - Support visual comparison between generated and loaded measurements first.
   - Add residuals and numerical metrics later.

3. Improve reconstruction observability beyond the first summary.
   - The app now reports iterations, stop reason, and final measurement error.
   - A later improvement could show convergence history across iterations.

4. Restore meaningful app tests.
   - Add tests for `AppLogic`, measurement slicing, plot model mode switching, and phase initialization.

### Product Direction

PhaseWeaver should move toward being a trustworthy interactive reconstruction workbench:

- The controls should map clearly to the physical/reconstruction model.
- Users should be able to compare true/input, measured, and reconstructed quantities without ambiguous units.
- Users should be able to generate data from the toy model, load measurement data, and reconstruct from the active measurement set without the main window becoming a control dump.
- Reconstruction results should expose convergence status, not just a plotted line.
- Exported data should include enough metadata to reproduce the run: profile parameters, measurement mode/ranges/scales, phase init mode, algorithm settings, and stop reason.

## Suggested Next Work Order

1. Redesign the main window around the two large plots.
2. Extract the toy model controls into an optional View window.
3. Extract reconstruction configuration into an optional View window.
4. Move load/save/export actions into the `File` menu.
5. Keep `.npz` measurement loading working, but defer schema expansion until real CRISP/IR and simulation files are available.
6. Decide whether stale `measurement_box.py` should be deleted or folded into the new window structure.
7. Verify exported `.npz` files contain all metadata needed after a run.
8. Add `.tmp-mpl/` to `.gitignore`.
9. Clean Ruff errors in source files.
10. Decide whether basedpyright should gate CI now or after PySide/test typing noise is reduced.

## Useful Commands

```bash
uv run pytest tests
```

Runs the currently passing active unit suite.

```bash
uv run pytest
```

Runs the full active suite.

```bash
uv run ruff check .
```

Currently fails; useful for identifying stale imports and undefined names.

```bash
uv run basedpyright
```

Currently fails with many errors; useful after the refactor is cleaned up.

```bash
uv run phase_weaver
```

Runs the GUI app in a normal desktop session.

## Bottom Line

The project is in a good place for the next structural step: the numerical core and active tests are alive, the GUI starts, and the first reconstruction workflow exists. The main product risk is now UI shape rather than raw capability: the main window carries too much setup machinery for the app to feel like a focused workbench.

The best next move is to make the app feel like the intended workflow: large time/frequency plots in the center, optional windows for toy-model generation and reconstruction settings, and file operations in menus. The exact real-data schemas can wait until the CRISP/IR and simulation examples are available.
