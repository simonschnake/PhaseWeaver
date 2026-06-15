# PhaseWeaver Project State

Last updated: 2026-06-15

## Executive Summary

PhaseWeaver is currently a small Python/PySide6 scientific workbench for exploring current-profile reconstruction from form-factor magnitude data. The app can generate a configurable time-domain current profile, compute its spectrum magnitude and phase, simulate partial measurements, and run a Gerchberg-Saxton-style magnitude-only reconstruction.

The project is functional enough that the full pytest suite passes and the Qt main window can be constructed in an offscreen smoke test. The last-phase reconstruction bug has been fixed. What remains is mostly refactor fallout: stale scripts, an unfinished measurement-control path, and lint/type errors in the broader workspace.

The next work should focus on stabilizing the refactor before adding more physics or UI features.

## Current Repository Snapshot

- Branch: `main`
- Tracking: `origin/main`
- Latest commit: `1e5ff15 small rewrite of algo`
- Worktree state: dirty
- Package version: `0.3.0`
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

- Modified:
  - `notebooks/scratch.ipynb`
  - `src/phase_weaver/app/config.py`
  - `src/phase_weaver/app/plot_model.py`
  - `src/phase_weaver/app/ui/plot_panel.py`
  - `src/phase_weaver/core/reconstruction.py`
  - `tests/core/test_base.py`
  - `tests/core/test_constraints.py`
  - `tests/core/test_reconstruction.py`
  - `tests/core/test_utils.py`
  - `tests/model/test_profile_model.py`
- Untracked:
  - `Makefile`
  - `scripts/reconstruction_dev.py`
  - `src/phase_weaver/app/ui/plot_controls_box.py`
  - `.tmp-mpl/`

## What The App Does Today

The app presents an interactive current-profile model:

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
4. `AppLogic.compute_measured_formfactor()` optionally slices the spectrum into CRISP and infrared bands.
5. `GerchbergSaxton` builds an initial form factor by Gaussian-extending measured magnitude data.
6. Iteration alternates between time-domain constraints and frequency-domain constraints.
7. The reconstructed profile and spectrum are rendered in the plot panel.

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
  - Bridges UI state to profile generation, measurement simulation, reconstruction, and NPZ export.

- `src/phase_weaver/app/ui/`
  - Contains Qt widgets for controls and plots.
  - The current controls panel uses generic option selector widgets for phase init and measurement source selection.
  - `plot_controls_box.py` adds display controls for time-plot mode and line visibility.

## Verification Results

Commands run on 2026-06-15:

```bash
uv run pytest
```

Result: pass.

- 152 tests collected.
- 152 passed.

```bash
uv run pytest tests
```

Result: pass.

- 152 tests collected.
- 152 passed.

```bash
uv run ruff check .
```

Result: fail.

Main causes:

- Stale notebook imports.
- Unused imports in `scripts/reconstruction_dev.py`, `controls_panel.py`, `core/reconstruction.py`, `core/utils.py`, `qt_theme.py`, and tests.
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

### 1. The Test Layout Is Confusing

`pytest` now runs cleanly, which is a good sign. The test layout is much less confusing now that `moved_tests/` has been removed, but the project still needs a single canonical test command in the repo metadata so contributors do not have to guess.

### 2. Measurement UI Is Half-Removed

There are two measurement-control stories in the code:

- Current active UI: `ControlsPanel` uses `MultiOptionSelectorBox` with `MEASUREMENT_MODE.CRISP` and `MEASUREMENT_MODE.INFRARED`.
- Stale/unused UI: `measurement_box.py` still expects removed config constants and a richer `MeasurementState` with `mode`, `f_min_hz`, `f_max_hz`, `overlap_width_hz`, and `scale`.

Impact:

- `measurement_box.py` is stale and fails type checking.
- Measurement scale fields exist in `MeasurementState`, but the current UI never exposes or updates them.
- CRISP and infrared ranges are currently hard-coded in config.

Likely next decision:

- Either delete the old `MeasurementBox` and commit to the simple CRISP/IR toggles.
- Or restore a richer measurement UI and make `MeasurementState` match it.

### 3. Plot Controls Are New But Need Polish

`plot_controls_box.py` and the plot-panel line visibility controls are newly introduced. The intent is good: users can switch time plot mode and hide/show plotted lines.

Known issue:

- In `_render_time()`, the normalized branch currently still uses `current_input_ui` and `current_recon_ui`, not `normalized_input_ui` and `normalized_recon_ui`.

Impact:

- The UI label can say "Normalized distribution (1/fs)" while plotting current in kA.

Likely fix:

- Use normalized arrays when `TIME_PLOT_MODE.NORMALIZED` is selected.
- Revisit the y-axis label; `Profile.values` are normalized density over seconds, not obviously `1/fs` unless converted.
- Add plot-model or UI tests for both plot modes.

### 4. Lint And Type Checking Are Not Clean

The project has working runtime tests, but the static tooling reports a lot of noise. Some issues are harmless typing friction around PySide6 stubs; others are real drift from refactors.

High-priority static issues:

- Stale config references in `measurement_box.py`.
- Stale development script imports.
- Generic enum type mismatch in option selector widgets.

Lower-priority/static-noise issues:

- PySide6 enum stub problems such as `Qt.Horizontal` and `QPalette.Window`.
- Tests that intentionally instantiate abstract classes inside `pytest.raises`.
- Tests that intentionally assign to slotted dataclasses to verify slot behavior.

### 5. App/Core Boundaries Are Blurred

`core/reconstruction.py` imports `PHASE_INIT_MODE` from `phase_weaver.app.state` inside a method, so the reconstruction layer still reaches into app state for one of its mode enums.

Impact:

- The core reconstruction module depends on app UI state.
- It is harder to reuse reconstruction outside the GUI.

Better direction:

- Move reconstruction configuration types into `core` or use a small protocol/value object.
- Keep app-specific widgets and config conversion in `app`.
- Let `core` accept primitive or core-owned enums.

### 6. Developer Artifacts Need Sorting

Current artifacts that should be intentionally handled:

- `.tmp-mpl/` should probably be gitignored.
- `notebooks/scratch.ipynb` has large uncommitted changes and lint noise.
- `scripts/reconstruction_dev.py` references removed APIs.
- `Makefile` is useful, but should be committed only after deciding the canonical test commands.

## Recommended Direction

### Near-Term Stabilization

1. Make default project verification honest.
   - Keep `make test` aligned with the canonical command.
   - Add any pytest configuration needed to keep the suite focused on the active tree.

2. Fix concrete runtime bugs.
   - Repair LAST phase initialization.
   - Fix normalized plot mode to use normalized data.
   - Add focused tests for both.

3. Clean stale code from the current refactor.
   - Either remove or revive `measurement_box.py`.
   - Update or remove `scripts/reconstruction_dev.py`.
   - Add `.tmp-mpl/` to `.gitignore`.

4. Bring Ruff close to green.
   - Auto-fix unused imports where safe.
   - Exclude notebooks if they are exploratory.
   - Keep lint focused on package code and tests.

### Medium-Term Architecture

1. Separate core reconstruction from app state.
   - Put reconstruction config in `core`.
   - Make the GUI adapt its controls into core config.
   - Avoid importing app enums from core modules.

2. Clarify measurement modeling.
   - Define whether the product supports fixed CRISP/IR windows only, arbitrary windows, overlap blending, scaling, or all of these.
   - Align `MeasurementState`, UI controls, tests, and reconstruction constraints around one model.

3. Improve reconstruction observability.
   - Return iteration count and stopping reason.
   - Track measurement error across iterations.
   - Surface these diagnostics in the UI or export file.

4. Restore meaningful app tests.
   - Add tests for `AppLogic`, measurement slicing, plot model mode switching, and phase initialization.

### Product Direction

PhaseWeaver should move toward being a trustworthy interactive reconstruction workbench:

- The controls should map clearly to the physical/reconstruction model.
- Users should be able to compare true/input, measured, and reconstructed quantities without ambiguous units.
- Reconstruction results should expose convergence status, not just a plotted line.
- Exported data should include enough metadata to reproduce the run: profile parameters, measurement mode/ranges/scales, phase init mode, algorithm settings, and stop reason.

## Suggested Next Work Order

1. Stabilize tests and collection.
2. Fix LAST phase initialization.
3. Fix normalized plot mode.
4. Resolve measurement UI drift.
5. Clean Ruff errors in source files.
6. Decide whether basedpyright should gate CI now or after PySide/test typing noise is reduced.
7. Commit the refactor in a small, reviewable state.

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

The project is not lost; it is in the middle of a useful but incomplete simplification. The numerical core and active tests are alive, and the GUI can start. The main risk is accumulated refactor drift: stale tests, stale UI modules, and static-tool failures make it unclear which APIs are current.

The best next move is to make the project boring again: one test command, one measurement model, one reconstruction config path, and no known runtime bugs in visible UI modes. After that, the physics and reconstruction quality work will have a much firmer floor.
