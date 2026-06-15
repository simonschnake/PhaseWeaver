# PhaseWeaver Project State

Last updated: 2026-06-15

## Executive Summary

PhaseWeaver is currently a small Python/PySide6 scientific workbench for exploring current-profile reconstruction from form-factor magnitude data. The app can generate a configurable time-domain current profile, compute its spectrum magnitude and phase, simulate partial measurements, and run a Gerchberg-Saxton-style magnitude-only reconstruction.

The project is functional enough that the full pytest suite passes and the Qt main window can be constructed in an offscreen smoke test. The last-phase reconstruction bug has been fixed, and the first experiment-use workflow is now in place: load real `.npz` measurement bundles, compare them against the super-Gaussian model `|F|`, run GS deliberately from a button, inspect a compact reconstruction summary, and export through a save dialog.

The next work should focus on validating this workflow with real experiment files, then tightening the remaining stale UI and static-tool drift.

## Operational Readiness

If the goal is to use PhaseWeaver for experiment operation in the next week, the important path is now:

1. Load real measurement bundles.
   - Supported input is one `.npz` file.
   - Single-measurement schema: `freq_hz`, `mag`, optional `label`.
   - Multi-measurement schema: `freq_hz_0`, `mag_0`, `label_0`, `freq_hz_1`, `mag_1`, `label_1`, etc.
   - Frequencies are in Hz; magnitudes are dimensionless `|F|`.

2. Run reconstruction deliberately.
   - Parameter changes redraw the input/model only.
   - GS runs only when the operator presses `Run Reconstruction`.
   - Loaded measurements override simulated CRISP/IR measurements; simulated measurements remain as a demo fallback.

3. Compare model, measurements, and reconstruction.
   - The super-Gaussian model `|F|` stays visible.
   - Loaded measurements render as spectrum markers.
   - Reconstructed `|F|` and phase render after a successful reconstruction run.

4. Export reproducibly.
   - Export uses a save dialog instead of always overwriting `export.npz`.
   - The `.npz` export includes plot arrays, loaded measurement arrays/labels, profile parameters, phase init mode, and reconstruction summary fields.

5. Use the bottom reconstruction overview.
   - The GUI shows measurement source, measurement count, reconstruction state, iterations, stop reason, and final measurement error.

6. Keep the UI responsive enough.
   - The current synchronous reconstruction is acceptable for now if it stays fast.
   - If experiment-sized data makes the window stall, move reconstruction off the UI thread next.

## Current Repository Snapshot

- Branch: `main`
- Tracking: `origin/main`
- Latest commit: `cce0b3c fix last phase init and refresh state`
- Worktree state: dirty
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
4. The operator may load one `.npz` measurement bundle with one or more measured form-factor magnitudes.
5. The operator presses `Run Reconstruction`.
6. `AppLogic.active_measurements()` uses loaded measurements when available, otherwise simulated CRISP/IR measurements.
7. `GerchbergSaxton` builds an initial form factor by Gaussian-extending measured magnitude data.
8. Iteration alternates between time-domain constraints and frequency-domain constraints.
9. The reconstructed profile, reconstructed spectrum, loaded measurement markers, and reconstruction summary are rendered in the GUI.

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
  - `plot_controls_box.py` now keeps only line-visibility controls; normalized time plotting is not exposed.

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

### 2. Measurement UI Still Has Some Drift

The active operator workflow now supports loading real `.npz` measurement bundles. There is still older measurement UI code that is not part of the active flow.

- Current active UI: `ControlsPanel` has load, run, export controls plus simulated CRISP/IR fallback selectors.
- Stale/unused UI: `measurement_box.py` still expects removed config constants and a richer `MeasurementState` with `mode`, `f_min_hz`, `f_max_hz`, `overlap_width_hz`, and `scale`.

Impact:

- `measurement_box.py` is stale and fails type checking.
- Simulated CRISP and infrared ranges are currently hard-coded in config.

Likely next decision:

- Delete the old `MeasurementBox` unless the richer range/scale UI is needed for the experiment.
- Keep external `.npz` measurements as the primary experiment path.

### 3. Plot Controls Are Now Simpler

`plot_controls_box.py` now controls line visibility only, and it sits to the left of the plots. The normalized time mode is no longer exposed in the active UI.

Remaining polish:

- The old `TIME_PLOT_MODE.NORMALIZED` enum still exists in config but is not reachable from the GUI.
- A later cleanup can remove the unused enum and normalized plot-model helpers.

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

2. Validate the experiment measurement workflow.
   - Try a real `.npz` measurement bundle from operations.
   - Confirm frequency units, magnitude scaling, and multi-measurement labels match actual files.
   - Adjust the loader only if real files differ from the agreed schema.

3. Clean stale code from the current refactor.
   - Delete or revive `measurement_box.py`.
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

2. Clarify simulated measurement modeling.
   - Decide whether the fallback CRISP/IR simulation needs adjustable windows/scales.
   - Keep real `.npz` measurements as the primary operational input.

3. Improve reconstruction observability beyond the first summary.
   - The app now reports iterations, stop reason, and final measurement error.
   - A later improvement could show convergence history across iterations.

4. Restore meaningful app tests.
   - Add tests for `AppLogic`, measurement slicing, plot model mode switching, and phase initialization.

### Product Direction

PhaseWeaver should move toward being a trustworthy interactive reconstruction workbench:

- The controls should map clearly to the physical/reconstruction model.
- Users should be able to compare true/input, measured, and reconstructed quantities without ambiguous units.
- Reconstruction results should expose convergence status, not just a plotted line.
- Exported data should include enough metadata to reproduce the run: profile parameters, measurement mode/ranges/scales, phase init mode, algorithm settings, and stop reason.

## Suggested Next Work Order

1. Test with real experiment `.npz` measurement bundles.
2. Verify exported `.npz` files contain all metadata needed after a run.
3. Decide whether stale `measurement_box.py` should be deleted.
4. Add `.tmp-mpl/` to `.gitignore`.
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
