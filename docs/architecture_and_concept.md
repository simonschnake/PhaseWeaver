# PhaseWeaver — Architecture, Concept & Rewrite Notes

> One document capturing the whole project: what it is, the physics/data concept,
> how the code is laid out, what happens where, and the thinking about how to
> clean up the concept before a real rewrite.
>
> Status: architecture simplification is complete through Phases A–G. `core/` is pure
> with algorithm-policy enums in `core/policy.py`; measurement carriers use the
> `Measurement` / `SquaredMagnitudeMeasurement` family in `core/measurement.py`;
> reconstruction dispatch is typed and declarative in `core/pipeline.py`.
> `AppLogic` now has loading, export, and simulation seams, and the toy scenario
> model has a canonical `ScenarioState` in `model/profile_model.py`.
> The unit, lint, type, and GUI smoke checks are clean. This file is **a living design doc**:
> the target structure below is updated to match the implemented architecture.

---

## 1. Executive summary

PhaseWeaver is a PySide6/pyqtgraph desktop workbench for **current-profile
reconstruction in accelerator/beam physics**: given measurements of the bunch's
form factor (or form-factor magnitude) over frequency, recover the electron-bunch
**current profile** (charge density vs. time) — a phase-retrieval problem.

The app started as a **magnitude-only toy workbench** (a parameterized
background + spikes toy model, synthetic "ideal" measurements, Gerchberg–Saxton
iterative reconstruction) and has grown into a **CRISP-aware tool** that:

- imports real CRISP XFEL HDF5 recordings (per-shot `|F|²`),
- reconstructs with an upstream-style CRISP algorithm (Gaussian extrapolation +
  Kramers–Kronig + modulus-band iteration),
- loads Ocean/NIR spectra as explicitly **relative** shape constraints, and
- can chain a **CRISP → Gerchberg–Saxton + IR** extension.

Two front-end detector **forward simulation** modules (CRISP detector, Ocean/NIR
spectrometer) let it exercise the whole pipeline without hardware.

The central design tension the whole codebase lives around is the **honest split
between calibrated absolute measurements and relative-shape data** — the physical
meaning of "form factor" depends on *which* instrument produced it and *how* it
was calibrated. Keeping that distinction explicit is the thread that runs through
every layer.

---

## 2. The domain concept

### 2.1 Time and frequency domain objects

Everything reduces to a **profile** <-> **form factor** pair on a common **grid**.

- **`Grid`** (`core/base.py`) — a uniform time grid characterised by `N` (power
  of two, even) and `dt`. Its `df`, `f_nyq`, `t`, `f_pos`, `f_shift` axes all
  derive from that. Two factory constructors (`from_dt_tmax`, `from_df_fmax`)
  pin it by real-space or Fourier-space spans.
- **`Profile` / `CurrentProfile`** (`core/base.py`) — a time-domain function on
  a grid. `CurrentProfile` enforces non-negativity + area normalization and, when
  a `charge` is set, yields physical `current = charge * values`.
- **`FormFactor`** (`core/base.py`) — a complex frequency-domain function,
  stored as `mag` + `phase` on the grid's positive half spectrum.
- **`Transform`** (`core/base.py`) — the mapping between the two. The concrete
  `DCPhysicalRFFT` is a DC-normalized, phase-unwrapped RFFT; `BandLimitedDCPhysicalRFFT`
  adds an optional high-frequency cutoff (compact or masked).

So the whole domain talks in one vocabulary: `Grid + Profile/FormFactor +
Transform`, all in `core/base.py`.

### 2.2 The reconstruction concept

This is a **phase-retrieval** problem: the profile's magnitude spectrum is
(partially) measured, the phase in the Fourier domain is unknown, and the phase
retrieval recovers a profile consistent with both the measured magnitude and the
prior constraints (a real, non-negative, finite-extent density).

Two algorithms implement this:

- **Gerchberg–Saxton** (`core/reconstruction.py`) — alternating projections
  between time and frequency domains. Time constraints (non-negativity, normalize
  area, center first moment, cut zeros) and frequency constraints (clamp
  magnitude, enforce DC, high-frequency decay, blend measured) are applied each
  iteration. Stop criteria (max/min iterations, phase stable, measurement error)
  terminate it. This is the "generic / toy" path.
- **CRISP** (`core/crisp_reconstruction.py`) — a port of the upstream XFEL
  reconstruction server. It works on **`|F|²`** (not `|F|`): preprocesses/filters
  inputs, fits low- and high-frequency Gaussians, extrapolates to a grid,
  Savitzky–Golay-smooths (ported), derives a **Kramers–Kronig** phase, isolates
  the positive main peak, iterates with a **measured-modulus-band** replacement
  (±20 % band), scales by charge, and emits rich diagnostics.

Because CRISP wants `|F|²` and the generic app layer stores `|F|`, the two
representations must be kept distinct and converted at the boundary — a recurring
theme (see §7 cleanup).

### 2.3 The measurement concept and the calibrated-vs-relative split

Measurements enter as **`Measurement`** and, for raw CRISP squared data,
**`SquaredMagnitudeMeasurement`** (`core/measurement.py`): unordered `freq`/`mag`
(sorted on load) plus optional `mag_std` and `detection_limit`, with calibration,
kind, and constraint-participation metadata carried explicitly.

The physical reality is that there are **two different kinds of input** with
different calibration status:

| Kind | Instrument | Units / meaning | Calibration status |
|------|-----------|-----------------|--------------------|
| CRISP | XFEL form-factor service | calibrated **`\|F\|²`** | absolute, calibrated |
| Ocean/NIR | NIRQuest spectrometer | wavelength/intensity (arbitrary) | **relative shape only** |

The app distinguishes these on **`LoadedMeasurement`** (`app/logic.py`) via a
`kind` field (`"crisp"`, `"ocean_nir"`, `"infrared"`) and a `calibration` field
(`"loaded"`-style values like `simulated_calibrated`, `simulated_relative_shape`,
`simulated_ideal`, and the loaded/calibrated variants). The **relative** Ocean/NIR
data is:

- excluded from the absolute-magnitude constraint by default
  (`active_measurements`, `logic.py:1016`),
- usable only as an **opt-in** relative-shape constraint
  (`use_ir_relative_constraint`) that matches the reconstructed average over the
  IR band or multiplies by a fixed user scale.

The design decision that keeps the physics honest: **relative spectra are never
promoted to absolute form-factor semantics.**

### 2.4 The two simulation forward models

To exercise the whole pipeline without hardware, the project has two forward
models (opt-in):

- **CRISP detector simulation** (`core/crisp_simulation.py`) — legacy 240-channel
  response table, charge-scaled ADC signal, repeatable electronic noise, shot
  averaging, detection limits, and reconstruction-ready `|F|²` uncertainties.
- **Ocean/NIR spectrometer simulation** (`core/ocean_simulation.py`) — 512-pixel
  896–2515 nm grid, count-domain noise, shot averaging, deterministic seeds,
  relative normalization, uncertainty, detection limits.

These produce the same `Measurement` family shapes the loaded data use, so the
reconstruction layer cannot tell (and does not need to tell) simulated from
measured — the semantics live in the `kind`/`calibration` metadata.

---

## 3. Module map — "what lives where"

```
src/phase_weaver/
├── core/                  # pure numerical/physics layer (no Qt, no app import)
│   ├── base.py               Grid, Profile/CurrentProfile, FormFactor, Transform
│   ├── measurement.py        Measurement family (|F| and |F|² carriers)
│   ├── constraints.py        time + frequency constraint toolkit
│   ├── reconstruction.py     Gerchberg–Saxton + stop criteria + phase init
│   ├── crisp_reconstruction.py  CRISP pipeline (|F|² -> profile), diagnostics
│   ├── crisp_simulation.py   CRISP detector forward model
│   ├── ocean_simulation.py   Ocean/NIR spectrometer forward model
│   ├── utils.py              FWHM/stats, Gaussian extension, interpolation, etc.
│   ├── constants.py          (tiny) physical constants
│   └── __init__.py           re-exports the public API
│
├── model/                 # toy "scenario" generator
│   ├── profiles.py           AsymSuperGaussParams + asymmetric_super_gaussian
│   └── profile_model.py      ScenarioState + ProfileModel (ProfileModelState alias)
│
├── app/                   # application / orchestration layer
│   ├── config.py             enums + physics constants + UI units + defaults (mixed bag)
│   ├── state.py              compatibility facade + ReconstructionState,
│   │                         MeasurementState, ControlsState
│   ├── loading.py            MeasurementLoader seam for file/shot loading
│   ├── simulation.py         SimulationService seam for detector forward models
│   ├── export.py             NpzExporter seam for result serialization
│   ├── logic.py              AppLogic facade: measurement policy + orchestration
│   ├── plot_model.py         TimePlotModel / SpectrumPlotModel (data -> display units)
│   ├── plot_theme.py, utils.py
│   ├── main.py               entry point
│   └── ui/                   PySide6 widgets
│       ├── main_window.py        window + menu wiring + action orchestration (645 ln)
│       ├── plot_panel.py         two-plot pane (time + frequency)
│       ├── reconstruction_panel.py  algorithm/sim/constraint/stop controls (320 ln)
│       ├── toy_model_panel.py, control_box.py, option_selector_box.py
│       ├── gaussian_group.py, band_limit_box.py, phase_end_box.py
│       ├── plot_controls_box.py, measurement_time_dialog.py
│
├── qt_theme.py, rc_resources.py, _resources.py, __main__.py
```

Key layout facts worth knowing:

- **`core/` is pure**: policy enums live in `core/policy.py`; the former
  core→app phase-init dependency has been removed.
- **The scenario model lives in `model/profile_model.py`**. `ScenarioState` is
  canonical; `ProfileModelState` remains a compatibility alias for existing UI
  and import callers.
- **`app/logic.py` remains the orchestrator**, but its major seams are explicit:
  loading (`app/loading.py`), simulation (`app/simulation.py`), and export
  (`app/export.py`). Format readers and measurement policy remain in `logic.py`
  for now to preserve compatibility.
- **`app/config.py` mixes concerns**: UI units (`TIME_UNIT`, `T_MAX_UI`), physics
  constants (`CHARGE_C`, `PHASE_END_REF_FREQ_HZ`), instrument band edges
  (`CRISP_MIN_HZ`…), enums for every option set, and default-value dictionaries —
  all in one module, and imported by `plot_model.py`, `state.py`, `logic.py`.
- **`app/` imports `core`**, and `model` imports `core`; the dependency direction
  is one-way as intended.

---

## 4. End-to-end flow — "what happens where"

### 4.1 The reconstruction pipeline (domain view)

```
 CurrentProfile ──Transform(profile_to_form_factor)──▶ FormFactor (mag+phase)
      │  charge                                          │
      │                                                  │  phase unknown in reality
      ▼                                                  ▼
 measurements: Measurement (freq, |F|, ±std, det.limit, provenance)
      │  (from loaded HDF5/NPZ  OR  forward simulations)
      ▼
 reconstruction algorithm (GS or CRISP)
      ├─ build initial form factor (gaussian_extend / phase init mode)
      ├─ iterate: time constraints <-> frequency constraints  (GS)
      │            OR  modulus-band iteration (CRISP)
      └─ measure error vs. measured |F|
      ▼
 CurrentProfile (reconstructed) + FormFactor (recon)  + ReconstructionSummary
```

### 4.2 Application flow (GUI perspective)

The `AppLogic` class is the orchestrator the UI drives. Main entry points:

1. **`compute_initial`** (`logic.py:656`) — builds the toy input profile
   (`ProfileModel.compute_profile`), its form factor, and the active measurement
   set. This is the "reference" everything is compared against.
2. **`active_measurements` / `visible_measurements`** (`logic.py:943`/`1026`) —
   decide the working measurement set. If measurements were loaded from file, use
   those; otherwise synthesize them from the toy profile + measurement-state
   flags (ideal band sampling or detector simulation). `visible_*` additionally
   tags each measurement with `kind`/`calibration` for display and honesty.
3. **`compute_reconstruction`** (`logic.py:672`) — the main dispatch through the
   typed `ReconstructionPipeline` (`core/pipeline.py`):
   - **CRISP algorithm** consumes a `SquaredMagnitudeMeasurement` and, when an
     IR constraint is present, uses `CrispThenIrSeed` for the GS extension.
   - **Gerchberg–Saxton algorithm**: runs GS directly, optionally with relative
     IR measurements as a shape constraint.
   - Both record a `ReconstructionSummary` and stash `phase_last` for `LAST`
     phase-init chaining.
4. **`export_npz`** (`logic.py:1286`) — delegates payload assembly to
   `NpzExporter` (`app/export.py`) while retaining the compatibility facade.

The UI layer (`main_window.py`) wires menu actions to these methods and refreshes
`TimePlotModel`/`SpectrumPlotModel`, which convert to display units (fs, kA, THz)
consumed by the pyqtgraph `plot_panel.py`.

### 4.3 Measurement loading

- **CRISP loading** (`load_crisp_measurements_file`) accepts HDF5 recordings,
  selects a shot by timestamp or explicit index, validates the CRISP arrays, and
  returns only CRISP data. It preserves squared-magnitude reconstruction inputs,
  charge metadata, and the optional SA1 reference current.
- **IR loading** (`load_ir_measurements_file`) accepts supported Ocean/NIR NPZ or
  HDF5 data, validates the relative spectrum, and returns only Ocean/NIR data.
- `MeasurementLoader` enforces the instrument-specific result contracts before
  `AppLogic` updates state. IR replacement preserves loaded CRISP data, and a
  failed load leaves the previous measurement and reconstruction state intact.

---

## 5. The two reconstruction algorithms in more detail

### 5.1 Gerchberg–Saxton (`core/reconstruction.py`)

- **Phase init** (`_calculate_init_formfactor`): zero / real / minimum-phase /
  last. Magnitude init from Gaussian extension of the measured data (or from an
  input form factor).
- **Time constraints** (`core/constraints.py`): cut-zeros-after-peak, non-negativity,
  normalize-area, center-first-moment — built from named selections
  (`_build_time_constraints`).
- **Frequency constraints**: clamp-magnitude, enforce-DC, high-frequency-decay
  (exponential tail past measurements), blend-measured (+ optional
  `BlendRelativeMeasuredShape` for IR), spline-interpolate gaps — built in
  `_build_frequency_constraints`.
- **Stop**: combined "any/all" of max-iter, phase-stable, measurement-error;
  `ReconstructionHistory` records per-iteration error/phase/profile deltas.

### 5.2 CRISP (`core/crisp_reconstruction.py`)

A faithful Python port of the upstream server operating on `|F|²`:

```
preprocess_crisp_input (mask NaNs & below-detection-limit, cutoff after bad run,
                        clamp [0,1], neighbor-mask)
  -> fit_low_frequency_sigma (LF Gaussian from early valid points)
  -> high_frequency_sigma (HF Gaussian from last valid point)
  -> extrapolate_crisp_ffsq (low -> 0, high -> max_frequency)
  -> smooth_intermediate_ffsq (Savitzky–Golay 9pt/order3, edge-preserving)
  -> interpolate_crisp_ffabs / _error (positive half-grid, |F|)
  -> kramers_kronig_phase
  -> build_crisp_full_spectrum (symmetric complex spectrum)
  -> iFFT start profile
  -> up to max_iterations: isolate_positive_maximum, FFT/DC normalize,
     _replace_modulus_outside_band (±modulus_error_fraction), iFFT, recenter
  -> scale current by charge & dt
  -> CrispReconstructionResult{profile, form_factor, CrispDiagnostics, stop_reason}
```

`CrispDiagnostics` carries intermediate arrays, iteration profiles, and profile
statistics (peak current, FWHM, RMS width, skewness) that get exported with the
run.

---

## 6. Current state snapshot

- **Branch `main`**, package `PhaseWeaver 0.7.0` (`pyproject.toml`), src-layout
  `src/phase_weaver/`, deps: PySide6, pyqtgraph, numpy, scipy, h5py; dev: pytest,
  ruff, basedpyright.
- **Tests**: the unit and smoke suites complete successfully. The optional
  reference-HDF5 comparison uses local recordings in git-ignored `test_data/`.
  The recordings are intentionally **not** in version control.
- **Lint**: `ruff check .` → clean.
- **Types**: `basedpyright` is scoped to `src/` and completes cleanly.
- **Docs**: `docs/{formfactor_calculation,crisp_reconstruction_algorithm,
  ocean_insight_server,ocean_relative_formfactor_path}.md` capture the physics/
  software sources.
- **Headless GUI smoke test** (`QT_QPA_PLATFORM=offscreen ... QTimer.singleShot`)
  constructs the window and runs the event loop.

---

## 7. Thinking about the rewrite — concept cleanup & target design

This is the meat for the rewrite planning. Ordered roughly from highest-impact to
lowest. It is deliberately opinionated — treat as a proposal to react to, not a
fait accompli.

### 7.1 Remove the `core` → `app` dependency (highest priority)

This item is complete. `PHASE_INIT_MODE` and the other algorithm-policy enums now
live in `core/policy.py`; `app/config.py` re-exports them for UI compatibility.
`core` is importable without knowledge of `app`/Qt.

### 7.2 Unify the measurement representation & make calibration first-class

The former parallel carriers have been replaced by an explicit measurement family:

- `Measurement` (mag `|F|`, generic)
- `SquaredMagnitudeMeasurement` (`|F|²`, + std + det.limit)
- `LoadedMeasurement` (label + measured + crisp_input + calibration + kind)
- `ReferenceCurrentProfile`

and the code converts between them ad hoc (`_h5_shot_to_measured_formfactor`,
`_active_crisp_input` recomputes `|F|²` from `|F|` with zero uncertainty, etc.).
The implemented `Measurement` family carries, explicitly:

- `freq`, `mag`, `mag_std`, `detection_limit` (as `|F|`),
- **and** optionally the raw `|F|²` + its uncertainty (for CRISP), never derived
  silently,
- **`kind`** (`crisp`, `ocean_nir`, `infrared`), **`calibration`** (absolute/relative/
  unknown), **source file / shot / timestamp / charge**, and **whether it
  participates in absolute vs. relative constraints**.

One type, no dual representation, no silent `sqrt`/square round-trips. Make the
docs' "keep `|F|²` and `|F|` semantics explicit" requirement part of the type
itself. This is the single biggest conceptual win: it kills the whole class of
magnitude/squared bugs and makes the calibrated-vs-relative guarantee structural
instead of a comment.

### 7.3 Re-home the physics config and the `config.py` sprawl

`app/config.py` mixes UI units, physics constants, instrument bands, enums, and
default sets. For a rewrite:

- put **pure physics/algorithm constants and defaults** in `core` (near the code
  that uses them, or a `core/physics.py`),
- keep **UI/display units** (`TIME_UNIT`, `T_MAX_UI`, plot ranges) in `app`,
- define enums (constraint names, stop conditions, phase init) in `core` and let
  `app` map them to labels.

### 7.4 Fix the module/package boundaries: reduce the god-module and dead files

The first decomposition pass is complete:

- `app/loading.py` provides `MeasurementLoader` for file/shot loading.
- `app/simulation.py` provides `SimulationService` for detector forward models.
- `app/export.py` provides `NpzExporter` for result serialization.
- `core/pipeline.py` provides typed algorithm dispatch and the
  `CrispThenIrSeed` composition.
- `model/profile_model.py` now contains `ProfileModel` and canonical
  `ScenarioState`; `ProfileModelState` is retained only as a compatibility alias.

`AppLogic` still owns format-specific readers, measurement policy, and UI-facing
orchestration. A future pass may split those remaining responsibilities further,
but the current seams preserve behavior and public compatibility.
- Give `main_window.py` (645 ln) a cleaner action/controller split instead of one
  window doing orchestration + menu + refresh.

### 7.5 Make the algorithm-dispatch table explicit

`core/pipeline.py` provides a small typed **`ReconstructionPipeline`** registry
for algorithm-policy dispatch. `compute_reconstruction` now uses this dispatch
and the named `CrispThenIrSeed` composition. The resulting target is
`{algorithm, inputs, options} -> {profile, formfactor, summary}`:

- `GerchbergSaxton.run()`
- `CrispReconstruction.run()`
- `CrispThenIrSeed.run()` (the existing CRISP→GS+IR composition, promoted from
  inline code in `logic.py:633` to a named, testable step).

This makes the `CRISP + IR` composition a first-class, unit-tested algorithm
rather than an inline special case.

### 7.6 Reduce the name-string coupling in constraints/stop-criteria

`reconstruction.py` builds constraints by matching **string names** derived from
enum `.name`s (`_selected_names`) against hard-coded literals
(`if "CUT_AFTER_ZERO" in selected`). This is brittle and untracked by the type
checker. Options: make the selection a set of the actual **constraint objects**
(rely on the existing composition via `+`), or key a registry by the enum values
so adding a constraint means one registration, not another string compare.

### 7.7 Define a first-class **`ReconstructionResult`** and **export schema**

Today `ReconstructionSummary` + `history` + `crisp_diagnostics` are assembled in
`logic.py` and flattened into `.npz` by hand (a big dict builder with many
`np.array(...)` coercions). For a rewrite:
- make `ReconstructionResult` a typed object returned by every algorithm (already
  partially true for CRISP),
- define an explicit **export schema** (dictionary keyed by stable names, with
  per-kind measurement blocks) so export stops being a bespoke function that must
  be hand-extended for every new diagnostic.

### 7.8 Separate forward simulation from inverse reconstruction

The two forward models live in `core/` alongside the inverse algorithms. They are
conceptually distinct (synthesis vs. analysis) and share only the `Measured*`
carriers. Proposing a `core/simulation/` or leaving them as sibling modules is a
small call — the important part is that their **output contract** is the unified
`Measurement` from §7.2 and that they never mutate the reconstruction path.

### 7.9 Keep static tooling clean

- `build/` is now excluded and basedpyright is restricted to `src/` in
  `pyproject.toml`, removing duplicate-package and test/script noise.
- The source tree passes Ruff and basedpyright. Keep both checks in the normal
  verification workflow as the architecture evolves.

### 7.10 Keep what is good

Not everything needs to change. The things worth preserving in a rewrite:
- The **`Grid`/`Profile`/`FormFactor`/`Transform`** core vocabulary — it is clean
  and small.
- The **constraint-by-composition** design (`+` operators, `Combined*`) — extend
  it, don't replace it.
- The **honesty discipline** (calibrated vs relative, never promoting relative
  data to absolute) — this must survive as a structural guarantee.
- The **CRISP pipeline as a faithful, documented port** with diagnostics — keep
  the algorithm module intact even as its input/schema is unified.

---

## 8. Open questions / decisions for the rewrite

1. **Export schema**: is `.npz` the right long-term format, or should a richer
   (HDF5 or JSON-metadata + arrays) schema replace it as part of the rewritten
   exporter?
2. **Further `logic.py` decomposition**: should the remaining NPZ/HDF5 readers
   move into dedicated format modules, or should the current compatibility facade
   remain the stable application boundary?

---

## 9. Quick references

- Verify: `make test` · `make smoke` · `uv run ruff check src tests` · `uv run basedpyright`
- Headless GUI smoke: `QT_QPA_PLATFORM=offscreen uv run python -c "<construct QApplication + MainWindow, singleShot(200, quit)>"`
- Specific commands in `PROJECT_STATE.md`.
