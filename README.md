# PhaseWeaver

PhaseWeaver is a small interactive workbench for exploring **current profile reconstruction**.

You shape a time-domain profile (background + spike), and PhaseWeaver immediately shows:

- the **current profile** in time
- the corresponding **spectrum magnitude** and **unwrapped phase**
- a **magnitude-only reconstruction** (iterative phase retrieval) overlaid with the original, so you can compare input vs. reconstructed current and phase

It enforces simple physical constraints (non-negativity, centering, fixed total charge) and is designed for quick “what-if” exploration and sharing with colleagues.

## Prerequisites: `uv`

PhaseWeaver uses **uv** to install and run the app as a “tool” (isolated environment, installs an executable onto your PATH).

Install uv by following the official documentation:  
https://docs.astral.sh/uv/getting-started/installation/

## Install PhaseWeaver (from a local checkout)

Clone / download this repository, then in the project root run:

```bash
uv tool install .
```

### PATH setup (if the command isn’t found)

If `phase_weaver` isn’t found after installation, run:

```bash
uv tool update-shell
```

Then open a **new terminal** (or restart your shell).

## Run

```bash
phase_weaver
```

## Updating / Uninstalling

Upgrade:

```bash
uv tool upgrade phase_weaver
```

Uninstall:

```bash
uv tool uninstall phase_weaver
```
[project.scripts]
phase_weaver = "current_predict.app:main"
```
