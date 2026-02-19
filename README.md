# PhaseWeaver
PhaseWeaver is a small interactive tool for investigating the reconstruction of current profiles.

You shape a time-domain profile (background + spike), and PhaseWeaver immediately shows:

the current profile in time

the corresponding spectrum magnitude and unwrapped phase

a magnitude-only reconstruction (iterative phase retrieval) overlaid with the original, so you can compare input vs. reconstructed current and phase

It enforces simple physical constraints (non-negativity, centering, fixed total charge) and is designed for quick “what-if” exploration and sharing with colleagues—more like a workbench than a production app.

## Install prerequisites: `uv`

We use **uv** to install and run the app as a “tool” (isolated environment, installs an executable onto your PATH). 

### macOS

**Option A (Homebrew):** Homebrew ([Astral Docs][1])

```bash
brew install uv
```

**Option B (Standalone installer):** ([Astral Docs][1])

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Linux (Ubuntu / Debian)

There isn’t an official `apt install uv` method in the uv docs; the recommended approach is the standalone installer. ([Astral Docs][1])

```bash
# if you don't have curl yet:
sudo apt update
sudo apt install -y curl

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(Alternative if you prefer Snap: see Snap “astral-uv”.) ([Snapcraft][2])

### Windows

**Option A (WinGet):** ([Astral Docs][1])

```powershell
winget install --id=astral-sh.uv -e
```

**Option B (Standalone PowerShell installer):** ([Astral Docs][1])

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Install Phase Weaver (from a local checkout)

1. Clone / download the repo, then in the project root:

```bash
uv tool install .
```

Local-directory installs via `uv tool install .` are a common workflow when sharing a repo directly. ([Asif Rahman][3])

2. Ensure the uv tool bin directory is on your `PATH`.

If `phase_weaver` isn’t found after installation, run:

```bash
uv tool phase_weaver
```

…and open a **new terminal** after that. ([Astral Docs][4])

## Run

Once installed:

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

([Astral Docs][5])

[1]: https://docs.astral.sh/uv/getting-started/installation/ "Installation | uv"
[2]: https://snapcraft.io/astral-uv?utm_source=chatgpt.com "Install uv on Linux | Snap Store"
[3]: https://asifr.com/uv-tool-local-install/?utm_source=chatgpt.com "Installing Local Python Packages with uv tool - Asif Rahman"
[4]: https://docs.astral.sh/uv/guides/tools/ "Using tools | uv"
[5]: https://docs.astral.sh/uv/reference/cli/ "Commands | uv"
