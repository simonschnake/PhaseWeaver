"""Guard tests enforcing the core->app layering.

`core/` must be importable with zero knowledge of `app` or Qt. These tests
fail if any core module pulls in `phase_weaver.app` (directly or transitively).
"""
import pkgutil
import subprocess
import sys

import phase_weaver.core as core_pkg

CORE_MODULES = [
    module.name
    for module in pkgutil.walk_packages(
        core_pkg.__path__, prefix=f"{core_pkg.__name__}."
    )
]

FORBIDDEN = (
    "phase_weaver.app",
    "phase_weaver.app.",
    "from phase_weaver.app",
    "from phase_weaver.qt_theme",
    "phase_weaver.qt_theme",
    "PySide6",
    "pyqtgraph",
)

# Run the dynamic import check in a FRESH interpreter: pytest shares one process,
# so other tests will have already imported the app layer into sys.modules.
_DYNAMIC_SCRIPT = """
import importlib, pkgutil, sys
import phase_weaver.core as pkg
for m in (
    m.name
    for m in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + ".")
):
    importlib.import_module(m)
loaded = sorted(n for n in sys.modules if n.startswith("phase_weaver.app"))
if loaded:
    print("FORBIDDEN:", loaded)
    raise SystemExit(1)
print("OK")
"""


def test_core_does_not_import_app():
    """Importing every core module in a fresh interpreter must not pull in app."""
    result = subprocess.run(
        [sys.executable, "-c", _DYNAMIC_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "core imported the app layer transitively:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


def test_core_source_never_references_app_or_qt():
    """No core source file may contain an app/Qt import or reference."""
    hits = []
    for module in CORE_MODULES:
        path = module.replace(".", "/") + ".py"
        try:
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
        except OSError:
            continue
        for line_no, line in enumerate(source.splitlines(), 1):
            if any(forbidden in line for forbidden in FORBIDDEN):
                hits.append(f"{module} (line {line_no}): {line.strip()}")
    assert not hits, "core references the app/Qt layer:\n" + "\n".join(hits)


def test_policy_enums_are_re_exported_from_app_config():
    """app/config.py re-exports the core policy enums unchanged."""
    from phase_weaver.app import config
    from phase_weaver.core import policy

    assert config.PHASE_INIT_MODE is policy.PHASE_INIT_MODE
    assert config.RECON_TIME_CONSTRAINT is policy.RECON_TIME_CONSTRAINT
    assert config.RECON_FREQUENCY_CONSTRAINT is policy.RECON_FREQUENCY_CONSTRAINT
    assert config.RECON_STOP_CONDITION is policy.RECON_STOP_CONDITION
    assert config.RECONSTRUCTION_ALGORITHM is policy.RECONSTRUCTION_ALGORITHM
