"""Algorithm / policy enums for the reconstruction core.

These are physics- and algorithm-policy vocabulary (which phase-iniitialization
to use, which constraints/stop conditions are selectable, which algorithm runs).
They live in ``core`` so that ``core`` never has to reach into ``app`` for
policy decisions. ``app/config.py`` re-exports them for the UI layer.
"""
from enum import Enum


class PHASE_INIT_MODE(Enum):
    ZERO = "zero"
    REAL = "real"
    MINPHASE = "minphase"
    LAST = "last"


PHASE_INIT_DEFAULT = PHASE_INIT_MODE.ZERO


class RECONSTRUCTION_ALGORITHM(Enum):
    GERCHBERG_SAXTON = "Gerchberg-Saxton"
    CRISP = "CRISP"


RECONSTRUCTION_ALGORITHM_DEFAULT = RECONSTRUCTION_ALGORITHM.GERCHBERG_SAXTON


class RECON_TIME_CONSTRAINT(Enum):
    CUT_AFTER_ZERO = "Cut zeros"
    NON_NEGATIVE = "Non-negative"
    NORMALIZE_AREA = "Normalize area"
    CENTER = "Center"


class RECON_FREQUENCY_CONSTRAINT(Enum):
    CLAMP_MAGNITUDE = "Clamp |F|"
    ENFORCE_DC = "DC = 1"
    HIGH_FREQ_DECAY = "HF decay"
    BLEND_MEASURED = "Blend measured"


class RECON_STOP_CONDITION(Enum):
    MAX_ITER = "Max iter"
    MIN_ITER = "Min iter"
    PHASE_STABLE = "Phase stable"
    MEASUREMENT_ERROR = "Measurement error"


RECON_TIME_CONSTRAINT_DEFAULT = {
    RECON_TIME_CONSTRAINT.CUT_AFTER_ZERO,
    RECON_TIME_CONSTRAINT.NON_NEGATIVE,
    RECON_TIME_CONSTRAINT.NORMALIZE_AREA,
    RECON_TIME_CONSTRAINT.CENTER,
}

RECON_FREQUENCY_CONSTRAINT_DEFAULT = {
    RECON_FREQUENCY_CONSTRAINT.CLAMP_MAGNITUDE,
    RECON_FREQUENCY_CONSTRAINT.ENFORCE_DC,
    RECON_FREQUENCY_CONSTRAINT.HIGH_FREQ_DECAY,
    RECON_FREQUENCY_CONSTRAINT.BLEND_MEASURED,
}

RECON_STOP_CONDITION_DEFAULT = {
    RECON_STOP_CONDITION.MAX_ITER,
    RECON_STOP_CONDITION.MIN_ITER,
    RECON_STOP_CONDITION.PHASE_STABLE,
    RECON_STOP_CONDITION.MEASUREMENT_ERROR,
}

__all__ = [
    "PHASE_INIT_DEFAULT",
    "PHASE_INIT_MODE",
    "RECONSTRUCTION_ALGORITHM",
    "RECONSTRUCTION_ALGORITHM_DEFAULT",
    "RECON_FREQUENCY_CONSTRAINT",
    "RECON_FREQUENCY_CONSTRAINT_DEFAULT",
    "RECON_STOP_CONDITION",
    "RECON_STOP_CONDITION_DEFAULT",
    "RECON_TIME_CONSTRAINT",
    "RECON_TIME_CONSTRAINT_DEFAULT",
]
