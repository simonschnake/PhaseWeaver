# -----------------------------
# Constants / UI units
# -----------------------------
from enum import Enum

TIME_UNIT = "fs"
S_TO_T = 1e15
T_TO_S = 1e-15

# Profile parameters
DT = 1e-16
T_MAX = 1e-12
T_MAX_UI: float = 1e-13
CHARGE_C = 250e-12  # 250 pC


class PHASE_INIT_MODE(Enum):
    ZERO = "zero"
    REAL = "real"
    MINPHASE = "minphase"
    LAST = "last"


PHASE_INIT_DEFAULT = PHASE_INIT_MODE.ZERO

SPIKE_SPEC = {
    "center_fs": (-20.0, 20.0, 0.1, -10.0),
    "width_fs": (0.1, 10.0, 0.005, 0.721),
    "skew": (-1.0, 1.0, 0.1, 0.0),
    "order": (0.01, 4.0, 0.01, 0.5),
    "amplitude": (0.0, 1e5, 100.0, 13000.0),
}

BACKGROUND_SPEC = {
    "width_fs": (1.0, 100.0, 0.5, 21.91914604893585),
    "skew": (-1.0, 1.0, 0.1, 0.0),
    "order": (0.01, 10.0, 0.10, 1.0),
    "amplitude": (0.0, 1e5, 100.0, 5819.0),
}

PEAK2_ENABLED = False

C_NM_THz = 299792.458
PHASE_END_REF_FREQ_HZ = 300e12


LINEAR_PHASE_END_ENABLED = False
LINEAR_PHASE_END_START_FREQ_HZ = 0.5 * PHASE_END_REF_FREQ_HZ
LINEAR_PHASE_END_PHASE_AT_300_THZ = 0.0
BAND_LIMIT_ENABLED = False
BAND_LIMIT_F_CUT_HZ = 250e12

DEFAULT_SPECTRUM_FMIN_HZ = 0.0
DEFAULT_SPECTRUM_FMAX_HZ = 333e12

# Measurement parameters
CRISP_MIN_HZ = 0.0e12  # 0.684283910
CRISP_MAX_HZ = 60.0e12
IR_MIN_HZ = 120.0e12
IR_MAX_HZ = 333.0e12


class MEASUREMENT_MODE(Enum):
    CRISP = "CRISP"
    INFRARED = "INFRARED"


INITIAL_GAUSSIAN_SIGMA_S = 10e-15


class TIME_PLOT_MODE(Enum):
    CURRENT = "Current (kA)"
    NORMALIZED = "Normalized distribution"


class PLOT_LINE_MODE(Enum):
    CURRENT_INPUT = "Input current"
    CURRENT_RECON = "Recon current"
    MAG_INPUT = "Input |F|"
    MAG_RECON = "Recon |F|"
    PHASE_INPUT = "Input phase"
    PHASE_RECON = "Recon phase"


TIME_PLOT_MODE_DEFAULT = TIME_PLOT_MODE.CURRENT

PLOT_LINE_MODE_DEFAULT = {
    PLOT_LINE_MODE.CURRENT_INPUT,
    PLOT_LINE_MODE.CURRENT_RECON,
    PLOT_LINE_MODE.MAG_INPUT,
    PLOT_LINE_MODE.MAG_RECON,
    PLOT_LINE_MODE.PHASE_INPUT,
    PLOT_LINE_MODE.PHASE_RECON,
}
