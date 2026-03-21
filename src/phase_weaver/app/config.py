# -----------------------------
# Constants / UI units
# -----------------------------
from typing import Literal

TIME_UNIT = "fs"
S_TO_T = 1e15
T_TO_S = 1e-15

# Profile parameters
DT = 1e-16
T_MAX = 1e-13
CHARGE_C = 250e-12  # 250 pC

MAG_INIT_MODES = ["exp", "real", "measext", "CRISP"]
MAG_INIT_DEFAULT = "exp"

PHASE_INIT_MODES = ["zero", "real", "minphase", "last", "CRISP"]
PHASE_INIT_DEFAULT = "zero"

SPIKE_SPEC = {
    "center_fs": (-20.0, 20.0, 1.0, -10.0),
    "width_fs": (0.1, 10.0, 0.005, 0.721),
    "skew": (-1.0, 1.0, 0.1, 0.0),
    "order": (0.01, 4.0, 0.01, 0.5),
    "amplitude": (0.0, 1e5, 100.0, 13000.0),
}

BACKGROUND_SPEC = {
    "width_fs": (1.0, 100.0, 0.5, 21.91914604893585),
    "skew": (-1.0, 1.0, 0.1, 0.0),
    "order": (1e-8, 10.0, 0.10, 1.0),
    "amplitude": (0.0, 1e5, 100.0, 5819.0),
}

PEAK2_ENABLED = False

C_NM_THz = 299792.458
F_EPS_THz = 1e-6
L_EPS_NM = 1e-6
PHASE_END_REF_FREQ_HZ = 300e12


LINEAR_PHASE_END_ENABLED = False
LINEAR_PHASE_END_START_FREQ_HZ = 0.5 * PHASE_END_REF_FREQ_HZ
LINEAR_PHASE_END_PHASE_AT_300_THZ = 0.0
BAND_LIMIT_ENABLED = False
BAND_LIMIT_F_CUT_HZ = 250e12

DEFAULT_SPECTRUM_FMIN_HZ = 0.0
DEFAULT_SPECTRUM_FMAX_HZ = 333e12

# Measurement parameters

MEASUREMENT_MODES = ["full", "below_cut", "between"]
MEASUREMENT_ENABLED = False
MEASUREMENT_MODE_DEFAULT = "full"
MEASUREMENT_RANGE_MIN_THZ = 0.0
MEASUREMENT_RANGE_MAX_THZ = 400.0
MEASUREMENT_RANGE_STEP_THZ = 0.1
MEASUREMENT_F_MIN_DEFAULT_THZ = 50.0
MEASUREMENT_F_MAX_DEFAULT_THZ = 250.0
MEASUREMENT_OVERLAP_WIDTH_DEFAULT_THZ = 30.0


INITIAL_GAUSSIAN_SIGMA_S = 10e-15
