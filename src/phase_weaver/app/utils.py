from PySide6.QtGui import QIcon

def load_app_icon() -> QIcon:
    try:
        return QIcon(":/icon.png")
    except Exception:
        return QIcon()


def thz_to_nm(f_thz):
    import numpy as np

    C_NM_THz = 299792.458
    F_EPS_THz = 1e-6
    f = np.asarray(f_thz, dtype=float)
    f = np.maximum(f, F_EPS_THz)
    return C_NM_THz / f


def nm_to_thz(l_nm):
    import numpy as np

    C_NM_THz = 299792.458
    L_EPS_NM = 1e-6
    wavelength = np.asarray(l_nm, dtype=float)
    wavelength = np.maximum(wavelength, L_EPS_NM)
    return C_NM_THz / wavelength