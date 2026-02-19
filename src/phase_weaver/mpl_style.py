import matplotlib.style as mplstyle
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication
import matplotlib as mpl

from importlib import resources

def apply_mpl_style() -> None:
    """
    Apply the bundled matplotlib style sheet.
    """
    style_ref = resources.files("current_predict").joinpath("stylelib/vscode_dark.mplstyle")
    with resources.as_file(style_ref) as style_path:
        mplstyle.use(str(style_path))

def sync_mpl_font_to_qt(app: QApplication, scale: float = 1.0) -> None:
    f = app.font()
    family = f.family()
    pt = f.pointSizeF()
    if pt <= 0:
        pt = float(f.pointSize()) if f.pointSize() > 0 else 10.0

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [family, "DejaVu Sans", "Arial", "Helvetica"]
    mpl.rcParams["font.size"] = pt * scale
    mpl.rcParams["axes.titlesize"] = pt * 1.1 * scale
    mpl.rcParams["axes.labelsize"] = pt * 1.0 * scale
    mpl.rcParams["xtick.labelsize"] = pt * 0.9 * scale
    mpl.rcParams["ytick.labelsize"] = pt * 0.9 * scale
    mpl.rcParams["legend.fontsize"] = pt * 0.9 * scale



def _qt_font_point_size(app) -> float:
    f = app.font()
    pt = float(f.pointSizeF())
    if pt > 0:
        return pt

    # Qt sometimes uses pixel-size fonts (pointSizeF == -1)
    px = f.pixelSize()
    if px > 0:
        screen = QGuiApplication.primaryScreen()
        dpi = float(screen.logicalDotsPerInch()) if screen else 96.0
        return px * 72.0 / dpi

    return 10.0


def sync_mpl_to_qt(app, fig=None, scale: float = 0.95) -> None:
    """
    - Sets rcParams (for future text)
    - If fig is passed, also force-applies to existing axes/text (so it always works).
    - Also overrides figure.dpi for GUI (critical if your mplstyle sets it high).
    """
    family = app.font().family()
    pt = _qt_font_point_size(app) * scale

    # --- Critical for embedded GUIs: don't use notebook-like DPI ---
    # If your mplstyle sets figure.dpi=200, it will look huge in Qt.
    mpl.rcParams["figure.dpi"] = 100
    mpl.rcParams["savefig.dpi"] = 200  # keep exports crisp if you want

    # --- Font family/size defaults ---
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [family, "DejaVu Sans", "Arial", "Helvetica"]
    mpl.rcParams["font.size"] = pt
    mpl.rcParams["axes.titlesize"] = pt * 1.1
    mpl.rcParams["axes.labelsize"] = pt * 1.0
    mpl.rcParams["xtick.labelsize"] = pt * 0.9
    mpl.rcParams["ytick.labelsize"] = pt * 0.9
    mpl.rcParams["legend.fontsize"] = pt * 0.9

    if fig is None:
        return

    # --- Force-apply to existing figure elements (this is what makes it "always works") ---
    fig.set_dpi(mpl.rcParams["figure.dpi"])

    for ax in fig.axes:
        ax.title.set_fontsize(mpl.rcParams["axes.titlesize"])
        ax.xaxis.label.set_fontsize(mpl.rcParams["axes.labelsize"])
        ax.yaxis.label.set_fontsize(mpl.rcParams["axes.labelsize"])

        for lbl in ax.get_xticklabels():
            lbl.set_fontsize(mpl.rcParams["xtick.labelsize"])
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(mpl.rcParams["ytick.labelsize"])

        leg = ax.get_legend()
        if leg:
            for txt in leg.get_texts():
                txt.set_fontsize(mpl.rcParams["legend.fontsize"])