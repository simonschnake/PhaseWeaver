from matplotlib.figure import Figure
from PySide6.QtWidgets import QApplication

from phase_weaver.mpl_style import COLORBLIND_FRIENDLY_CYCLE, sync_mpl_to_qt


def test_sync_mpl_to_qt_preserves_existing_figure_dpi():
    app = QApplication.instance() or QApplication([])
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Demo")

    initial_dpi = fig.get_dpi()

    sync_mpl_to_qt(app, fig)

    assert fig.get_dpi() == initial_dpi
    assert ax.title.get_fontsize() > 0


def test_colorblind_friendly_cycle_starts_with_distinct_colors():
    assert COLORBLIND_FRIENDLY_CYCLE[:4] == (
        "#4E79A7",
        "#F28E2B",
        "#B07AA1",
        "#76B7B2",
    )
