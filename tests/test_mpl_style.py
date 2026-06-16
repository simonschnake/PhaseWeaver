from matplotlib.figure import Figure
from PySide6.QtWidgets import QApplication

from phase_weaver.mpl_style import sync_mpl_to_qt


def test_sync_mpl_to_qt_preserves_existing_figure_dpi():
    app = QApplication.instance() or QApplication([])
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Demo")

    initial_dpi = fig.get_dpi()

    sync_mpl_to_qt(app, fig)

    assert fig.get_dpi() == initial_dpi
    assert ax.title.get_fontsize() > 0
