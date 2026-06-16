from dataclasses import dataclass

from phase_weaver.qt_theme import APP_THEME


COLORBLIND_FRIENDLY_CYCLE = (
    "#4E79A7",
    "#F28E2B",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#9C755F",
    "#BAB0AC",
)


@dataclass(frozen=True)
class PlotTheme:
    background: str
    plot_background: str
    text: str
    muted: str
    axis: str
    grid: str
    legend_background: str
    legend_border: str
    fwhm_fill: tuple[int, int, int, int]


PLOT_THEMES = {
    APP_THEME.DARK: PlotTheme(
        background="#1e1e1e",
        plot_background="#1e1e1e",
        text="#ffffff",
        muted="#cccccc",
        axis="#777777",
        grid="#3c3c3c",
        legend_background="#1e1e1e",
        legend_border="#555555",
        fwhm_fill=(0, 0, 0, 46),
    ),
    APP_THEME.LIGHT: PlotTheme(
        background="#ffffff",
        plot_background="#ffffff",
        text="#1f1f1f",
        muted="#616161",
        axis="#888888",
        grid="#d6d6d6",
        legend_background="#ffffff",
        legend_border="#c8c8c8",
        fwhm_fill=(255, 255, 255, 184),
    ),
}
