from dataclasses import dataclass
from enum import StrEnum

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


class APP_THEME(StrEnum):
    DARK = "Dark"
    LIGHT = "Light"


@dataclass(frozen=True)
class QtTheme:
    bg: str
    panel: str
    text: str
    muted: str
    border: str
    accent: str
    hover: str
    pressed: str
    highlighted_text: str


QT_THEMES = {
    APP_THEME.DARK: QtTheme(
        bg="#1e1e1e",
        panel="#252526",
        text="#ffffff",
        muted="#cccccc",
        border="#444444",
        accent="#4FC3F7",
        hover="#2f2f2f",
        pressed="#1b1b1b",
        highlighted_text="#000000",
    ),
    APP_THEME.LIGHT: QtTheme(
        bg="#ffffff",
        panel="#f3f3f3",
        text="#1f1f1f",
        muted="#616161",
        border="#d0d0d0",
        accent="#007acc",
        hover="#e8e8e8",
        pressed="#dcdcdc",
        highlighted_text="#ffffff",
    ),
}


def set_app_theme(app: QApplication, theme: APP_THEME = APP_THEME.DARK) -> None:
    """
    Set an Atom / VS Code style theme for the Qt application.
    """
    theme_def = QT_THEMES[theme]
    app.setStyle("Fusion")

    bg = QColor(theme_def.bg)
    panel = QColor(theme_def.panel)
    text = QColor(theme_def.text)
    muted = QColor(theme_def.muted)
    border = QColor(theme_def.border)
    accent = QColor(theme_def.accent)

    pal = QPalette()
    role = QPalette.ColorRole
    pal.setColor(role.Window, bg)
    pal.setColor(role.WindowText, text)
    pal.setColor(role.Base, panel)
    pal.setColor(role.AlternateBase, bg)
    pal.setColor(role.Text, text)
    pal.setColor(role.Button, panel)
    pal.setColor(role.ButtonText, text)
    pal.setColor(role.ToolTipBase, panel)
    pal.setColor(role.ToolTipText, text)
    pal.setColor(role.Highlight, accent)
    pal.setColor(role.HighlightedText, QColor(theme_def.highlighted_text))
    pal.setColor(role.PlaceholderText, muted)
    app.setPalette(pal)

    app.setStyleSheet(f"""
        QMainWindow, QWidget {{
            background-color: {bg.name()};
            color: {text.name()};
        }}

        QMenuBar, QMenu {{
            background-color: {bg.name()};
            color: {text.name()};
            border: 1px solid {border.name()};
        }}
        QMenuBar::item:selected, QMenu::item:selected {{
            background-color: {theme_def.hover};
        }}

        QLabel {{
            color: {text.name()};
        }}

        QGroupBox {{
            border: 1px solid {border.name()};
            border-radius: 6px;
            margin-top: 8px;
            padding: 8px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            color: {muted.name()};
        }}

        QPushButton {{
            background-color: {panel.name()};
            border: 1px solid {border.name()};
            padding: 6px 10px;
            border-radius: 6px;
        }}
        QPushButton:hover {{
            background-color: {theme_def.hover};
        }}
        QPushButton:pressed {{
            background-color: {theme_def.pressed};
        }}
        QPushButton:checked {{
            background-color: {accent.name()};
            color: {theme_def.highlighted_text};
            border: 1px solid {accent.name()};
            font-weight: 600;
        }}
        QPushButton:checked:hover {{
            background-color: {accent.lighter(110).name()};
        }}

        QDoubleSpinBox, QSpinBox, QLineEdit {{
            background-color: {panel.name()};
            border: 1px solid {border.name()};
            padding: 4px 6px;
            border-radius: 6px;
            color: {text.name()};
        }}

        QSlider::groove:horizontal {{
            background: {border.name()};
            height: 6px;
            border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            background: {accent.name()};
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }}
    """)


def set_dark_theme(app: QApplication) -> None:
    """
    Set a dark theme for the Qt application.
    """
    set_app_theme(app, APP_THEME.DARK)


def set_light_theme(app: QApplication) -> None:
    """
    Set a light theme for the Qt application.
    """
    set_app_theme(app, APP_THEME.LIGHT)
