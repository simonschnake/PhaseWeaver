from PySide6.QtGui import QColor, QPalette, QFont
from PySide6.QtWidgets import QApplication

def set_dark_theme(app: QApplication) -> None:
    """
    Set a dark theme for the Qt application.
    """
    app.setStyle("Fusion")

    bg = QColor("#1e1e1e")
    panel = QColor("#252526")
    text = QColor("#ffffff")
    muted = QColor("#cccccc")
    border = QColor("#444444")
    accent = QColor("#4FC3F7")

    pal = QPalette()
    pal.setColor(QPalette.Window, bg)
    pal.setColor(QPalette.WindowText, text)
    pal.setColor(QPalette.Base, panel)          # input background
    pal.setColor(QPalette.AlternateBase, bg)
    pal.setColor(QPalette.Text, text)
    pal.setColor(QPalette.Button, panel)
    pal.setColor(QPalette.ButtonText, text)
    pal.setColor(QPalette.ToolTipBase, panel)
    pal.setColor(QPalette.ToolTipText, text)
    pal.setColor(QPalette.Highlight, accent)
    pal.setColor(QPalette.HighlightedText, QColor("#000000"))
    pal.setColor(QPalette.PlaceholderText, muted)
    app.setPalette(pal)

    # Optional: try to align fonts with your mplstyle (falls back if not installed)
    #app.setFont(QFont("Inter", 10))

    # A small stylesheet for “polish” (borders, sliders, hover states)
    app.setStyleSheet(f"""
        QMainWindow, QWidget {{
            background-color: {bg.name()};
            color: {text.name()};
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
            background-color: #2f2f2f;
        }}
        QPushButton:pressed {{
            background-color: #1b1b1b;
        }}

        QDoubleSpinBox, QSpinBox, QLineEdit {{
            background-color: {panel.name()};
            border: 1px solid {border.name()};
            padding: 4px 6px;
            border-radius: 6px;
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