"""Shared desktop styling for the Phase 3 inspection window."""
import os
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette


def apply(app):
    fonts = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
    for name in ('segoeui.ttf', 'seguisb.ttf', 'segoeuib.ttf', 'consola.ttf'):
        if (fonts / name).exists():
            QFontDatabase.addApplicationFont(str(fonts / name))

    app.setStyle('Fusion')
    app.setFont(QFont('Segoe UI', 9))
    palette = QPalette()
    for role, color in (
        (QPalette.Window, '#f2f1ee'), (QPalette.WindowText, '#202328'),
        (QPalette.Base, '#fcfbf9'), (QPalette.AlternateBase, '#eeede9'),
        (QPalette.Text, '#202328'), (QPalette.Button, '#faf9f7'),
        (QPalette.ButtonText, '#202328'), (QPalette.Highlight, '#38566a'),
        (QPalette.HighlightedText, '#ffffff'),
    ):
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setStyleSheet(STYLE)


STYLE = """
QMainWindow { background: #f2f1ee; }
QWidget { font-family: "Segoe UI"; font-size: 12px; }
QFrame#rail, QFrame#readout, QFrame#panel { background: #fbfaf8; border: 1px solid #cfcec8; }
QLabel#title, QLabel#hero, QLabel#big { font-size: 16px; font-weight: 600; }
QLabel#muted { color: #52575d; }
QLabel#section, QLabel#eyebrow { font-weight: 600; color: #333e47; }
QLabel#feed { background: #e5e4e0; border: 1px solid #d1d0cb; }
QLabel#error { background: #f4e3df; color: #7b2d20; padding: 7px; }
QPushButton { background: #fbfaf8; border: 1px solid #bdbdb7; border-radius: 2px; padding: 5px 9px; }
QPushButton:hover { background: #e9eef1; border-color: #718794; }
QPushButton:pressed, QPushButton:checked { background: #dce6ec; border-color: #526d80; }
QPushButton:disabled { color: #6a6c6e; background: #eeedea; border-color: #d5d4cf; }
QPushButton#primary { color: white; background: #38566a; border-color: #38566a; font-weight: 600; }
QPushButton#primary:hover { background: #29475a; }
QPushButton#primary:disabled { color: #5d6469; background: #e5e8e9; border-color: #c6ccce; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit {
    background: #fcfbf9; border: 1px solid #bdbdb7; padding: 4px 6px; min-height: 17px;
}
QComboBox QAbstractItemView { background: #fcfbf9; }
QListWidget, QTableWidget { background: #fcfbf9; border: 1px solid #d2d0ca; gridline-color: #dfded9; }
QListWidget#benchModels::item { padding: 5px; border-bottom: 1px solid #dfded9; }
QHeaderView::section { background: #e9e7e3; border: 0; border-right: 1px solid #d0cfca; padding: 4px; }
QScrollArea { border: 0; }
QScrollBar:vertical { background: #eeede9; width: 11px; }
QScrollBar::handle:vertical { background: #a5a7a6; min-height: 28px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QTabWidget::pane { border: 1px solid #cfcec8; }
QTabBar::tab { color: #434b52; padding: 7px 12px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #233d50; background: #fbfaf8; border-bottom: 2px solid #38566a; }
QTabWidget#workspaces::pane { border-left: 0; border-right: 0; }
QTabWidget#workspaces > QTabBar::tab { padding: 9px 21px; font-weight: 600; }
QStatusBar { color: #404b54; background: #e9e8e4; border-top: 1px solid #d2d0ca; }
QStatusBar::item { border: 0; }
"""
