def apply_theme(app):
    from pathlib import Path
    from PySide6.QtGui import QFont, QFontDatabase
    # Explicit loading also works in offscreen sessions without registry font
    # enumeration. Fonts remain OS assets; they are not copied into the repo.
    for name in ('segoeui.ttf', 'seguisb.ttf', 'consola.ttf'):
        path = Path('C:/Windows/Fonts') / name
        if path.exists():
            QFontDatabase.addApplicationFont(str(path))
    app.setFont(QFont('Segoe UI', 10))
    app.setStyle('Fusion')
    from PySide6.QtGui import QPalette, QColor
    palette = QPalette()
    for role, color in ((QPalette.Window, '#f4f7fa'), (QPalette.WindowText, '#213447'),
                        (QPalette.Base, '#ffffff'), (QPalette.Text, '#213447'),
                        (QPalette.Button, '#ffffff'), (QPalette.ButtonText, '#213447'),
                        (QPalette.Highlight, '#187f83'), (QPalette.HighlightedText, '#ffffff')):
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setStyleSheet(STYLE)


STYLE = '''
QMainWindow, QWidget#root { background: #f4f7fa; color: #213447; }
QWidget { color: #213447; font-family: "Segoe UI"; font-size: 13px; }
QFrame#panel { background: #ffffff; border: 1px solid #d7e1e7; border-radius: 10px; }
QLabel#eyebrow { color: #16767c; font-size: 10px; font-weight: 700; letter-spacing: 2px; }
QLabel#title { font-size: 30px; font-weight: 600; }
QLabel#muted { color: #5a7081; font-size: 12px; }
QLabel#big { font-size: 22px; font-weight: 600; }
QPushButton { background: #ffffff; border: 1px solid #cbd8e0; border-radius: 6px; padding: 10px 14px; font-weight: 600; }
QPushButton:hover { background: #e8f3f3; border-color: #388e92; }
QPushButton:pressed, QPushButton:checked { background: #d4edeb; border-color: #187f83; }
QPushButton:disabled { color: #94a2ad; background: #edf1f4; border-color: #dce4e9; }
QPushButton#primary { background: #187f83; color: #ffffff; border: 0; }
QPushButton#primary:hover { background: #126d72; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox { background: #ffffff; border: 1px solid #cbd8e0; border-radius: 5px; padding: 8px; }
QComboBox QAbstractItemView { background: #ffffff; selection-background-color: #d4edeb; color: #213447; }
QListWidget { background: transparent; border: 0; outline: 0; }
QListWidget::item { border-bottom: 1px solid #e1e8ed; padding: 12px 6px; }
QListWidget::item:selected { background: #e3f2ef; border-radius: 5px; }
QScrollArea { background: transparent; border: 0; }
QScrollBar:vertical { background: #edf2f5; width: 8px; }
QScrollBar::handle:vertical { background: #aac0cd; border-radius: 4px; min-height: 25px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QStatusBar { background: #e7eef3; color: #486373; }
QToolTip { color: #213447; background: #ffffff; border: 1px solid #94b9be; padding: 6px; }
QSlider::groove:horizontal { height: 5px; background: #cfdee5; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #187f83; border-radius: 2px; }
QSlider::handle:horizontal { background: #ffffff; border: 2px solid #187f83; width: 12px; margin: -5px 0; border-radius: 6px; }
'''
