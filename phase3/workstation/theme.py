import os
from pathlib import Path
from PySide6.QtGui import QColor,QFont,QPalette,QFontDatabase,QPixmap,QPainter,QPen,QIcon
from PySide6.QtCore import Qt,QRectF
def apply(app):
 # The offscreen Windows platform has no automatic system-font discovery.
 # Register the existing OS fonts for this process only; never install fonts.
 for name in ('segoeui.ttf','segoeuib.ttf','segoeuil.ttf'):
  path=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/name
  if path.exists():QFontDatabase.addApplicationFont(str(path))
 app.setStyle('Fusion');app.setFont(QFont('Segoe UI',10));palette=QPalette()
 for role,color in [(QPalette.Window,'#0c1721'),(QPalette.WindowText,'#e9f2f6'),(QPalette.Base,'#111f2b'),(QPalette.AlternateBase,'#182b38'),(QPalette.Text,'#e9f2f6'),(QPalette.Button,'#203645'),(QPalette.ButtonText,'#e9f2f6'),(QPalette.Highlight,'#28b99b'),(QPalette.HighlightedText,'#061c1a')]:palette.setColor(role,QColor(color))
 icon=QPixmap(128,128);icon.fill(Qt.transparent);p=QPainter(icon);p.setRenderHint(QPainter.Antialiasing);p.setPen(Qt.NoPen);p.setBrush(QColor('#32dbb0'));p.drawRoundedRect(QRectF(4,4,120,120),27,27);p.setPen(QPen(QColor('#0b3630'),8))
 for x in (36,64,92):p.drawLine(x,29,x-8,99)
 p.end();app.setWindowIcon(QIcon(icon))
 app.setPalette(palette);app.setStyleSheet('''
 QWidget{color:#e4eef3;font-family:'Segoe UI';font-size:13px;}
 QMainWindow{background:#0b1620;} QFrame#header{background:#132735;border-bottom:1px solid #31505e;}
 QFrame#rail{background:#10212d;border:1px solid #263e4d;border-radius:12px;}
 QFrame#readout{background:#10212d;border:1px solid #263e4d;border-radius:12px;}
 QLabel#brand{font-size:26px;font-weight:700;letter-spacing:2px;color:#effafa;}
 QLabel#mark{background:#36dbb1;color:#082d27;border-radius:12px;font-size:26px;font-weight:800;padding:10px;}
 QLabel#muted{color:#8faab8;font-size:12px;} QLabel#section{color:#73d9c3;font-weight:700;font-size:12px;letter-spacing:1px;}
 QLabel#hero{font-size:22px;font-weight:700;color:#e8f6f5;}
 QLabel#pill{background:#1c403e;color:#7ce8c8;border-radius:10px;padding:6px 12px;}
 QLabel#error{background:#4a2828;color:#ffc8b8;padding:10px;border-radius:6px;}
 QPushButton{background:#1d3443;border:1px solid #365462;border-radius:7px;padding:9px 12px;}
 QPushButton:hover{background:#29495a;border-color:#53bca6;} QPushButton:pressed{background:#125449;}
 QPushButton:disabled{color:#607783;background:#142734;border-color:#213b48;}
 QPushButton#primary{background:#42dcb5;color:#09251f;font-weight:700;border:0;padding:11px 16px;}
 QPushButton#primary:hover{background:#6be8c8;} QPushButton#primary:disabled{background:#265449;color:#91b8ac;}
 QComboBox,QLineEdit,QSpinBox,QTextEdit,QListWidget{background:#0d1b26;border:1px solid #304b5a;border-radius:6px;padding:7px;selection-background-color:#27796b;}
 QComboBox QAbstractItemView{background:#152d3a;selection-background-color:#275f59;}
 QCheckBox{spacing:7px;} QCheckBox::indicator{width:15px;height:15px;}
 QSlider::groove:horizontal{height:4px;background:#2d4655;border-radius:2px;} QSlider::handle:horizontal{background:#55d9bc;width:13px;margin:-5px 0;border-radius:6px;}
 QTabWidget::pane{border:0;} QTabBar::tab{background:#122735;color:#91aab8;padding:10px 17px;border-bottom:2px solid #223b48;}
 QTabBar::tab:selected{color:#6ce4c2;border-bottom:2px solid #57d9b5;}
 QTableWidget{background:#10212d;gridline-color:#2a4352;border:0;} QHeaderView::section{background:#1a3240;color:#a8c0cb;padding:5px;border:0;}
 QScrollArea{background:transparent;border:0;} QScrollBar:vertical{background:#10212d;width:9px;} QScrollBar::handle:vertical{background:#365363;border-radius:4px;min-height:30px;}
 QStatusBar{color:#92b0be;background:#0c1a24;border-top:1px solid #294352;}
 ''')
