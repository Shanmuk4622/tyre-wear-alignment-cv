"""Keep prototype packages, downloads and configuration inside this folder."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.vendor'))
sys.path.insert(0, str(ROOT.parent / 'tyrelib'))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.cache' / 'ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('YOLO_OFFLINE', 'true')
for key in ('HF_HOME', 'YOLO_CONFIG_DIR', 'MPLCONFIGDIR'):
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
qt_plugins = ROOT / '.vendor' / 'PySide6' / 'plugins'
if qt_plugins.exists():
    os.environ['QT_PLUGIN_PATH'] = str(qt_plugins)
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(qt_plugins / 'platforms')
