"""Use private Qt bindings; never create caches in the original workstation."""
import os,sys
from pathlib import Path
from paths import ROOT,CACHE,initialize
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
initialize();private=ROOT/'.vendor';sys.path.insert(0,str(private))
for key,folder in [('HF_HOME','huggingface'),('YOLO_CONFIG_DIR','ultralytics'),('MPLCONFIGDIR','matplotlib')]:
 os.environ[key]=str(CACHE/folder)
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
plugins=private/'PySide6/plugins'
os.environ['QT_PLUGIN_PATH']=str(plugins)
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=str(plugins/'platforms')
