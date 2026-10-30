"""Stop only our exact audit worker so its public transfer strategy can resume."""
import json,sys
from pathlib import Path
import psutil
assert Path(sys.prefix).name=='cv_conda'
stopped=[]
for p in psutil.process_iter(['pid','name','cmdline']):
    try:
        if p.info['name'].lower() not in ('python.exe','python'):continue
        args=p.info['cmdline'] or []
        if not any(Path(a).name=='verify_nb11_publication.py' for a in args):continue
        p.terminate();p.wait(timeout=15);stopped.append(p.pid)
    except (psutil.NoSuchProcess,psutil.AccessDenied):continue
print(json.dumps(dict(stopped_owned_audit_workers=stopped,other_processes_changed=False,verified_cache_retained=True)))
