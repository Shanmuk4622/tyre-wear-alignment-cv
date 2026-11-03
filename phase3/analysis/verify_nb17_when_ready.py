"""Start the complete audit as soon as the independent public mirror is ready."""
import json,runpy,sys,time
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
folder=Path(json.loads((ROOT/'tread_two/nb17_completion/LATEST.json').read_text())['path'])
while True:
    status=json.loads((folder/'DOWNLOAD_STATUS.json').read_text())
    if status['status']=='complete':break
    print('Waiting for remaining pinned raw bodies:',status['files'],'/',status['total'],'verified files.',flush=True)
    time.sleep(30)
runpy.run_path(str(ROOT/'analysis/verify_nb17_publication.py'),run_name='__main__')
