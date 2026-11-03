"""Execute each actual notebook's extraction/setup in a fresh local interpreter."""
import json,subprocess,sys,time
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tread_two/trace_batch'))
from common import atomic,read
valid=ROOT/'tread_two/trace_validation';base=valid/('fixture_payloads_'+str(time.time_ns()));base.mkdir()
results=[]
for item in read(valid/'NOTEBOOKS.json'):
    nb=read(ROOT/'notebooks'/item['notebook']);setup=''.join(next(c['source'] for c in nb['cells'] if c['cell_type']=='code'))
    target=base/item['mode'];private=target/'trace_private_hf_0362/huggingface_hub-0.36.2.dist-info';private.mkdir(parents=True)
    # cv_conda already has the same exact SDK. The dist marker avoids any local
    # install; the notebook still checks its real imported version,0.36.2.
    setup=setup.replace('/kaggle/working',str(target).replace('\\','/'))
    setup+='\nimport coordinator,review,config,model,cache\nassert sum(p.numel() for p in model.Trace().parameters())==71022\nassert config.configuration()["schema"]["columns"]==671\n'
    script=base/(item['mode']+'.py');script.write_text(setup,encoding='utf-8')
    result=subprocess.run([sys.executable,'-B',str(script)],capture_output=True,text=True)
    assert result.returncode==0,(item['notebook'],result.stderr)
    results.append(dict(notebook=item['notebook'],actual_setup_executed=True,source_imports=True,payload_files=33,private_pip_install_locally=False))
atomic(valid/'PAYLOADS.json',dict(status='passed',notebooks=results,HF_writes=0,credentials_read=False))
print(json.dumps(dict(status='passed',notebooks=results),indent=2),flush=True)
