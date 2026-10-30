"""Actual completed-update flush marker and SIGINT-entry recovery tests."""
from PIL import Image
import json,os,subprocess,sys,tempfile
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training'))
from transactions import atomic,read,sha
import gate

def execute(folder,config,wrapper=None):
 command=[sys.executable,'-B',str(wrapper or ROOT/'training/worker.py'),'--config',str(config),'--folder',str(folder)]
 if os.name=='nt' and wrapper:
  # Use the verified local activation route with workspace-relative generated
  # paths, avoiding nested cmd quoting of the workspace's spaces.
  parts=['phase3\\run_cv.cmd','-B',str(wrapper.relative_to(ROOT.parent)),'--config',str(config.relative_to(ROOT.parent)),'--folder',str(folder.relative_to(ROOT.parent))]
  assert all(' ' not in p for p in parts)
  command=[os.environ['COMSPEC'],'/d','/c',' '.join(parts)]
 env=dict(os.environ);env.pop('HF_TOKEN',None)
 r=subprocess.run(command,capture_output=True,text=True,env=env,cwd=ROOT.parent)
 atomic(folder/'RETURN_CODE.json',dict(code=r.returncode,wrapper=bool(wrapper)))
 (folder/('signal.log' if wrapper else 'control.log')).write_text(r.stdout+r.stderr,encoding='utf-8');return r

def main():
 done=read(ROOT/'training_gates/GATE_COMPLETION.json');path=Path(done['devices'][0]['path']);cfg=read(path/'CONFIG.json')
 assert all(sha(ROOT/n)==h for n,h in cfg['source_hashes'].items())
 base=Path(sys.argv[sys.argv.index('--reuse')+1]).resolve() if '--reuse' in sys.argv else Path(tempfile.mkdtemp(prefix='validation_control_',dir=ROOT/'training_gates'))
 assert base.is_relative_to(ROOT/'training_gates');config=base/'CONFIG.json';atomic(config,cfg)
 source=Path(read(path/'published_base_fixture/LATEST_SNAPSHOT.json')['path'])
 expected=gate.store(path/'uninterrupted',cfg).state()
 paused=gate.store(base/'timer',cfg)
 if not paused.chain:paused.install(source)
 if paused.state()['updates']<3:
  atomic(base/'FLUSH_REQUEST.json',dict(reason='actual_completed_update_marker_fixture'))
  r=execute(base/'timer',config);assert r.returncode==91,r.stderr[-3000:]
  s=gate.store(base/'timer',cfg);assert s.state()['updates']==2 and read(base/'timer/FLUSH_ACK.json')['update']==2
  assert (base/'FLUSH_REQUEST.json').resolve().is_relative_to(base.resolve());(base/'FLUSH_REQUEST.json').unlink()
  r=execute(base/'timer',config);assert r.returncode==0,r.stderr[-3000:]
 assert read(base/'timer/FLUSH_ACK.json')['update']==2
 timer_delta=gate.comparison(expected,gate.store(base/'timer',cfg).state())
 signal_case=gate.store(base/'signal',cfg)
 if not signal_case.chain:signal_case.install(source)
 assert signal_case.state()['updates']==1
 # Deliver a real SIGINT at entry to the real instrumented update. This is
 # precise and repeatable; it is not claimed to interrupt a CUDA kernel mid-call.
 wrapper=base/'signal_entry.py'
 wrapper.write_text("import signal,sys\nfrom pathlib import Path\nsys.path.insert(0,"+repr(str(ROOT/'training'))+")\nimport worker\noriginal=worker.instrumented_update\ndef trigger(*args,**kwargs):\n signal.raise_signal(signal.SIGINT)\n return original(*args,**kwargs)\nworker.instrumented_update=trigger\nworker.main()\n",encoding='utf-8')
 r=execute(base/'signal',config,wrapper);assert r.returncode!=0 and 'Stop before incomplete update' in r.stderr,(r.returncode,r.stdout[-1500:],r.stderr[-1500:])
 assert gate.store(base/'signal',cfg).state()['updates']==1
 assert 'InterruptedError' in (base/'signal/FAILURES.jsonl').read_text()
 r=execute(base/'signal',config);assert r.returncode==0,r.stderr[-3000:]
 signal_delta=gate.comparison(expected,gate.store(base/'signal',cfg).state())
 result=dict(status='passed',protocol=done['devices'][0]['protocol'],completed_update_flush_marker_exit=91,
   marker_committed_updates=2,marker_resume_differences=timer_delta,actual_SIGINT_at_update_entry=True,
   SIGINT_retained_previous_completed_update=1,SIGINT_resume_differences=signal_delta,
   mid_CUDA_kernel_signal_tested=False,live_HF_stop_flush_tested=False,fixture_directory=str(base))
 atomic(ROOT/'training_gates/CONTROL_TESTS.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
