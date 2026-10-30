"""Qt supervises one owned Torch subprocess; UI never runs a model on its thread."""
import json,os,queue,subprocess,sys
from pathlib import Path
from PySide6.QtCore import QThread,Signal
from paths import ROOT,CACHE
from imaging import pack,unpack,working_image
class InferenceBridge(QThread):
 ready=Signal(object);result=Signal(object,object);error=Signal(str)
 def __init__(self,device='auto'):
  super().__init__();self.device=device;self.jobs=queue.Queue(maxsize=2);self.process=None;self.stopping=False
 def submit(self,request_id,rgb):self.jobs.put_nowait(dict(op='predict',request_id=request_id,image=pack(working_image(rgb))))
 def request_stop(self):
  self.stopping=True
  try:self.jobs.put_nowait(None)
  except queue.Full:pass
 def kill_owned_process(self):
  if self.process and self.process.poll() is None:self.process.terminate()
 def run(self):
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');kwargs={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
  try:
   with (CACHE/'inference-worker.log').open('a',encoding='utf-8') as log:
    python=Path(sys.executable).with_name('python.exe') if os.name=='nt' else Path(sys.executable)
    self.process=subprocess.Popen([str(python),'-B','-u',str(ROOT/'worker.py'),'--device',self.device],cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',**kwargs)
    def receive():
     line=self.process.stdout.readline()
     if not line:raise RuntimeError('Inference process closed. See the station worker log; restart the model from the device selector.')
     value=json.loads(line)
     if value.get('status')=='error':raise RuntimeError(value['message'])
     return value
    value=receive();assert value['status']=='ready';self.ready.emit(value)
    while not self.stopping:
     job=self.jobs.get()
     if job is None:break
     self.process.stdin.write(json.dumps(job,separators=(',',':'))+'\n');self.process.stdin.flush();value=receive();assert value['status']=='result'
     result=value['result'];result['arrays']={k:unpack(v) for k,v in result['arrays'].items()};self.result.emit(value['request_id'],result)
    if self.process.poll() is None:self.process.stdin.write('{"op":"quit"}\n');self.process.stdin.flush()
    self.process.wait(timeout=5)
  except Exception as exc:
   if not self.stopping:self.error.emit(str(exc))
  finally:
   if self.process and self.process.poll() is None:self.process.terminate();self.process.wait(timeout=5)

class CaptureBridge(QThread):
 saved=Signal(str);error=Signal(str)
 def __init__(self,rgb,result,source,note,captures=None):super().__init__();self.values=(rgb,result,source,note,captures)
 def run(self):
  try:
   from records import save_capture
   self.saved.emit(str(save_capture(*self.values)))
  except Exception as exc:self.error.emit(str(exc))
