"""All mutable station state belongs to this independent application folder."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PHASE3=ROOT.parent
DATA=ROOT/'data';CACHE=ROOT/'.cache';MODELS=ROOT/'models'
def initialize():
 for p in (DATA,CACHE,DATA/'captures',DATA/'sessions'):p.mkdir(parents=True,exist_ok=True)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(value,indent=2,allow_nan=False)+'\n'
 if path.exists() and path.read_text(encoding='utf-8')==text:return
 path.write_text(text,encoding='utf-8')
def model_record():return read(MODELS/'MODEL.json')
