"""Private long-lived inference subprocess; JSON lines, one frame at a time."""
import sys
sys.dont_write_bytecode=True
import argparse,json,traceback
from paths import initialize
from imaging import unpack,pack
from engine import Engine
def emit(value):print(json.dumps(value,allow_nan=False,separators=(',',':')),flush=True)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda:0']);args=parser.parse_args();initialize()
 try:engine=Engine(args.device);emit(dict(status='ready',identity=engine.identity,environment=engine.environment))
 except Exception as exc:emit(dict(status='error',message=str(exc)));traceback.print_exc(file=sys.stderr);return
 for line in sys.stdin:
  try:
   request=json.loads(line)
   if request.get('op')=='quit':break
   assert request['op']=='predict';rgb=unpack(request['image']);assert rgb.shape==(512,384,3)
   result=engine.predict(rgb);result['arrays']={k:pack(v) for k,v in result['arrays'].items()};emit(dict(status='result',request_id=request['request_id'],result=result))
  except Exception as exc:emit(dict(status='error',message=str(exc)));traceback.print_exc(file=sys.stderr)
if __name__=='__main__':main()
