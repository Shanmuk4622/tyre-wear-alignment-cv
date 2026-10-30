"""Preserve and inspect the supplied executed NB07 without executing its cells."""
import ast,base64,gzip,hashlib,json,re,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import nbformat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'data_gate'))
from common_gate import atomic,read,sha,inside
OUT=ROOT/'data_gate_validation/nb07_publication'

def save_json(path,value):
 if path.exists() and read(path)==value:return
 atomic(path,value)

def inspect():
 OUT.mkdir(parents=True,exist_ok=True);path=ROOT/'notebooks/phase3_NB07_TREAD_ONE_Data_And_Protocol_Gate.ipynb';raw=path.read_bytes();archive=OUT/'executed.ipynb'
 if archive.exists():assert archive.read_bytes()==raw,'Keep distinct supplied executions in separate audit folders'
 else:archive.write_bytes(raw)
 nb=nbformat.reads(raw.decode('utf-8'),as_version=4);nbformat.validate(nb);counts=[];errors=[];files={};receipts=[];summaries=[];streams=[]
 for cell in nb.cells:
  if cell.cell_type!='code':continue
  counts.append(cell.execution_count)
  for output in cell.outputs:
   if output.output_type=='error':errors.append(dict(cell=cell.execution_count,name=output.ename,message=re.sub(r'hf_[A-Za-z0-9]+','[REDACTED]',output.evalue)))
   if output.output_type!='stream':continue
   value=output.get('text','');value=''.join(value) if isinstance(value,list) else value;value=re.sub(r'hf_[A-Za-z0-9]+','[REDACTED]',value);streams.append(value)
   for offset,char in enumerate(value):
    if char!='{':continue
    try:item,_=json.JSONDecoder().raw_decode(value[offset:])
    except ValueError:continue
    if not isinstance(item,dict):continue
    if item.get('status')=='published_and_verified':receipts.append(item)
    if item.get('status')=='audit_completed':summaries.append(item)
  for node in ast.walk(ast.parse(cell.source)):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):
    payload=json.loads(gzip.decompress(base64.b64decode(node.args[0].value)))
    for name,value in payload.items():
     body=base64.b64decode(value);assert body==inside(ROOT,name).read_bytes(),name;files[name]=hashlib.sha256(body).hexdigest()
 delivery=read(ROOT/'data_gate_validation/DELIVERY.json');assert files==delivery['embedded_files'] and len(files)==42
 result=dict(status='inspected',sha256=sha(path),bytes=len(raw),execution_counts=counts,errors=errors,embedded_files=files,publication_receipts=receipts,audit_summaries=summaries,clean_delivery_sha256=delivery['notebook_sha256'])
 save_json(OUT/'NOTEBOOK_INSPECTION.json',result);stream_text='\n'.join(streams);stream_path=OUT/'notebook_streams.txt'
 if not stream_path.exists() or stream_path.read_text(encoding='utf-8')!=stream_text:stream_path.write_text(stream_text,encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k!='embedded_files'},indent=2))
 assert not errors and len(counts)==5 and all(type(n) is int and n>0 for n in counts) and counts==sorted(set(counts)) and receipts and all(r==receipts[0] for r in receipts),'Notebook must complete all five cells in order and retain its verified publication receipt'
 assert receipts[-1]['files']==46 and receipts[-1]['zip_uploads']==0 and receipts[-1]['owner_released']
 save_json(OUT/'PUBLICATION.json',receipts[-1]);return result

if __name__=='__main__':inspect()
