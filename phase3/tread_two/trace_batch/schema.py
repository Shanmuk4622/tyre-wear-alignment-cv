"""Executable direct-CSV schema, exact finite formatting and explicit missingness."""
import csv,io,math
from common import HERE,read
SCHEMA=read(HERE/'SCHEMAS.json')['families']

def encoded(family,records,run,header=False):
    columns=SCHEMA[family]['columns'];names=[c['name'] for c in columns]
    stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n')
    if header:writer.writerow(names)
    count=0
    for record in records:
        assert not set(record)-set(names),(family,set(record)-set(names))
        row=[]
        for col in columns:
            name,kind=col['name'],col['dtype'];value=record.get(name,run if name=='run' else None)
            if value is None:
                assert col.get('nullable',True),(family,name,'Required value missing')
                row.append('');continue
            if kind=='utf8':row.append(str(value))
            elif kind=='bool':
                assert value in (0,1,True,False),(family,name,value);row.append(str(int(value)))
            elif kind=='int64':
                n=int(value);assert n==value,(family,name,value);row.append(str(n))
            else:
                n=float(value);assert math.isfinite(n),(family,name,n)
                row.append(format(n,'.9g' if kind=='float32' else '.17g'))
        writer.writerow(row);count+=1
    return stream.getvalue().encode('utf-8'),count

def add(records,family,stage,**fields):
    records.setdefault(family,[]).append(dict(stage=stage,**fields))

def specification():
    return dict(status='executable_pre_fit_contract',families=SCHEMA,CSV_families=len(SCHEMA),
                columns=sum(len(f['columns']) for f in SCHEMA.values()),float32_precision=9,float64_precision=17)
