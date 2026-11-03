"""Bounded active prefixes, immutable direct CSV/NPY shards, atomic complete-unit restore."""
import hashlib,os,shutil
from pathlib import Path
import numpy as np
import torch
from common import atomic,canonical,cpu,digest,file_meta,inside,native,read,restore_rng,rng_state,sha,sync_directory
from schema_fit import SCHEMA,encoded


def prefix_meta(path,length):
    h=hashlib.sha256();remaining=length
    with Path(path).open('rb') as stream:
        while remaining:
            block=stream.read(min(4*1024**2,remaining));assert block,'Truncated committed prefix'
            h.update(block);remaining-=len(block)
    return dict(bytes=length,sha256=h.hexdigest())


def semantic_hash(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            y=x.detach().cpu().contiguous();h.update(canonical([str(y.dtype),list(y.shape)]));h.update(y.numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(canonical([str(x.dtype),list(x.shape)]));h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        elif isinstance(x,(bytes,bytearray)):h.update(x)
        else:h.update(canonical(x))
    visit(value);return h.hexdigest()


class Store:
    def __init__(self,root,protocol):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.protocol=protocol;self.run=protocol[:12]
        self.closed={};self.csv={};self.arrays={};self.generations={};self.unit=0;self.chain=None;self.rows={f:0 for f in SCHEMA}

    def register(self,path,relative=None):
        path=Path(path);name=relative or path.relative_to(self.root).as_posix()
        target=inside(self.root,name)
        if target!=path.resolve():
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        meta=file_meta(target)
        assert name not in self.closed or self.closed[name]==meta,'Immutable artifact changed'
        self.closed[name]=meta
        return name

    def initialize_headers(self):
        for family in SCHEMA:
            target=self.root/'csv'/family/'empty.csv';target.parent.mkdir(parents=True,exist_ok=True)
            body,_=encoded(family,[],self.run,header=True)
            if target.exists():assert target.read_bytes()==body
            else:target.write_bytes(body)
            self.register(target)

    def append_csv(self,family,records,seal=False):
        body,count=encoded(family,records,self.run)
        if not count:return
        if family not in self.csv:
            generation=self.generations.get('csv:'+family,0)
            relative=f'active/csv_{family}_{generation:05d}.csv'
            path=inside(self.root,relative);path.parent.mkdir(parents=True,exist_ok=True)
            header,_=encoded(family,[],self.run,header=True)
            path.write_bytes(header)
            self.csv[family]=dict(path=relative,bytes=len(header),rows=0,generation=generation)
        state=self.csv[family];path=inside(self.root,state['path'])
        with path.open('r+b') as stream:
            stream.truncate(state['bytes']);stream.seek(state['bytes']);stream.write(body);stream.flush();os.fsync(stream.fileno())
        state['bytes']+=len(body);state['rows']+=count;self.rows[family]+=count
        if seal or state['bytes']>=8*1024**2:self.seal_csv(family)

    def seal_csv(self,family):
        state=self.csv.pop(family)
        target=self.root/f'csv/{family}/{state["generation"]:05d}.csv';target.parent.mkdir(parents=True,exist_ok=True)
        self.copy_prefix(inside(self.root,state['path']),target,state['bytes'])
        self.register(target);self.generations['csv:'+family]=state['generation']+1

    def append_native(self,name,array,first_sample):
        array=np.asarray(array);assert array.dtype in (np.dtype('float32'),np.dtype('bool'))
        assert np.isfinite(array).all() and array.ndim>=1
        if name not in self.arrays:
            generation=self.generations.get('native:'+name,0)
            relative=f'active/native_{name}_{generation:05d}.raw'
            path=inside(self.root,relative);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'')
            self.arrays[name]=dict(path=relative,bytes=0,samples=0,shape=list(array.shape[1:]),dtype=str(array.dtype),generation=generation,first_sample=first_sample,last_sample=first_sample-1)
        state=self.arrays[name];assert state['shape']==list(array.shape[1:]) and state['dtype']==str(array.dtype)
        body=array.tobytes(order='C');path=inside(self.root,state['path'])
        with path.open('r+b') as stream:
            stream.truncate(state['bytes']);stream.seek(state['bytes']);stream.write(body);stream.flush();os.fsync(stream.fileno())
        state['bytes']+=len(body);state['samples']+=len(array);state['last_sample']=first_sample+len(array)-1

    def seal_native(self,name,epoch,unit):
        state=self.arrays.pop(name)
        target=self.root/f'native/{name}/{state["generation"]:05d}.npy';target.parent.mkdir(parents=True,exist_ok=True)
        mapped=np.memmap(inside(self.root,state['path']),mode='r',dtype=state['dtype'],shape=(state['samples'],*state['shape']))
        native(target,mapped);del mapped
        relative=self.register(target);meta=self.closed[relative]
        self.generations['native:'+name]=state['generation']+1
        return dict(file=relative,kind=name,epoch=epoch,first_sample=state['first_sample'],last_sample=state['last_sample'],bytes=meta['bytes'],sha256=meta['sha256'],immutable=True,committed_unit=unit)

    def commit(self,model,optimizer,cursor,records,arrays=None,seal=False,crash=None,kind=None):
        unit=self.unit+1;first=cursor.get('last_first_sample',cursor.get('sample',0))
        for name,value in (arrays or {}).items():self.append_native(name,value,first)
        indexes=[]
        for name in list(self.arrays):
            if seal or self.arrays[name]['samples']>=64:indexes.append(self.seal_native(name,cursor.get('record_epoch',cursor.get('epoch',0)),unit))
        if indexes:records.setdefault('native_chunks_index',[]).extend(indexes)
        for family,rows in records.items():self.append_csv(family,rows)
        if crash=='after_evidence':raise InterruptedError('fixture_after_evidence')
        if seal:
            for family in list(self.csv):self.seal_csv(family)
        tensor_state=dict(model=cpu(model.state_dict()),optimizer=cpu(optimizer.state_dict()),rng=rng_state(),cursor=cpu(cursor),precision='FP32_no_scaler',scheduler='cosine_at_previous_completed_global_update')
        state_hash=semantic_hash(tensor_state)
        csv_prefix={name:prefix_meta(inside(self.root,v['path']),v['bytes']) for name,v in self.csv.items()}
        native_prefix={name:prefix_meta(inside(self.root,v['path']),v['bytes']) for name,v in self.arrays.items()}
        chain=digest(dict(previous=self.chain,unit=unit,state=state_hash,csv=csv_prefix,native=native_prefix))
        self.append_csv('unit_ledger',[dict(unit=unit,kind=kind or cursor.get('record_stage',cursor['phase']),update=cursor.get('update',0),csv_prefix_sha256=digest(csv_prefix),native_state_sha256=state_hash,previous_unit_sha256=self.chain,current_unit_sha256=chain,state_semantic_sha256=state_hash)])
        state=dict(tensor_state,protocol=self.protocol,unit=unit,chain=chain,closed=self.closed,csv=self.csv,arrays=self.arrays,generations=self.generations,rows=self.rows,
                   csv_prefix={n:prefix_meta(inside(self.root,v['path']),v['bytes']) for n,v in self.csv.items()},native_prefix=native_prefix)
        target=self.root/f'state_{unit%2}.pt';native(target,state)
        if crash=='after_state':raise InterruptedError('fixture_after_state')
        pointer=dict(protocol=self.protocol,unit=unit,chain=chain,state=target.name,state_meta=file_meta(target))
        atomic(self.root/'POINTER.json',pointer)
        self.unit,self.chain=unit,chain
        self.cleanup_active()
        if crash=='after_pointer':raise InterruptedError('fixture_after_pointer')
        return pointer

    def restore(self,model,optimizer,verify_closed=True):
        pointer=read(self.root/'POINTER.json');assert pointer['protocol']==self.protocol
        assert file_meta(self.root/pointer['state'])==pointer['state_meta']
        state=torch.load(self.root/pointer['state'],map_location='cpu',weights_only=False)
        assert state['protocol']==self.protocol and state['chain']==pointer['chain']
        if verify_closed:
            for name,meta in state['closed'].items():assert file_meta(inside(self.root,name))==meta,name
        for key,prefixkey in [('csv','csv_prefix'),('arrays','native_prefix')]:
            for name,v in state[key].items():
                path=inside(self.root,v['path']);assert prefix_meta(path,v['bytes'])==state[prefixkey][name]
                with path.open('r+b') as stream:stream.truncate(v['bytes'])
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        self.closed,self.csv,self.arrays,self.generations,self.rows=state['closed'],state['csv'],state['arrays'],state['generations'],state['rows']
        self.unit,self.chain=state['unit'],state['chain']
        return state['cursor']

    def cleanup_active(self):
        # Keep the last two generations, including the preceding state's prefix.
        for path in (self.root/'active').glob('*'):
            stem=path.stem
            try:kind,name,generation=stem.split('_',1)[0],stem.rsplit('_',1)[0],int(stem.rsplit('_',1)[1])
            except (ValueError,IndexError):continue
            key=('csv:' if kind=='csv' else 'native:')+name.split('_',1)[1]
            current=self.generations.get(key,0)
            if generation<current-2:
                assert path.resolve().is_relative_to((self.root/'active').resolve());path.unlink()

    @staticmethod
    def copy_prefix(source,target,length):
        target.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as a,target.open('wb') as b:
            remaining=length
            while remaining:
                data=a.read(min(4*1024**2,remaining));assert data;b.write(data);remaining-=len(data)
            b.flush();os.fsync(b.fileno())

    def snapshot(self,target,allow_copy=False):
        target=Path(target);target.mkdir(parents=True,exist_ok=True)
        pointer=read(self.root/'POINTER.json')
        assert pointer['protocol']==self.protocol and file_meta(self.root/pointer['state'])==pointer['state_meta']
        state=torch.load(self.root/pointer['state'],map_location='cpu',weights_only=False)
        inventory={}
        for name,meta in state['closed'].items():
            source=inside(self.root,name);dest=inside(target,name);dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                try:os.link(source,dest)
                except OSError:
                    assert allow_copy,'Same-filesystem hardlinks required for the measured20GB staging budget'
                    shutil.copyfile(source,dest)
            assert file_meta(dest)==meta;inventory[name]=meta
        active_bytes=0
        for category,prefix in [('csv','csv_prefix'),('arrays','native_prefix')]:
            for name,v in state[category].items():
                source=inside(self.root,v['path']);dest=inside(target,v['path'])
                self.copy_prefix(source,dest,v['bytes']);meta=file_meta(dest)
                assert meta==state[prefix][name];inventory[v['path']]=meta;active_bytes+=v['bytes']
        for name in ('POINTER.json',pointer['state']):
            dest=target/name;shutil.copyfile(self.root/name,dest);inventory[name]=file_meta(dest)
        assert active_bytes<128*1024**2,'Active snapshot exceeds bounded staging allowance'
        atomic(target/'MANIFEST.json',dict(contract=self.protocol,files=inventory))
        return dict(path=str(target),active_copy_bytes=active_bytes,files=len(inventory))
