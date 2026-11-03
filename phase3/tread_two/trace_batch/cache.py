"""Chunk-atomic fixed atlas/target construction; train-only fitted statistics."""
import os,time
from pathlib import Path
import numpy as np
import torch
from common import HERE,atomic,read,sha,file_meta,digest,native,canonical
from schema import encoded
from data import dataset,load_original
from frozen_anchor import TreadOne
from frozen_geometry import FrozenGeometry,EXPORT_SHA256
from atlas_geo import canonicalize
from targets import fixed_targets,fit_statistics,tiles,CHANNELS

def table(root,family,rows,protocol,name=None):
    path=Path(root)/(name or f'csv/{family}/00000.csv');path.parent.mkdir(parents=True,exist_ok=True)
    body,_=encoded(family,rows,protocol[:12],header=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path);return path

def prepare(root,data_root,export,protocol,geometry_device='cuda:1',fixture=False,service=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    rows=dataset(data_root,verify=True)
    if fixture:
        # Disposable actual-data gate uses eight train images twice; held-out
        # rows never supply fitted stats or gate optimizer observations.
        rows=[dict(r,slot=i) for i,r in enumerate([r for r in rows if r['role']=='train'][:8])]
    identity=dict(protocol=protocol,export_sha256=EXPORT_SHA256,rows_sha256=digest(rows),fixture=fixture)
    pointer=root/'CACHE.json'
    index=read(pointer) if pointer.exists() else dict(identity=identity,files={},chunks=[],rows=rows,completed=False)
    assert index['identity']==identity,'Cache identity changed; preserve original run'
    for name,meta in index['files'].items():assert file_meta(root/name)==meta,name
    if index['completed']:return Cache(root)
    geometry=FrozenGeometry(export,HERE/'frozen_anchor.py',TreadOne,geometry_device)
    frozen=digest({k:sha_bytes(v) for k,v in geometry.state_dict().items()})
    for start in range(len(index['chunks'])*8,len(rows),8):
        if service:service(index)
        from resources import peaks_reset,records as resource_records
        peaks_reset()
        selected=rows[start:start+8];values={k:[] for k in ['rgb','valid','target','weight','row_parameters','row_valid','geometry_logits','points']}
        target_records=[];sentinels=[];diagnostics=[]
        for row in selected:
            a,m,p=load_original(data_root,row)
            rgb=torch.from_numpy(a[None]).to(geometry_device);guides=rgb.new_tensor(row['guide_fractions'])[None]
            result=geometry(rgb,guides)
            masks=torch.from_numpy(m[None]).float().to(geometry_device)
            atlas=canonicalize(rgb,result['masks'],masks[:,2:3])
            body=fixed_targets(tiles(atlas['rgb'].cpu()),tiles(atlas['valid'].cpu()),tiles((1-atlas['ignored']).cpu()))
            for k,v in [('rgb',atlas['rgb']),('valid',atlas['valid'].bool()),('target',body['target']),('weight',body['weight']),
                        ('row_parameters',atlas['row_parameters']),('row_valid',atlas['row_valid']),('geometry_logits',result['seg_logits']),('points',result['points'])]:
                values[k].append(v.detach().cpu().numpy())
            for t in range(32):
                for y in range(4):
                    for x in range(4):
                        target_records.append(dict(stage='fixed_targets',image_id=row['image_id'],target_revision='trace-fixed-target-v2',tile=t,cell_y=y,cell_x=x,
                            valid_pixels=int(body['weight'][0,t,0,y,x]),**{c:float(body['target'][0,t,k,y,x]) for k,c in enumerate(CHANNELS)}))
            sentinels.append(dict(stage='frozen_geometry_cache',image_id=row['image_id'],source_sha256=row['image_sha256'],frozen_export_sha256=EXPORT_SHA256,
                logits_sha256=sha_bytes(result['seg_logits']),points_sha256=sha_bytes(result['points']),human_points_available=row['point_kind']=='human'))
        chunk={};number=start//8
        for kind,parts in values.items():
            path=root/f'native/cache_{kind}/{number:05d}.npy';native(path,np.concatenate(parts));chunk[path.relative_to(root).as_posix()]=file_meta(path)
        for family,records in [('target_cells',target_records),('geometry_sentinels',sentinels)]:
            path=table(root,family,records,protocol,f'csv/{family}/{number:05d}.csv');chunk[path.relative_to(root).as_posix()]=file_meta(path)
        path=table(root,'resource_measurements',resource_records(root,'geometry_cache',number),protocol,f'csv/resource_measurements/{number:05d}.csv');chunk[path.relative_to(root).as_posix()]=file_meta(path)
        index['files'].update(chunk);index['chunks'].append(dict(first=start,last=start+len(selected)-1,files=chunk))
        atomic(pointer,index)
        print('TRACE fixed cache:',min(start+8,len(rows)),'/',len(rows),'images; completed chunks retained.',flush=True)
    assert digest({k:sha_bytes(v) for k,v in geometry.state_dict().items()})==frozen
    del geometry
    if torch.cuda.is_available():torch.cuda.empty_cache()
    cache=Cache(root);train=[r['slot'] for r in rows if r['role']=='train']
    target=torch.from_numpy(cache.array('target',train));weight=torch.from_numpy(cache.array('weight',train))
    stats=fit_statistics(target,weight,['train']*len(train));native(root/'controls/target_statistics.pt',stats)
    files=['controls/target_statistics.pt']
    records=[]
    for k,name in enumerate(CHANNELS):
        count=float(weight.double().sum());weighted_sum=float((target.double()*weight.double()).sum((0,1,3,4))[k]);ss=float((target.double().square()*weight.double()).sum((0,1,3,4))[k])
        records.append(dict(stage='train_only_target_fit',channel=name,weighted_sum=weighted_sum,weighted_square_sum=ss,count=count,mean=float(stats['mean'][k]),std=float(stats['std'][k]),floor_applied=max(0,ss/count-(weighted_sum/count)**2)**.5<.001,fit_role='train'))
    path=table(root,'target_normalization',records,protocol);files.append(path.relative_to(root).as_posix())
    records=[]
    for t in range(32):
        for k,c in enumerate(CHANNELS):
            for y in range(4):
                for x in range(4):
                    count=float(stats['position_count'][t,0,y,x]);mean=float(stats['position_mean'][t,k,y,x])
                    records.append(dict(stage='train_only_target_fit',tile=t,channel=c,cell_y=y,cell_x=x,weighted_sum=float((target[:,t,k,y,x].double()*weight[:,t,0,y,x].double()).sum()),count=count,mean=mean,available=count>0))
    path=table(root,'target_position_control',records,protocol);files.append(path.relative_to(root).as_posix())
    records=[dict(stage='source_binding',image_id=r['image_id'],source_sha256=r['image_sha256'],role=r['role'],component=r['component'],domain=r['domain'],label_proxy=r['class_index'],geometry_kind=r['point_kind'],original_path=r['image_path']) for r in rows]
    path=table(root,'source_images',records,protocol);files.append(path.relative_to(root).as_posix())
    diagnostics=[];spaces=[]
    from metrics import latent
    for role in sorted({r['role'] for r in rows}):
        ids=[r['slot'] for r in rows if r['role']==role];a=cache.array('target',ids);w=cache.array('weight',ids)
        for k,c in enumerate(CHANNELS):
            use=a[:,:,k][w[:,:,0]>0]
            diagnostics.append(dict(stage='fixed_target_diagnostics',role=role,channel=c,count=int(use.size),mean=float(use.mean()) if use.size else None,std=float(use.std()) if use.size else None,minimum=float(use.min()) if use.size else None,maximum=float(use.max()) if use.size else None,zero_fraction=float((use==0).mean()) if use.size else None,unavailable_cells=int((w==0).sum())))
        z=(a-stats['mean'].numpy()[None,None,:,None,None])/stats['std'].numpy()[None,None,:,None,None]
        pooled=(z*w).sum((1,3,4))/w.sum((1,3,4)).clip(1)
        st=latent(pooled)
        spaces.append(dict(stage='fixed_target_diagnostics',role=role,space='channel_mean6',dimensions=6,images=len(ids),within_component_centered=False,effective_rank=st['rank'],leading_fraction=st['leading'],scope='image_descriptors_not_learned_representation'))
    for f,records in [('target_channel_diagnostics',diagnostics),('target_space_diagnostics',spaces)]:
        path=table(root,f,records,protocol);files.append(path.relative_to(root).as_posix())
    for name in files:index['files'][name]=file_meta(root/name)
    inv=[dict(stage='fixed_target_input_inventory',logical_file=n,sha256=m['sha256'],bytes=m['bytes'],source_revision='1113bf8a9dd69687b23570fe985ed3a979a5f0e1',target_revision='trace-fixed-target-v2') for n,m in index['files'].items()]
    path=table(root,'target_input_inventory',inv,protocol);index['files'][path.relative_to(root).as_posix()]=file_meta(path)
    index['completed']=True;index['scientific_hash']=digest(index['files']);atomic(pointer,index)
    return Cache(root)

def sha_bytes(value):
    import hashlib
    a=value.detach().cpu().contiguous().numpy() if torch.is_tensor(value) else np.ascontiguousarray(value)
    return hashlib.sha256(a.tobytes()).hexdigest()

class Cache:
    def __init__(self,root):self.root=Path(root);self.index=read(self.root/'CACHE.json');self.rows=self.index['rows'];self.loaded={}
    def array(self,kind,ids):
        result=[]
        for slot in ids:
            number,offset=divmod(int(slot),8);key=(kind,number)
            if key not in self.loaded:
                if len(self.loaded)>=40:self.loaded.clear()
                self.loaded[key]=np.load(self.root/f'native/cache_{kind}/{number:05d}.npy',mmap_mode='r',allow_pickle=False)
            result.append(self.loaded[key][offset])
        return np.stack(result)
    def batch(self,ids,device):
        return {name:torch.from_numpy(self.array(name,ids)).to(device) for name in ('rgb','valid','target','weight')}
    def statistics(self,device='cpu'):
        return {k:v.to(device) for k,v in torch.load(self.root/'controls/target_statistics.pt',map_location='cpu',weights_only=True).items()}
    def verify(self):
        assert self.index['completed']
        for n,m in self.index['files'].items():assert file_meta(self.root/n)==m,n
        assert digest(self.index['files'])==self.index['scientific_hash']
