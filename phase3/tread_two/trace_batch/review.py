"""CPU saved-evidence audit, independent selection and interactive residual passport."""
import argparse,base64,csv,hashlib,io,json,os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from common import HERE,atomic,read,digest,file_meta,sha
from cache import Cache,table
from metrics import evaluate

def review_contract(protocol):return dict(version='TRACE-review-v1',fit_protocol=protocol,sources=read(HERE/'FILES.json'),scope='CPU_saved_evidence_no_image_forward_no_fit')

class Saved:
    def __init__(self,root):
        self.root=Path(root);pointer=read(self.root/'POINTER.json');assert file_meta(self.root/pointer['state'])==pointer['state_meta']
        self.state=__import__('torch').load(self.root/pointer['state'],map_location='cpu',weights_only=False)
        self.closed=self.state['closed'];self.index=self.frame('native_chunks_index');self.mapped={}
    def paths(self,family):
        paths=[self.root/n for n in self.closed if n.startswith('csv/'+family+'/')]
        if family in self.state['csv']:paths.append(self.root/self.state['csv'][family]['path'])
        return sorted(paths)
    def frame(self,family):
        parts=[pd.read_csv(p,keep_default_na=False,float_precision='round_trip') for p in self.paths(family)]
        return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    def get(self,kind,ids):
        ids=np.asarray(ids,dtype='int64');result=None;seen=np.zeros(len(ids),dtype=bool)
        for row in self.index[self.index.kind==kind].itertuples():
            use=(ids>=row.first_observation)&(ids<=row.last_observation)
            if not use.any():continue
            if row.file not in self.mapped:
                if len(self.mapped)>=40:self.mapped.clear()
                self.mapped[row.file]=np.load(self.root/row.file,mmap_mode='r',allow_pickle=False)
            a=self.mapped[row.file]
            if result is None:result=np.empty((len(ids),*a.shape[1:]),dtype=a.dtype)
            assert not seen[use].any();result[use]=a[ids[use]-int(row.first_observation)];seen[use]=True
        assert seen.all(),(kind,ids[~seen].tolist());return result

def stage(out,name,protocol,fn):
    receipt=out/(name+'_RECEIPT.json')
    if receipt.exists():
        old=read(receipt);assert old['contract']==protocol
        assert all(file_meta(out/n)==m for n,m in old['files'].items()),'Completed review receipt changed'
        print('Verified completed TRACE review stage; skip:',name,flush=True);return
    fn()
    declared={'FILES':['FILE_AUDIT.csv','INPUT_PROVENANCE.json'],
              'SCIENCE':['VALIDATION_RECONSTRUCTION.csv','OBJECTIVE_RECONSTRUCTION.csv','csv/reconstruction_verification/00000.csv','SCIENTIFIC_AUDIT.json'],
              'VISUALS':['TRACE_Evidence_Passport.html','TRACE_Validation.png','MODEL_CARD.md','RESULT.json']}
    after=[out/n for n in declared[name]];assert all(p.is_file() for p in after)
    atomic(receipt,dict(contract=protocol,files={p.relative_to(out).as_posix():file_meta(p) for p in after}))
    files={}
    for path in out.glob('*_RECEIPT.json'):
        files.update(read(path)['files']);files[path.name]=file_meta(path)
    atomic(out/'MANIFEST.json',dict(contract=protocol,files=files))

def run(root,out,fit_protocol,fit_revision):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);saved=Saved(root);contract=review_contract(fit_protocol);protocol=digest(contract)
    cfg=read(saved.root/'CONTRACT.json');result=read(saved.root/'RESULT.json');assert result['status']=='completed' and result['protocol']==fit_protocol
    cache=Cache(saved.root/'inputs/cache');cache.verify()
    def inventory():
        rows=[]
        for i,(n,m) in enumerate(saved.closed.items()):
            assert file_meta(saved.root/n)==m,n
            if n.endswith('.npy'):
                a=np.load(saved.root/n,mmap_mode='r',allow_pickle=False);assert np.isfinite(a).all(),n
            rows.append(dict(file=n,bytes=m['bytes'],sha256=m['sha256'],verified=True))
            if i%200==0:print('TRACE review verified objects:',i,flush=True)
        pd.DataFrame(rows).to_csv(out/'FILE_AUDIT.csv',index=False)
        atomic(out/'INPUT_PROVENANCE.json',dict(protocol=fit_protocol,revision=fit_revision,objects=len(rows),selected=result['selected'],original_GEO_selected=None,scope='all retained files and native finite bodies'))
    stage(out,'FILES',protocol,inventory)
    observations=saved.frame('sample_observations').sort_values('observation');losses=saved.frame('image_losses');controls=saved.frame('control_losses')
    assert observations.observation.tolist()==list(range(result['observations']))
    source={r['image_id']:r for r in cache.rows}
    def scientific():
        checks=[];seen=np.zeros(len(observations)*2*32*4*4,dtype=bool);reduced={}
        stats=cache.statistics();mu=stats['mean'].numpy();std=stats['std'].numpy()
        for path in saved.paths('reconstruction_cells'):
            for frame in pd.read_csv(path,chunksize=65536,float_precision='round_trip'):
                for (target_id,view),g in frame.groupby(['target_id','view']):
                    slot_text,variant=target_id.split(':',1);slot=int(slot_text);raw=cache.array('target',[slot])[0];w=cache.array('weight',[slot])[0]
                    if variant in ('reverse_bands','reverse_columns'):
                        axis=0 if variant=='reverse_bands' else 1;raw=raw.reshape(8,4,6,4,4);w=w.reshape(8,4,1,4,4);raw=np.flip(raw,axis).reshape(32,6,4,4);w=np.flip(w,axis).reshape(32,1,4,4)
                    target=(raw-mu[None,:,None,None])/std[None,:,None,None]
                    ids=g.observation.to_numpy(dtype='int64');unique=np.unique(ids);t=g.tile.to_numpy(dtype='int64');y=g.cell_y.to_numpy(dtype='int64');x=g.cell_x.to_numpy(dtype='int64')
                    assert np.array_equal(g.valid_pixels.to_numpy(),w[t,0,y,x])
                    truth=target[t,:,y,x].astype('float64');weights=w[t,0,y,x].astype('float64');hidden=g.hidden.to_numpy(dtype=bool)
                    support=(cache.array('valid',[slot])[0].reshape(1,512,256).reshape(1,8,64,4,64).transpose(1,3,0,2,4).reshape(32,1,64,64).mean((1,2,3))>=.25)
                    if variant in ('reverse_bands','reverse_columns'):support=np.flip(support.reshape(8,4),0 if variant=='reverse_bands' else 1).reshape(32)
                    for kind,prefix in [('local_prediction','local'),('masked_prediction','masked')]:
                        values=g[[f'{prefix}_{k}' for k in range(6)]].to_numpy(dtype='float32')
                        native=saved.get(kind,unique);lookup={obs:i for i,obs in enumerate(unique)};offset=np.array([lookup[a] for a in ids])
                        assert np.array_equal(values,native[offset,int(view),t,:,y,x]),(kind,target_id,'CSV/native mismatch')
                        delta=np.abs(values.astype('float64')-truth);huber=np.where(delta<.1,.5*delta*delta/.1,delta-.05)
                        use=weights*(support[t] if prefix=='local' else hidden)
                        nums=(huber*use[:,None]).sum(1);counts=use*6
                        for obs in unique:
                            mask=ids==obs;key=(int(obs),int(view),prefix);a=reduced.setdefault(key,[0.,0.]);a[0]+=float(nums[mask].sum());a[1]+=float(counts[mask].sum())
                    positions=((((ids*2+int(view))*32+t)*4+y)*4+x)
                    assert len(np.unique(positions))==len(positions) and not seen[positions].any();seen[positions]=True
            print('TRACE review reconstructed:',path.name,flush=True)
        assert seen.all()
        for row in losses[losses.objective.isin(['local','masked'])].itertuples():
            value,count=reduced[(int(row.observation),int(row.view),row.objective)]
            assert count==row.weighted_channel_count
            assert np.isclose(value,row.huber_sum,atol=.02,rtol=2e-5),(row.observation,row.objective,value,row.huber_sum)
        # All moment CSV values also round-trip exactly to their native body.
        names={'mean':0,'row':1,'column':2,'row_column':3,'row_squared':4};moment_count=0
        for path in saved.paths('spatial_moment_vectors'):
            for frame in pd.read_csv(path,chunksize=131072,float_precision='round_trip'):
                if not len(frame):continue
                ids=frame.observation.to_numpy(dtype='int64');unique=np.unique(ids);native=saved.get('pooled',unique);lookup={v:i for i,v in enumerate(unique)}
                a=np.array([lookup[v] for v in ids]);d=frame.moment.map(names).to_numpy()*48+frame.channel.to_numpy()
                assert np.array_equal(frame.value.to_numpy(dtype='float32'),native[a,frame.view.to_numpy(dtype='int64'),d]);moment_count+=len(frame)
        assert moment_count==len(observations)*2*240
        checks.append(dict(stage='independent_review',artifact='reconstruction_cells',grain='every source/view/tile/cell/channel',compared_values=len(seen)*12,max_absolute_difference=0.,tolerance='FP32 CSV/native exact; raw Huber sums atol0.02 rtol2e-5',passed=True))
        objectives=saved.frame('objective_terms');train_rows=observations[observations.stage=='train'];objective_checks=[]
        for update,frame in train_rows.groupby('update'):
            ids=frame.observation.to_numpy(dtype='int64');z=saved.get('projected',ids).astype('float64');local=[];masked=[];variance=[];covariance=[];ok=[]
            for view in (0,1):
                q=losses[(losses.observation.isin(ids))&(losses.view==view)]
                a=q[(q.objective=='local')&(q.available==1)];b=q[(q.objective=='masked')&(q.eligible==1)]
                local.append(float(a.image_mean.astype(float).mean()) if len(a) else 0.);masked.append(float(b.image_mean.astype(float).mean()) if len(b) else 0.)
                use=np.isin(ids,b.observation.to_numpy(dtype='int64'));ok.append(use);value=z[use,view]
                if len(value)>=7:
                    c=np.cov(value,rowvar=False,ddof=1);variance.append(float(np.maximum(0,.1-np.sqrt(np.diag(c)+1e-4)).mean()));off=c-np.diag(np.diag(c));covariance.append(float((off**2).sum()/30))
                else:variance.append(0.);covariance.append(0.)
            paired=ok[0]&ok[1];pair=float(((z[paired,0]-z[paired,1])**2).mean()) if paired.any() else 0.
            terms=dict(local=float(np.mean(local)),masked=float(np.mean(masked)),pair=pair,variance=float(np.mean(variance)),covariance=float(np.mean(covariance)))
            terms['total']=terms['local']+terms['masked']+.1*pair+.1*terms['variance']+.01*terms['covariance']
            original=objectives[objectives['update']==update].iloc[0]
            for name,value in terms.items():assert np.isclose(value,float(original[name]),atol=3e-6,rtol=3e-5),(update,name,value,original[name])
            objective_checks.append(dict(update=int(update),**terms,max_absolute_difference=max(abs(v-float(original[k])) for k,v in terms.items())))
        pd.DataFrame(objective_checks).to_csv(out/'OBJECTIVE_RECONSTRUCTION.csv',index=False)
        # Rebuild each validation eligibility and original selected epoch using
        # independently read per-image controls, losses and native vectors.
        original=saved.frame('eligibility');epochs=[]
        train=observations[observations.stage=='train']
        for epoch,frame in observations[observations.stage=='validate'].groupby('epoch'):
            ids=frame.observation.to_numpy(dtype='int64');p=saved.get('projected',ids);local=saved.get('local_pooled',ids);z=saved.get('pooled',ids);ctx=saved.get('context_mean',ids)
            samples=[]
            for i,row in enumerate(frame.itertuples()):
                l=losses[(losses.observation==row.observation)&(losses.view==0)].set_index('objective');c=controls[(controls.image_id==row.image_id)&(controls.epoch==epoch)&(controls.stage=='validate')&(controls.view==0)].set_index('control')
                row_source=source[row.image_id];den=np.linalg.norm(p[i,0])*np.linalg.norm(p[i,1]);native_count=float(l.loc['masked','weighted_channel_count']);common=float(c.iloc[0].common_count)
                verified_cosine=float(1-p[i,0].astype('float64')@p[i,1].astype('float64')/den) if den>=1e-8 else None
                recorded_cosine=float(row.paired_cosine) if row.paired_cosine!='' else None
                assert recorded_cosine is None and verified_cosine is None or recorded_cosine is not None and np.isclose(recorded_cosine,verified_cosine,atol=2e-6,rtol=2e-5)
                sample=dict(component=row_source['component'],eligible=bool(row.eligible),local_count=float(l.loc['local','weighted_channel_count']),
                    local_loss=float(l.loc['local','image_mean']) if l.loc['local','image_mean']!='' else None,local_control=float(l.loc['local_position_control','image_mean']) if l.loc['local_position_control','image_mean']!='' else None,
                    model_common=float(c.iloc[0].model_mean) if common else None,coverage=common/native_count if native_count else None,cosine=recorded_cosine,
                    projected=p[i,0],local_pooled=local[i,0],pooled=z[i,0],context_mean=ctx[i,0])
                sample.update({'control_'+name:float(c.loc[name,'control_mean']) if common else None for name in ['channel_mean','position_mean','visible_band_mean']});samples.append(sample)
            primary=train[(train.epoch==epoch)&(train.primary==1)];fraction=float(primary.eligible.mean())
            summary=evaluate(samples,cfg,fraction);old=original[(original.stage=='validate')&(original.epoch==epoch)].iloc[0]
            assert bool(old.eligible)==summary['eligible']
            for name,value in summary['gates'].items():assert bool(old[name])==value,(epoch,name)
            if summary['eligible']:assert np.isclose(float(old.objective),summary['score'],atol=2e-6,rtol=2e-5)
            epochs.append(dict(epoch=int(epoch),eligible=summary['eligible'],score=summary['score'],rejection=summary['rejection'],projector_rank=summary['spectra']['projected']['rank'],local_rank=summary['spectra']['local_pooled']['rank'],spatial_rank=summary['spectra']['pooled']['rank'],masked_loss=summary['model'],best_control=min((v for v in summary['controls'].values() if v is not None),default=None),local_loss=summary['local']))
        eligible=[e for e in epochs if e['eligible']];chosen=min(eligible,key=lambda e:(e['score'],e['epoch']))['epoch'] if eligible else None
        assert chosen==(result['selected']['epoch'] if result['selected'] else None)
        pd.DataFrame(epochs).to_csv(out/'VALIDATION_RECONSTRUCTION.csv',index=False)
        table(out,'reconstruction_verification',checks,protocol)
        atomic(out/'SCIENTIFIC_AUDIT.json',dict(status='passed',all_prediction_values_CSV_native_exact=len(seen)*12,all_moment_values_CSV_native_exact=moment_count,all_raw_local_masked_losses_reconstructed=True,all_training_objectives_reconstructed=len(objective_checks),selection_reconstructed=True,selected_epoch=chosen,optimizer_updates=0,image_model_forwards=0))
    stage(out,'SCIENCE',protocol,scientific)
    def visuals():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        final=observations[observations.stage=='endpoint_final'];cases=[]
        for row in final.itertuples():
            slot=source[row.image_id]['slot'];image=cache.array('rgb',[slot])[0].transpose(1,2,0)
            buffer=io.BytesIO();Image.fromarray((image.clip(0,1)*255).astype('uint8')).resize((96,192)).save(buffer,format='PNG')
            target=cache.array('target',[slot])[0];stats=cache.statistics();target=(target-stats['mean'].numpy()[None,:,None,None])/stats['std'].numpy()[None,:,None,None]
            prediction=saved.get('local_prediction',[row.observation])[0,0];masked=saved.get('masked_prediction',[row.observation])[0,0]
            raw_residual=np.abs(prediction-target).mean(1);weight=cache.array('weight',[slot])[0,:,0]
            residual=raw_residual.reshape(8,4,4,4).transpose(0,2,1,3).reshape(32,16)
            available=(weight>0).reshape(8,4,4,4).transpose(0,2,1,3).reshape(32,16)
            cases.append(dict(image_id=row.image_id,role=source[row.image_id]['role'],component=source[row.image_id]['component'],eligible=bool(row.eligible),image='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode(),residual=np.where(available,residual,0).tolist(),available=available.tolist(),mean_residual=float((raw_residual*weight).sum()/weight.sum()) if weight.sum() else None))
        payload=json.dumps(cases,separators=(',',':')).replace('<','\\u003c')
        html=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>TRACE residual atlas</title><style>body{font-family:system-ui;background:#edf3f4;color:#173c48;margin:auto;max-width:1100px;padding:24px}h1{font-size:32px}select,input,button{padding:9px;margin:6px;border:1px solid #bcc;border-radius:8px}#gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}article{background:white;padding:14px;border-radius:12px;overflow-wrap:anywhere}img,canvas{width:96px;height:192px}small{font-size:11px}</style><h1>TREAD-TWO / TRACE evidence passport</h1><p id="status"></p><p>Image-derived standardised reconstruction residuals; not measured wear, depth, damage or calibrated confidence. Existing, previously inspected cohorts.</p><select id="role"><option value="all">All roles</option><option>train</option><option>validation</option><option>test</option></select><input id="search" placeholder="Search source ID"><button id="previous">Previous</button><button id="next">Next</button><p id="count"></p><div id="gallery"></div><script type="application/json" id="data">__DATA__</script><script>
const cases=JSON.parse(document.getElementById('data').textContent);let page=0;
const el=id=>document.getElementById(id);
function render(){
 const rows=cases.filter(c=>(el('role').value==='all'||c.role===el('role').value)&&c.image_id.includes(el('search').value));
 page=Math.min(page,Math.max(0,Math.ceil(rows.length/8)-1));el('count').textContent=rows.length+' records / page '+(page+1);el('gallery').replaceChildren();
 for(const c of rows.slice(page*8,page*8+8)){
  const a=document.createElement('article'),h=document.createElement('small'),img=document.createElement('img'),canvas=document.createElement('canvas'),p=document.createElement('p');
  h.textContent=c.image_id+' / '+c.role;img.src=c.image;img.alt='Existing canonical tyre';canvas.width=16;canvas.height=32;const ctx=canvas.getContext('2d');
  for(let y=0;y<32;y++)for(let x=0;x<16;x++){
   const t=Math.min(1,c.residual[y][x]/2);ctx.fillStyle=c.available[y][x]?'rgb('+Math.round(30+220*t)+','+Math.round(100*(1-t))+',80)':'#fff';ctx.fillRect(x,y,1,1);
  }
  p.textContent='Weighted descriptor residual '+(c.mean_residual===null?'unavailable':c.mean_residual.toFixed(4));a.append(h,document.createElement('br'),img,canvas,p);el('gallery').append(a);
 }
}
el('role').addEventListener('change',()=>{page=0;render()});el('search').addEventListener('input',()=>{page=0;render()});
el('next').onclick=()=>{page++;render()};el('previous').onclick=()=>{page=Math.max(0,page-1);render()};
el('status').textContent='Research only. Original GEO selection remains unavailable. TRACE selected epoch: __SELECTED__';render();</script></html>'''
        (out/'TRACE_Evidence_Passport.html').write_text(html.replace('__DATA__',payload).replace('__SELECTED__',str(result['selected']['epoch']) if result['selected'] else 'unavailable'),encoding='utf-8')
        df=pd.read_csv(out/'VALIDATION_RECONSTRUCTION.csv');fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        axes[0].plot(df.epoch,df.masked_loss,label='TRACE');axes[0].plot(df.epoch,df.best_control,label='Strongest common-support control');axes[0].set(title='Validation fixed-target reconstruction',xlabel='Epoch',ylabel='Image-mean Huber');axes[0].legend()
        for key in ['projector_rank','local_rank','spatial_rank']:axes[1].plot(df.epoch,df[key],label=key)
        axes[1].axhline(4,color='gray',ls='--');axes[1].set(title='Diversity across declared spaces',xlabel='Epoch',ylabel='Effective rank');axes[1].legend();fig.savefig(out/'TRACE_Validation.png',dpi=150);plt.close(fig)
        (out/'MODEL_CARD.md').write_text('# TREAD-TWO / TRACE research passport\n\nOne existing-data texture candidate; no physical-depth, safety or improved mileage-classification claim.\n\nSelected: '+(str(result['selected']) if result['selected'] else 'unavailable; all checkpoints rejected by the original rule')+'\n\nInherited geometry: GEO final epoch60, unselected research input. Original GEO selected=None is preserved.\n\nOriginal roles:386train/81validation/103test, already inspected; no independent new test.\n\nFull training evidence is direct CSV/native at fit revision '+fit_revision+'.\n',encoding='utf-8')
        atomic(out/'RESULT.json',dict(status='completed',fit_protocol=fit_protocol,fit_revision=fit_revision,selected=result['selected'],scientific_status='research_only_unpromoted',existing_photos_only=True,new_optimizer_updates=0))
    stage(out,'VISUALS',protocol,visuals)
    return read(out/'RESULT.json')

if __name__=='__main__':
    assert not os.environ.get('HF_TOKEN')
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--fit-protocol',required=True);p.add_argument('--fit-revision',required=True);a=p.parse_args()
    __import__('torch').set_num_threads(4);run(a.input,a.output,a.fit_protocol,a.fit_revision)
