"""Finite CPU review with verified stage receipts; scientific rejection is a valid result."""
import os,time
from pathlib import Path
import numpy as np
import pandas as pd
from common import atomic,digest,file_meta,read
from inputs import Saved,PIN,locate,ground_truth
import audit_core,audit_losses,audit_native,audit_files

ARTIFACTS={
    'files':['source_file_inventory.csv','source_CSV_inventory.csv','source_family_inventory.csv','FILE_VERIFICATION.json'],
    'core':[n+'.csv' for n in ['observation_coverage','source_registry','quality_comparison','quality_calibration','case_metrics','validation_reconstruction','validation_spectra','selection_source_comparison','endpoint_latent_spaces','endpoint_cohorts','training_exposure_audit','training_task_gradients','tensor_gradient_summary','resource_summary','timing_summary','intervention_summary','intervention_cases','retrieval_reconstruction','retrieval_audit']]+['projected.npy','global.npy','FINDINGS.json'],
    'losses':['loss_reconstruction.csv'],
    'native':[n+'.csv' for n in ['native_inventory','raw_geometry_audit','point_order_audit','native_vector_pooling_audit','epoch_parameter_audit','component_variance_decomposition','endpoint_token_spaces','endpoint_patch_latent']]+['NATIVE_VERIFICATION.json'],
    'masked':['raw_masked_reconstruction.csv'],
    'dense':['dense_confusion_audit.csv'],
    'visuals':['figures/'+n+'.png' for n in ['GEO_Validation','GEO_Representation_And_Probes','GEO_Quality_And_Gradients']]+['GEO_Review_Atlas.html','VISUALS.json']}

def load_core(saved,out):
    from tables import Tables
    tables=Tables(out)
    return dict(tables=tables,obs=saved.table('sample_observations'),source=saved.table('source_images'),case=pd.read_csv(out/'case_metrics.csv'),
                regions=saved.table('region_metrics'),points=saved.table('point_values'),quality=saved.table('quality_predictions'),geometry=saved.table('geometry_losses'),
                patches=saved.table('masked_patch_metrics'),projected=np.load(out/'projected.npy',mmap_mode='r'),global_z=np.load(out/'global.npy',mmap_mode='r'),findings=read(out/'FINDINGS.json'))

class Stages:
    def __init__(self,out,contract):
        self.out=Path(out);self.receipts=self.out/'stages';self.receipts.mkdir(parents=True,exist_ok=True);self.contract=contract
    def run(self,name,fn):
        receipt=self.receipts/(name+'.json')
        if receipt.exists():
            old=read(receipt);assert old['contract']==self.contract
            if all((self.out/p).exists() and file_meta(self.out/p)==meta for p,meta in old['files'].items()):
                print('Verified completed stage; skip:',name,flush=True);return old['result']
        atomic(self.out/'STATUS.json',dict(stage=name,status='running',started=time.time(),contract=self.contract))
        start=time.monotonic();result=fn()
        if name in ARTIFACTS:
            assert all((self.out/p).is_file() for p in ARTIFACTS[name]),('Missing completed stage output',name)
            after={p:file_meta(self.out/p) for p in ARTIFACTS[name]}
        else:after={p.relative_to(self.out).as_posix():file_meta(p) for p in self.out.rglob('*') if p.is_file() and 'stages' not in p.parts and p.name not in ('STATUS.json','TABLES.json','NATIVE_PROGRESS.json','ENVIRONMENT.json','ERROR.json','RESULT.json') and not p.name.endswith(('.tmp','.partial.csv'))}
        # Bind every declared output, including identical files left by an
        # interruption. Never bind an unfinished later stage's artifacts.
        atomic(receipt,dict(contract=self.contract,stage=name,seconds=time.monotonic()-start,files=after,result=result))
        atomic(self.out/'STATUS.json',dict(stage=name,status='completed',contract=self.contract))
        return result

def review(root,manifest,out,contract,dataset=''):
    out=Path(out);saved=Saved(root,manifest);stages=Stages(out,contract)
    stages.run('files',lambda:audit_files.run(saved,out))
    core_holder={}
    def core_step():
        core_holder['value']=audit_core.run(saved,out);return core_holder['value']['findings']
    stages.run('core',core_step)
    core=core_holder.get('value') or load_core(saved,out)
    stages.run('losses',lambda:audit_losses.run(saved,core))
    truth=ground_truth(locate(dataset),core['source'])
    stages.run('native',lambda:audit_native.run(saved,core,truth)[1])
    index=saved.table('native_chunks_index')
    stages.run('masked',lambda:audit_losses.raw_masked(saved,core,index))
    stages.run('dense',lambda:audit_native.dense(saved,core,truth))
    from render_review import render
    stages.run('visuals',lambda:render(saved,core))
    result=dict(status='completed',scientific_status='research_only_unselected',input=PIN,stages=['files','core','losses','native','masked','dense','visuals'],
                findings=core['findings'],new_optimizer_updates=0,image_model_forwards=0,new_photos=0)
    atomic(out/'RESULT.json',result);return result
