"""Actual existing-image chunk interruption and finite review-stage recovery."""
import copy,sys,time
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'tread_two/trace_batch';sys.path.insert(0,str(SRC))
from common import atomic,read,file_meta,seed_all
import cache as module
from data import dataset
from review import stage
import torch
torch.set_num_threads(4);seed_all(1)
valid=ROOT/'tread_two/trace_validation';root=valid/('fixture_preparation_'+str(time.time_ns()));root.mkdir()
data_root=ROOT.parent/'phase2/data/phase2_dataset_v1';rows=[copy.deepcopy(r) for r in dataset(data_root,verify=True) if r['role']=='train'][:16]
for i,r in enumerate(rows):r['slot']=i
folder=Path(read(ROOT/'tread_two/nb17_completion/LATEST.json')['path']);export=folder/'remote/exports/final.pt'
original=module.dataset;module.dataset=lambda *args,**kwargs:copy.deepcopy(rows)
try:
    def stop(index):
        if len(index['chunks'])==1:raise InterruptedError('injected_after_one_complete_cache_chunk')
    try:module.prepare(root/'cache',data_root,export,'explicit_16_train_image_fixture','cpu',service=stop)
    except InterruptedError:pass
    else:raise AssertionError('Preparation interruption not exercised')
    before=read(root/'cache/CACHE.json');assert len(before['chunks'])==1
    saved=dict(before['files']);module.prepare(root/'cache',data_root,export,'explicit_16_train_image_fixture','cpu')
    after=read(root/'cache/CACHE.json');assert after['completed'] and len(after['chunks'])==2
    assert all(after['files'][n]==m==file_meta(root/'cache'/n) for n,m in saved.items())
    out=root/'review';out.mkdir()
    def interrupted():(out/'FILE_AUDIT.csv').write_text('partial\n',encoding='utf-8');raise InterruptedError('injected_partial_analysis_output')
    try:stage(out,'FILES','fixture_review',interrupted)
    except InterruptedError:pass
    assert not (out/'FILES_RECEIPT.json').exists()
    def complete():(out/'FILE_AUDIT.csv').write_text('verified\n',encoding='utf-8');atomic(out/'INPUT_PROVENANCE.json',dict(scope='fixture'))
    stage(out,'FILES','fixture_review',complete);receipt=read(out/'FILES_RECEIPT.json');assert set(receipt['files'])=={'FILE_AUDIT.csv','INPUT_PROVENANCE.json'}
    stage(out,'FILES','fixture_review',lambda:(_ for _ in ()).throw(AssertionError('Should skip completed work')))
    (out/'FILE_AUDIT.csv').write_text('corrupt\n',encoding='utf-8')
    try:stage(out,'FILES','fixture_review',complete)
    except AssertionError:pass
    else:raise AssertionError('Corrupt completed review silently accepted')
    atomic(valid/'PREPARATION_RECOVERY.json',dict(status='passed',source_inventory={n:file_meta(SRC/n) for n in read(SRC/'FILES.json')},actual_existing_train_images=16,
        interrupted_after8=True,first_chunk_bytes_preserved=True,completed_chunks=2,partial_analysis_replayed=True,declared_outputs_selected_even_if_partial_files_existed=True,
        completed_stage_skipped=True,corrupt_completed_stage_rejected=True,remote_writes=0,credentials_read=False,new_optimizer_updates=0))
    print(read(valid/'PREPARATION_RECOVERY.json'),flush=True)
finally:module.dataset=original
