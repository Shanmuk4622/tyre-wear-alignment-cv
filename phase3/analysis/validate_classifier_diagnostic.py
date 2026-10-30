"""Verify completed diagnostic data, native maps, notebook sources and safe report.

This is a scientific/source validation. It does not claim a browser render pass.
"""
from PIL import Image
import ast
import base64
import csv
import gzip
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter,defaultdict
from pathlib import Path

assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'diagnostic'))
from storage import read,sha,digest,atomic
import numpy as np
import nbformat


def main():
    complete=[]
    for p in (ROOT/'diagnostics/classifier_r1').glob('*/COMPLETE.json'):
        contract=read(p.parent/'CONTRACT.json')
        if not contract['fixture']: complete.append(p)
    assert complete,'Full diagnostic is not yet complete.'
    path=max(complete,key=lambda p:p.stat().st_mtime);done=read(path);folder=path.parent;snapshot=Path(done['snapshot'])
    contract=read(snapshot/'CONTRACT.json');assert digest(contract)==done['contract']
    assert len(contract['rows'])==467 and len(contract['jobs'])==6
    assert not contract['fixture'] and done['training_updates']==0
    for name,expected in contract['sources'].items(): assert sha(ROOT/name)==expected,name
    for name,expected in contract['frozen_sources'].items(): assert sha(ROOT/'runtime'/name)==expected,name
    manifest=read(snapshot/'MANIFEST.json')
    for name,meta in manifest['files'].items():
        assert sha(snapshot/name)==meta['sha256'] and (snapshot/name).stat().st_size==meta['bytes'],name
    overlay=read(ROOT/'completion_20261028/hf/OVERLAY.json');source={r['image_id']:r for r in overlay['rows']}
    with open(snapshot/'csv/predictions.csv',encoding='utf-8',newline='') as stream: predictions=list(csv.DictReader(stream))
    assert len(predictions)==467*6*5==14010
    keys=set();job_support=Counter();normal={}
    for row in predictions:
        original=source[row['image_id']]
        assert row['split']==original['role'] and row['split'] in ('train','validation')
        assert row['image_sha256']==original['image_sha256'] and int(row['label'])==original['class_index']
        key=(row['job'],row['image_id'],row['variant']);assert key not in keys;keys.add(key)
        p=np.array([float(row[f'probability_{c}']) for c in range(3)])
        assert np.isfinite(p).all() and (p>=0).all() and abs(p.sum()-1)<1e-6
        assert int(p.argmax())==int(row['prediction'])
        assert abs(float(row['nll'])+math.log(max(p[int(row['label'])],1e-12)))<1e-10
        assert abs(float(row['brier'])-np.square(p-np.eye(3)[int(row['label'])]).sum())<1e-10
        if row['variant']=='normal': normal[(row['job'],row['image_id'])]=row
        job_support[(row['job'],row['split'],row['variant'])]+=1
        assert float(row['evidence_residual'])<1e-4
    assert set(job_support.values())=={386,81}
    for row in predictions:
        old=normal[(row['job'],row['image_id'])]
        delta=sum(abs(float(row[f'probability_{c}'])-float(old[f'probability_{c}'])) for c in range(3))
        assert abs(delta-float(row['probability_l1_delta']))<1e-6
        if row['variant'] in ('uniform_gate','oracle_tread_gate'):
            assert row['mean_two_region_dice']==old['mean_two_region_dice'] and row['human_point_mae']==old['human_point_mae']
    counts={}
    for table in contract['telemetry']:
        with open(snapshot/'csv'/(table+'.csv'),encoding='utf-8',newline='') as stream:
            reader=csv.DictReader(stream);assert reader.fieldnames==contract['telemetry'][table]
            counts[table]=sum(1 for _ in reader)
    assert counts['points']==84060 and counts['regions']==28020 and counts['latents']==168120 and counts['evidence']==42030
    region_dice=defaultdict(list);point_errors=defaultdict(list)
    with open(snapshot/'csv/regions.csv',encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            tp,fp,fn,tn=[int(row[k]) for k in ('tp','fp','fn','tn')]
            assert tp+fp+fn+tn==int(row['valid_pixels'])
            assert abs(float(row['dice'])-(2*tp+1)/(2*tp+fp+fn+1))<1e-12
            region_dice[(row['job'],row['image_id'],row['variant'])].append(float(row['dice']))
    human_points=0;max_point_rounding=0.;max_native_point_rounding=0.
    with open(snapshot/'csv/points.csv',encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            original=source[row['image_id']];k=int(row['point_index'])
            assert row['target_kind']==original['point_kind']
            if row['target']:
                truth=original['points'][k];assert abs(float(row['target'])-truth)<1e-12
                error=abs(float(row['prediction'])-truth)
                # The saved per-point scalar subtraction is FP32 (NumPy scalar
                # promotion); independently reproduce that arithmetic exactly.
                expected=float(np.float32(np.float32(float(row['prediction']))-np.float32(truth)))
                assert abs(float(row['signed_error_width_fraction'])-expected)<1e-12
                stored=abs(expected)
                assert abs(float(row['absolute_error_width_fraction'])-stored)<1e-12
                assert abs(float(row['absolute_error_native_pixels'])-stored*(original['width']-1))<1e-9
                rounding=abs(stored-error);assert rounding<=2*np.finfo(np.float32).eps
                max_point_rounding=max(max_point_rounding,rounding)
                max_native_point_rounding=max(max_native_point_rounding,rounding*(original['width']-1))
                if row['human_eligible']=='1': point_errors[(row['job'],row['image_id'],row['variant'])].append(error)
            human_points+=int(row['human_eligible'])
    cohort=defaultdict(list)
    for row in predictions:
        key=(row['job'],row['image_id'],row['variant'])
        assert len(region_dice[key])==2 and abs(np.mean(region_dice[key])-float(row['mean_two_region_dice']))<1e-12
        if row['human_point_mae']:
            assert len(point_errors[key])==6 and abs(np.mean(point_errors[key])-float(row['human_point_mae']))<1e-12
        cohort[(row['job'],row['split'],row['variant'])].append(row)
    with open(snapshot/'summary/cohort_group_metrics.csv',encoding='utf-8',newline='') as stream:
        checked_cohorts=0
        for row in csv.DictReader(stream):
            if row['level']!='cohort': continue
            members=cohort[(row['job'],row['split'],row['variant'])]
            labels=np.array([int(r['label']) for r in members]);pred=np.array([int(r['prediction']) for r in members])
            confusion=np.bincount(3*labels+pred,minlength=9).reshape(3,3)
            supports=confusion.sum(1);observed=supports>0
            ba=float(np.mean(np.diag(confusion)[observed]/supports[observed]))
            assert len(members)==int(row['images']) and abs(ba-float(row['balanced_accuracy_observed']))<1e-12
            checked_cohorts+=1
    assert checked_cohorts==60
    with open(snapshot/'csv/gradients.csv',encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['cosine']: assert math.isfinite(float(row['cosine'])) and abs(float(row['cosine']))<=1.000001
            if row['gradient_norm']: assert math.isfinite(float(row['gradient_norm'])) and float(row['gradient_norm'])>=0
    for job in contract['jobs']:
        identity=read(folder/(job['key']+'-IDENTITY.json'))
        assert identity['tensor_sha256']==job['tensor_sha256'] and identity['unchanged_after_inference_and_gradients']
    gallery=read(snapshot/'gallery/METADATA.json')['images'];arrays=np.load(snapshot/'gallery/arrays.npy',mmap_mode='r',allow_pickle=False)
    assert arrays.shape==(len(gallery),3,19,128,96) and arrays.dtype==np.float32
    assert np.isfinite(arrays).all()
    report=snapshot/'TREAD_ONE_Classifier_Diagnostic.html';text=report.read_text(encoding='utf-8')
    start=text.index('const DATA=')+len('const DATA=');end=text.index('const $',start)
    payload=json.loads(text[start:end].strip().removesuffix(';'))
    assert len(payload['rows'])==14010 and len(payload['gallery'])==len(gallery)
    for item in payload['gallery']:
        assert set(item['variants'])==set(contract['variants'])
        for variant,p in item['variants'].items():
            assert len(p['layers'])==6 and all(len(base64.b64decode(v))==48*64 for v in p['layers'])
            assert p['image'].startswith('data:image/jpeg;base64,')
            with Image.open(io.BytesIO(base64.b64decode(p['image'].split(',',1)[1]))) as im:
                assert im.size==(384,512)
    assert "connect-src 'none'" in text and not re.search(r'<script[^>]+src=',text,re.I)
    validation=ROOT/'diagnostics/source_validation';validation.mkdir(exist_ok=True)
    script=validation/'explorer-script.js'
    script.write_text(text.split('<script>',1)[1].rsplit('</script>',1)[0],encoding='utf-8')
    checked=subprocess.run(['node','--check',str(script)],capture_output=True,text=True)
    assert checked.returncode==0,checked.stderr
    notebook=nbformat.read(ROOT/'notebooks/phase3_NB03_TREAD_ONE_Classifier_Diagnostic.ipynb',as_version=4);nbformat.validate(notebook)
    embedded={}
    for cell in notebook.cells:
        if cell.cell_type!='code': continue
        for call in ast.walk(ast.parse(cell.source)):
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='b64decode' and call.args and isinstance(call.args[0],ast.Constant):
                for name,value in json.loads(gzip.decompress(base64.b64decode(call.args[0].value))).items():
                    assert base64.b64decode(value)==(ROOT/name).read_bytes(),name
                    embedded[name]=hashlib.sha256(base64.b64decode(value)).hexdigest()
    assert len(embedded)==15
    protected=read(ROOT/'audit/markdown-baseline.json')
    assert all(sha(ROOT.parent/r['path'])==r['sha256'] for r in protected)
    result=dict(status='passed',diagnostic_contract=done['contract'],snapshot=str(snapshot),
        images_per_checkpoint=467,checkpoint_jobs=6,variants=5,csv_rows=counts,human_point_rows=human_points,
        point_residual_arithmetic='FP32 subtraction reproduced exactly; float64 reconstruction discrepancy bounded by two FP32 epsilons.',
        cohort_summaries_independently_recalculated=checked_cohorts,per_image_region_and_human_point_aggregates_verified=True,
        max_point_residual_float64_difference=max_point_rounding,max_native_point_residual_float64_difference_pixels=max_native_point_rounding,
        no_test_images=True,checkpoint_parameters_unchanged=True,gallery_images=len(gallery),raw_gallery_shape=list(arrays.shape),
        html_bytes=report.stat().st_size,html_javascript_syntax='passed',html_data_and_image_checks='passed',
        browser_render_tested=False,browser_preview_limitation='Local-file browser access was rejected by browser security policy. No workaround attempted.',
        embedded_sources=len(embedded),protected_original_markdown=len(protected),training_updates=0,
        hf_diagnostic_publication='not_attempted_locally; use NB03 Kaggle HF_TOKEN',kaggle_nb03_execution_tested=False)
    alias=ROOT/'diagnostics/TREAD_ONE_Classifier_Explorer.html'
    shutil.copy2(report,alias)
    assert sha(alias)==sha(report)
    result['explorer_alias']=str(alias)
    result['explorer_sha256']=sha(report)
    atomic(ROOT/'diagnostics/DIAGNOSTIC_COMPLETION.json',dict(status='complete_locally',contract=done['contract'],
        snapshot=str(snapshot),explorer=str(alias),explorer_sha256=sha(report),training_updates=0,
        hf_publication='pending_Kaggle_NB03',source_revision=contract['source_revision'],source_protocol=contract['source_protocol']))
    atomic(ROOT/'diagnostics/DIAGNOSTIC_VALIDATION.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
