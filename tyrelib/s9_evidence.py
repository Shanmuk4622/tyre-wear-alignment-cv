"""Exploratory fixed probability fusion; no training, selection or S9 completion claim."""
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

REPO = 'Shanmuk4622/tyre-wear-study'
SOURCE = 'a3b29a71f8e6af6c50e68eb64a5bbae9ccf6d1c5'
S5 = 's5/s5-manual-2026-09-10-r1/1f6694577253e0054f7a22df6ec52d30797063cf498b345bd71fe9b98bab93df'
ARMS = {'full': ['full'], 'tyre_only': ['pred_tyre'], 'tread_only': ['pred_tread'],
        'tyre_tread_equal': ['pred_tyre', 'pred_tread'],
        'full_tyre_tread_equal': ['full', 'pred_tyre', 'pred_tread']}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(probs, head_type):
    if head_type=='coral':
        # CORAL predicts the count of cumulative ordinal thresholds > 0.5,
        # not the most probable individual class.
        cumulative = np.stack((1-probs[:,0],probs[:,2]),axis=1)
        return (cumulative > .5).sum(1)
    assert head_type=='softmax', 'Unsupported classifier decision rule'
    return probs.argmax(1)


def analyse(frame, job, expected, labels, head_type='softmax'):
    columns = ['prob_low', 'prob_mid', 'prob_high']
    parts = {}
    for mode in ('full', 'pred_tyre', 'pred_tread'):
        x = frame.loc[frame['mode'].eq(mode)].sort_values('image_id').reset_index(drop=True)
        assert set(x.image_id)==set(expected) and x.image_id.is_unique, 'Incomplete/duplicate validation predictions'
        assert x.truth.tolist()==x.image_id.map(labels).tolist(), 'Labels differ from frozen manifest'
        p = x[columns].to_numpy(float)
        assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        assert np.allclose(p.sum(1),1,atol=1e-5), 'Probabilities must sum to one'
        assert np.array_equal(decode(p,head_type),x.prediction), f'Saved prediction differs from recorded {head_type} decision rule: {job["run_id"]}/{mode}'
        parts[mode]=x
    predictions, metrics = [], []
    full = parts['full']
    baseline = f1_score(full.truth,full.prediction,labels=[0,1,2],average='macro',zero_division=0)
    for arm, modes in ARMS.items():
        probs = np.mean([parts[m][columns].to_numpy(float) for m in modes],axis=0)
        pred = decode(probs,head_type)
        score = f1_score(full.truth,pred,labels=[0,1,2],average='macro',zero_division=0)
        metrics.append(dict(**job,decision_rule=head_type,arm=arm,n=len(full),macro_f1=float(score),
            accuracy=float(accuracy_score(full.truth,pred)),delta_vs_full=float(score-baseline)))
        for i,row in full.iterrows():
            predictions.append(dict(image_id=row.image_id,truth=int(row.truth),arm=arm,
                prediction=int(pred[i]),**{k:float(probs[i,j]) for j,k in enumerate(columns)}))
    return dict(job=job,decision_rule=head_type,metrics=metrics,predictions=predictions)


def run(sess):
    from email.utils import parsedate_to_datetime
    import tyrelib as tl
    token = sess.uploader.token
    assert token and sess.uploader.enabled, 'Enable writable HF_TOKEN and Internet'
    assert sess.num_workers==1, 'CPU analysis needs one copy, not four workers'
    code_hash = digest(Path(__file__).read_bytes())
    prefix = f's9/exploratory-fusion-r1/{code_hash}'
    root = Path(sess.stage_dir)/'s9_evidence'/code_hash
    root.mkdir(parents=True,exist_ok=True)
    def pull(path,revision=SOURCE,optional=False):
        for attempt in range(8):
            try:
                return Path(hf_hub_download(REPO,path,repo_type='dataset',revision=revision,
                    token=token,cache_dir=str(root/'cache'))).read_bytes()
            except EntryNotFoundError:
                if optional:return None
                raise
            except Exception as exc:
                response=getattr(exc,'response',None)
                if getattr(response,'status_code',None) not in (429,500,502,503,504) or attempt==7:raise
                delay=max(min(300,5*2**attempt),tl.parse_retry_after(str(exc)) or 0)
                hint=response.headers.get('Retry-After','')
                if hint:
                    try:delay=max(delay,float(hint))
                    except ValueError:
                        try:delay=max(delay,parsedate_to_datetime(hint).timestamp()-time.time())
                        except (ValueError,TypeError,OverflowError):pass
                print(f'HF read backoff {delay:.0f}s; progress remains resumable',flush=True)
                until=time.monotonic()+delay+2
                while time.monotonic()<until:time.sleep(min(5,max(0,until-time.monotonic())))
    def save(name,value):
        path=root/name
        raw=json.dumps(value,indent=2,allow_nan=False).encode()
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(raw);tmp.replace(path)
        sess.uploader.enqueue(path,prefix+'/'+name)
    def flush(reason):
        assert sess.push_now(reason), 'HF upload failed; retain session and retry'
    report=json.loads(pull(S5+'/report/STATUS.json'))
    assert report['status']=='complete' and report['verified_runs']==81
    plan=json.loads(pull(S5+'/protocol.json'))
    assert digest(json.dumps(plan,sort_keys=True,separators=(',',':')).encode())==S5.split('/')[-1]
    inventory=pd.read_csv(io.BytesIO(pull(S5+'/report/inventory.csv')))
    assert len(inventory)==81 and inventory.run_id.is_unique and inventory.status.eq('completed_verified').all()
    labels={x['image_id']:tl.C2I[x['proxy_label']] for x in plan['data']['records']}
    # Pin our resume read, too. No mutable result selection mid-run.
    from huggingface_hub import HfApi
    resume_rev=HfApi(token=token).repo_info(REPO,repo_type='dataset').sha
    completed=[]; configs={}; last=time.monotonic()
    try:
        for row in inventory.to_dict('records'):
            job={k:row[k] for k in ('run_id','model','backend','fold','seed')}
            name=job['run_id']+'.json'; path=root/name
            state=json.loads(pull(S5+'/runs/'+job['run_id']+'/STATUS.json'))
            assert state['job']==job and state['status']=='completed' and state['epoch']==60
            input_hash=state['artifact_sha256']['roi_predictions.csv']
            classifier=plan['downstream']['classifier'].format(**job)
            if classifier not in configs:
                import yaml
                raw_cfg=pull(f'runs/{classifier}/config.yaml',plan['source_revision'])
                cfg=yaml.safe_load(raw_cfg)
                configs[classifier]=('coral' if cfg['head_type']=='coral' else 'softmax',digest(raw_cfg))
            head_type,config_hash=configs[classifier]
            cached=path.read_bytes() if path.exists() else pull(prefix+'/'+name,resume_rev,optional=True)
            result=None
            if cached:
                result=json.loads(cached)
                assert result['source_revision']==SOURCE and result['input_sha256']==input_hash and result['code_sha256']==code_hash
                assert result['decision_rule']==head_type and result['classifier_config_sha256']==config_hash
                assert result['job']==job and result['payload_sha256']==digest(json.dumps(result['payload'],sort_keys=True).encode())
            else:
                raw=pull(S5+'/runs/'+job['run_id']+'/roi_predictions.csv');assert digest(raw)==input_hash
                payload=analyse(pd.read_csv(io.BytesIO(raw)),job,plan['data']['splits'][str(job['fold'])]['validation'],labels,head_type)
                result=dict(job=job,source_revision=SOURCE,input_sha256=input_hash,code_sha256=code_hash,
                    classifier=classifier,classifier_config_sha256=config_hash,decision_rule=head_type,
                    payload=payload,payload_sha256=digest(json.dumps(payload,sort_keys=True).encode()))
            save(name,result);completed.extend(result['payload']['metrics'])
            print(f'{len(completed)//5}/81 processed: {job["run_id"]}',flush=True)
            if time.monotonic()-last>=1800:flush('S9 analysis 30-minute progress');last=time.monotonic()
        frame=pd.DataFrame(completed)
        frame.to_csv(root/'fusion_by_run.csv',index=False)
        summary=frame.groupby(['model','fold','arm']).agg(n_seeds=('seed','nunique'),mean_f1=('macro_f1','mean'),
            mean_delta=('delta_vs_full','mean'),std_delta=('delta_vs_full','std')).reset_index()
        summary.to_csv(root/'fusion_summary.csv',index=False)
        for name in ('fusion_by_run.csv','fusion_summary.csv'):sess.uploader.enqueue(root/name,prefix+'/'+name)
        save('STATUS.json',dict(status='exploratory_analysis_complete',processed_runs=81,
            source_revision=SOURCE,code_sha256=code_hash,full_s9_complete=False,arms=ARMS,
            limitations=plan['limitations']+['Post-hoc fixed probability fusion, no fitted weights or independent model selection',
                'Same frozen Stage-A ResNet50 classifier; not a winning-backbone integrated pipeline'],
            missing=['HRNet landmark labels','Verified healthy reference pool for PatchCore',
                'Frozen full-pipeline specification and independent evaluation design'],
            artifact_sha256={n:digest((root/n).read_bytes()) for n in ('fusion_by_run.csv','fusion_summary.csv')}))
        sess.uploader.enqueue(Path(__file__),prefix+'/implementation.py')
        flush('S9 exploratory fusion analysis completed')
    except BaseException:
        flush('S9 analysis interrupted: preserve finished runs')
        raise
    print(summary.to_string(index=False))
    print('Analysis complete. Full S9 remains blocked on missing components; do not label this full pipeline completion.')
    return prefix
