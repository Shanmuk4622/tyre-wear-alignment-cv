"""Pin only consumed already verified evidence; seed a local cache by verified copy."""
import json,shutil,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];GATE=ROOT/'tread_two/readiness_gate';sys.path.insert(0,str(GATE))
from public_inputs import sha,atomic
done=Path(json.loads((ROOT/'tread_two/nb14_completion/LATEST.json').read_text())['path']);v=json.loads((done/'VERIFICATION.json').read_text());assert v['status']=='passed'
old=ROOT/'data_gate_validation/nb07_publication';ov=json.loads((old/'VERIFICATION.json').read_text());assert ov['status']=='passed'
inputs={};sources={};cache=ROOT/'tread_two/readiness_validation/cache'
for tag,folder,revision,prefix,names in [
    ('NB14',done,v['revision'],v['prefix'],{'review_source.csv':'source_registry.csv','review_predictions.csv':'prediction_audit.csv','review_context.csv':'context_observations.csv'}),
    ('NB07',old,ov['revision'],ov['destination'],{'identity_images.csv':'csv/source_images.csv','identity_components.csv':'csv/leakage_components.csv','identity_tyres.csv':'csv/source_tyres.csv'})]:
    mp=folder/'REMOTE_MANIFEST.json' if tag=='NB14' else folder/'remote/REMOTE_MANIFEST.json'
    manifest=json.loads(mp.read_text());sources[tag]=dict(revision=revision,prefix=prefix,manifest_contract=manifest['contract'],manifest_local_receipt_sha256=sha(mp))
    for key,name in names.items():
        meta=manifest['files'][name];p=folder/'remote'/name;assert p.stat().st_size==meta['bytes'] and sha(p)==meta['sha256']
        inputs[key]=dict(meta,name=name,revision=revision,prefix=prefix)
        target=cache/key;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists() or sha(target)!=meta['sha256']:shutil.copy2(p,target)
spec=dict(kind='existing-data readiness and algebraic online correction utility',version=1,inputs=inputs,publications=sources,scope='Training rows only; no model fitting, inference, selection or workstation changes')
path=GATE/'INPUTS.json'
if path.exists():assert json.loads(path.read_text())==spec,'Preserve an already frozen input contract'
else:atomic(path,spec)
print(json.dumps(dict(input_files=len(inputs),bytes=sum(v['bytes'] for v in inputs.values()),local_cache=str(cache)),indent=2))
