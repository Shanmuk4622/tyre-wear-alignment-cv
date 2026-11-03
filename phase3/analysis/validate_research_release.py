"""Check research handoff against pinned receipts/manifests, CSVs and actual JS."""
import hashlib,json,re,subprocess,sys
from pathlib import Path
from urllib.parse import unquote
import numpy as np
import pandas as pd
from PIL import Image
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tread_two/research_release';CHECK=ROOT/'tread_two/research_release_validation';CHECK.mkdir(exist_ok=True)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
release=read(OUT/'RELEASE.json');assert release['status']=='local_research_handoff_complete' and not release['HF_publication_of_this_index'] and not release['workstation_changes']
assert release['builder_sha256']==sha(ROOT/'analysis/build_tread_two_research_release.py')
assert all(sha(ROOT/n)==h for n,h in release['source_hashes'].items())
for name,m in release['artifacts'].items():assert (OUT/name).stat().st_size==m['bytes'] and sha(OUT/name)==m['sha256']
for name,m in release['tables'].items():
    frame=pd.read_csv(OUT/(name+'.csv'));assert len(frame)==m['rows'] and list(frame.columns)==m['columns'] and sha(OUT/(name+'.csv'))==m['sha256']
pub=pd.read_csv(OUT/'publications.csv');models=pd.read_csv(OUT/'models.csv');metrics=pd.read_csv(OUT/'endpoint_metrics.csv');artifacts=pd.read_csv(OUT/'artifacts.csv')
assert len(pub)==9 and pub.groupby('kind').size().to_dict()==dict(fit=3,gate=3,review=3)
assert pub.owner_released.all() and pub.verification_status.eq('passed').all()
assert len(models)==3 and models.status.eq('research_only_unpromoted').all() and models.selected_epoch.tolist()==[18,7,7]
assert models.epochs.eq(60).all() and models.optimizer_updates.eq(2940).all()
assert len(metrics)==6 and metrics.observed_class_BA.eq(.5).all() and metrics.low_recall.eq(0).all() and metrics.high_recall.eq(1).all()
manifests={}
for row in pub.itertuples():
    receipt=(OUT/row.receipt).resolve();assert sha(receipt)==row.receipt_sha256 and read(receipt)['status']=='passed'
    folder=receipt.parent;path=folder/('fit_MANIFEST.json' if row.publication=='atlas_v1_fit' else 'remote/REMOTE_MANIFEST.json' if row.publication=='identity_gate' else 'REMOTE_MANIFEST.json')
    mf=read(path);assert mf['contract']==row.contract and len(mf['files'])+1==row.files_with_manifest and sum(m['bytes'] for m in mf['files'].values())==row.payload_bytes
    assert '/'+row.revision+'/' in row.HF_tree and row.prefix in row.HF_tree
    manifests[row.publication]=(mf,row)
assert 'not every large CSV' in pub.set_index('publication').loc['atlas_v1_fit','row_count_scope']
assert len(artifacts)==14
for row in artifacts.itertuples():
    mf,publication=manifests[row.publication];meta=mf['files'][row.logical_name]
    assert meta['sha256']==row.sha256 and meta['bytes']==row.bytes and row.revision==publication.revision
    assert row.url=='https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/'+row.revision+'/'+publication.prefix+'/'+meta['object']
    assert not row.logical_name.endswith(('.zip','.npz'))
links=0
for path in OUT.glob('*.md'):
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
        if target.startswith(('https://','#')):continue
        assert (path.parent/unquote(target)).resolve().exists(),(path,target);links+=1
with Image.open(OUT/'figures/candidate_comparison.png') as image:image.verify()
ui=subprocess.run(['node',str(ROOT/'analysis/test_research_hub.cjs'),str(OUT/'Evidence_Hub.html')],capture_output=True,text=True,encoding='utf-8');assert ui.returncode==0,ui.stderr;viewer=json.loads(ui.stdout)
html=(OUT/'Evidence_Hub.html').read_text(encoding='utf-8')
for target in re.findall(r'(?:href|src)="([^"]+)"',html.split('<script type=')[0]):
    if target.startswith('https://'):continue
    assert (OUT/target).exists(),target
result=dict(status='passed',release_sha256=sha(OUT/'RELEASE.json'),all_source_and_artifact_hashes_match=True,model_cards=3,selected_epochs=[18,7,7],model_status='research_only_unpromoted',
    publications=9,publication_types=dict(fit=3,review=3,gate=3),artifact_descriptors=14,CSV_tables=len(release['tables']),CSV_rows=sum(m['rows'] for m in release['tables'].values()),
    original_row_audit_scope_distinctions_preserved=True,local_model_card_links_checked=links,viewer=viewer,model_forwards=0,optimizer_updates=0,remote_writes=0,HF_publication_of_new_index=False,workstation_changes=False)
(CHECK/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
