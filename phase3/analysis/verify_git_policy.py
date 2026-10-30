"""Verify ignore rules, retained frozen inputs and index-only cleanup integrity."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'phase3/audit'
def git(*args):return subprocess.run(['git',*args],cwd=ROOT,capture_output=True,check=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--snapshot-index',action='store_true');args=parser.parse_args()
receipt=OUT/'git-index-cleanup.json'
if args.snapshot_index:
    names=git('ls-files','-ci','--exclude-standard','-z').stdout.decode('utf-8').strip('\0').split('\0')
    names=[n for n in names if n]
    assert len(names)==7 and names[0]=='phase2/local_validation/assets/segformer/pytorch_model.bin'
    assert all(n.endswith('.bin') or n.startswith('phase2/notebooks/execution_archives/') for n in names)
    assert not receipt.exists(),'Preserve existing cleanup baseline'
    receipt.write_text(json.dumps(dict(action='stop tracking only; local files must remain unchanged',files={n:dict(bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in names}),indent=2)+'\n',encoding='utf-8')
    print('Snapshotted seven local files before index-only cleanup');raise SystemExit(0)

ignored=['.env','phase3/.env.local','phase3/.aws/credentials','phase2/model.bin','phase3/future/checkpoint.pt','phase3/future/weights.safetensors',
 'phase3/tread_two/nb11_completion/16fa0443f902/remote/csv/epoch_0001/dense_pixels.csv',
 'phase3/tread_two/nb10_completion/499c84cf128a/remote/report/candidate_bank.png',
 'phase3/tread_two/lc_validation/acceptance_example/source/model.py','phase3/tread_two/lc_validation/nb11_extract_example/source/worker.py',
 'phase3/tread_two/lc_validation/upstream/artifacts/train/0010.pt','phase3/tread_two/lc_review_validation/acceptance_example/fresh_report/prediction_audit.csv',
 'phase3/tread_two/lc_analysis_report/prediction_audit.csv','phase3/tread_two/lc_analysis_report/ATLAS_Failure_Explorer.html',
 'phase3/tread_two/nb13_completion/a/remote/csv/history.csv','phase3/tread_two/nb13_completion/a/executed.ipynb',
 'phase3/next/objects/telemetry.csv','phase3/next/bundle/review.csv','phase2/notebooks/execution_archives/example.ipynb','phase3/analysis/__pycache__/test.pyc']
visible=['.gitignore','.env.example','README.md','docs/report/assets/study_overview.svg','phase3/AGENTS.md',
 'phase3/notebooks/phase3_NB11_TREAD_TWO_ATLAS_LC_Resumable_Pilot.ipynb','phase3/notebooks/phase3_NB12_TREAD_TWO_ATLAS_LC_Completed_Review.ipynb',
 'phase3/tread_two/lc_pilot/worker.py','phase3/tread_two/lc_pilot/FILES.json','phase3/tread_two/lc_review/assets/v1_cohort_metrics.csv',
 'phase3/tread_two/lc_analysis_report/cohort_metrics.csv','phase3/tread_two/lc_analysis_report/figures/paired_learning.png',
 'phase3/tread_two/lc_validation/figures/reference_channels.png','phase3/workstation/app.py']
visible += [f'phase3/tread_two/{folder}/assets/{name}.npy' for folder,names in [('pilot',['bank_rgb','bank_labels']),('lc_pilot',['bank_rgb','bank_labels','bank_valid','bank_sha256'])] for name in names]
for name in ignored:assert git('check-ignore','--no-index','-q','--',name).returncode==0,('must ignore',name)
for name in visible:assert git('check-ignore','--no-index','-q','--',name).returncode==1,('must retain',name)
baseline=json.loads(receipt.read_text(encoding='utf-8'))
tracked=set(git('ls-files','-z').stdout.decode().split('\0'))
for name,m in baseline['files'].items():
    assert name not in tracked and (ROOT/name).stat().st_size==m['bytes'] and sha(ROOT/name)==m['sha256'],name
assert not git('ls-files','-ci','--exclude-standard').stdout.strip(),'Previously tracked ignored files remain in the index'

# Copy only four published figures out of ignored mirrors so Git documentation
# keeps exact images without reopening a multi-GB remote tree.
figures=[
 ('phase3/phase3NB07Completion.md','data_gate_validation/nb07_publication/remote/figures/NB07_class_and_subject_support.png','data_gate_validation/nb07_publication/figures/NB07_class_and_subject_support.png'),
 ('phase3/phase3NB07Completion.md','data_gate_validation/nb07_publication/remote/figures/NB07_quality_and_annotation_review.png','data_gate_validation/nb07_publication/figures/NB07_quality_and_annotation_review.png'),
 ('phase3/phase3NB06Completion.md','revision2_validation/nb06_publication/remote/figures/NB05_endpoint_comparison.png','revision2_validation/nb06_publication/figures/NB05_endpoint_comparison.png'),
 ('phase3/phase3NB10Completion.md','tread_two/nb10_completion/499c84cf128a/remote/report/candidate_bank.png','tread_two/nb10_completion/499c84cf128a/figures/candidate_bank.png')]
for doc,old,new in figures:
    source=ROOT/'phase3'/old;dest=ROOT/'phase3'/new
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():shutil.copy2(source,dest)
    assert sha(dest)==sha(source) and git('check-ignore','--no-index','-q','--',dest.relative_to(ROOT).as_posix()).returncode==1
    p=ROOT/doc;text=p.read_text(encoding='utf-8');p.write_text(text.replace(old,new),encoding='utf-8')
result=dict(status='passed',ignored_path_checks=len(ignored),retained_path_checks=len(visible),index_only_untracked_files=len(baseline['files']),all7_local_files_preserved_byte_identical=True,
  documentation_figures_copied_byte_identical=len(figures),frozen_reference_arrays_visible=6,history_rewritten=False,source_notebooks_rewritten=False)
(OUT/'GIT_IGNORE_POLICY_CHECK.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
