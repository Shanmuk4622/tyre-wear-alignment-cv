"""Exercise selected-export provenance with an explicitly injected fixture branch only."""
import csv,json,sys,time
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tread_two/geo_pilot'))
from common import atomic,digest,read
from engine_fit import Engine
source=read(ROOT/'tread_two/geo_pilot/FILES.json')
root=ROOT/('tread_two/geo_validation/fixture_selected_'+digest(source)[:12])
engine=Engine(root,ROOT/'workstation/models/weights.pt',str(ROOT.parent/'phase2/data/phase2_dataset_v1'),'cpu','cpu',True)
while engine.cursor['phase'] not in ('endpoint_setup','completed'):engine.step()
if engine.cursor['phase']!='completed':
    assert engine.cursor['update']==4
    # This is a branch injection in a disposable, never-published fixture. It
    # does not bypass the real measured checkpoint-selection rule.
    engine.cursor['best']=dict(epoch=1,score=0.,path='epochs/epoch_001.pt');engine.cursor['endpoint']='selected'
    while engine.cursor['phase']!='completed':engine.step()
rows=[]
for path in (engine.store.root/'csv/sample_observations').glob('*.csv'):
    with path.open(encoding='utf-8',newline='') as stream:rows+=list(csv.DictReader(stream))
selected=[r for r in rows if r['endpoint']=='selected' and r['stage']=='endpoint']
assert len(selected)==8 and all(int(r['epoch'])==1 and int(r['parameter_update'])==2 and int(r['completed_update'])==4 for r in selected)
assert (engine.store.root/'exports/selected.pt').exists() and (engine.store.root/'exports/final.pt').exists()
result=dict(status='passed',scope='injected_fixture_selected_branch_not_actual_selection',selected_epoch=1,prediction_parameter_update=2,completed_training_updates=4,images=8,remote_writes=0,measured_updates=0)
atomic(ROOT/'tread_two/geo_validation/SELECTED_BRANCH.json',result);print(json.dumps(result,indent=2))
