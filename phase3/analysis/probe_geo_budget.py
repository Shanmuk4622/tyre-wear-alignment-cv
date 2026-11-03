import json,sys
from pathlib import Path
from types import SimpleNamespace
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tread_two/geo_pilot'))
import torch
from common import read
from model_geo import Geo
from budget_fit import project
previous=read(ROOT/'tread_two/geo_validation/LIFECYCLE.json');root=Path(previous['fixture']);pointer=read(root/'run/POINTER.json')
state=torch.load(root/'run'/pointer['state'],map_location='cpu',weights_only=False)
engine=SimpleNamespace(root=root,model=Geo(),store=SimpleNamespace(root=root/'run',csv=state['csv']))
report=project(engine)
print(json.dumps({k:v for k,v in report.items() if k not in ('measured_CSV','typed_width_bounds')},indent=2))
