import json, sys
from pathlib import Path
assert Path(sys.prefix).name == 'cv_conda', sys.prefix
import torch, numpy, PIL
print(json.dumps(dict(python=sys.executable, torch=torch.__version__, numpy=numpy.__version__,
                     cuda=torch.cuda.is_available(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)))
