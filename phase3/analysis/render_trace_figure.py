"""Curated observed-data figure from the already-audited four-example payload."""
import base64
import io
import json
import sys
from pathlib import Path
assert Path(sys.prefix).name == 'cv_conda'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tread_two/trace_report'
cases=json.loads((OUT/'EXAMPLES.json').read_text(encoding='utf-8'))
fig,axes=plt.subplots(3,4,figsize=(12,8),constrained_layout=True)
for k,case in enumerate(cases):
    with Image.open(io.BytesIO(base64.b64decode(case['image'].split(',')[1]))) as image:
        axes[0,k].imshow(np.asarray(image))
    axes[0,k].set_title('Train example '+str((0,100,200,300)[k]),fontsize=11)
    for row,channel in ((1,1),(2,2)):
        a=case['channels'][channel];v=np.array(a['values']);valid=np.array(a['available'])
        axes[row,k].imshow(np.ma.masked_where(~valid,v),interpolation='nearest',
                           cmap='magma' if row==1 else 'RdBu_r',vmin=0 if row==1 else -1,vmax=.5 if row==1 else 1)
    for row in range(3):
        axes[row,k].set_xticks([]);axes[row,k].set_yticks([])
        for spine in axes[row,k].spines.values():spine.set_visible(False)
axes[0,0].set_ylabel('Existing canonical image',fontsize=11)
axes[1,0].set_ylabel('Gradient energy',fontsize=11)
axes[2,0].set_ylabel('Orientation axis',fontsize=11)
fig.colorbar(axes[1,0].images[0],ax=axes[1,:].tolist(),shrink=.8,label='Observed image field')
fig.colorbar(axes[2,0].images[0],ax=axes[2,:].tolist(),shrink=.8,label='Observed image field')
fig.suptitle('TRACE / fixed observed targets, before any measured training',fontsize=14)
fig.savefig(OUT/'figures/TRACE_Fixed_Targets.png',dpi=150)
plt.close(fig)
print('TRACE curated observed-data figure rendered.',flush=True)
