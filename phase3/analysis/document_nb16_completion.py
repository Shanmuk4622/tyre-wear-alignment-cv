"""Document independent NB16 public verification; preserve executed delivery identity."""
import json,re,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(json.loads((ROOT/'tread_two/nb16_completion/LATEST.json').read_text())['path'])
v=json.loads((OUT/'VERIFICATION.json').read_text());assert v['status']=='passed'
fig=ROOT/'tread_two/nb16_report/figures';fig.mkdir(parents=True,exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(2);allocated=[m['peak_allocated_bytes']/1e6 for m in v['GPU_memory']];reserved=[m['peak_reserved_bytes']/1e6 for m in v['GPU_memory']]
figure,axis=plt.subplots(figsize=(8,4),layout='constrained');width=.35
for offset,values,label,color in [(-width/2,allocated,'Allocated','#2875b8'),(width/2,reserved,'Reserved','#e8a32b')]:
    bars=axis.bar(x+offset,values,width,label=label,color=color);axis.bar_label(bars,fmt='%.1f',padding=3)
axis.set_xticks(x,['GPU0 · trainable modules','GPU1 · frozen anchor']);axis.set_ylabel('Peak device memory (decimal MB)');axis.set_ylim(0,max(reserved)*1.25);axis.legend();axis.set_title('NB16 actual dual-T4 engineering fixture\nFull measured-fit peak remains a separate runtime observation')
figure.savefig(fig/'NB16_Resources.png',dpi=150);plt.close(figure)
text=f'''# NB16 completed — independently verified on Hugging Face

**1 November 2026. NB16 passes; no rerun is required.** All three code cells
completed without saved errors. The executed notebook is preserved unchanged:
`{v['executed_notebook_sha256']}`. Its27 embedded files match the original new
GEO runtime exactly. [Inspection](tread_two/nb16_completion/{v['executed_notebook_sha256'][:12]}/INSPECTION.json)
and [independent verification](tread_two/nb16_completion/{v['executed_notebook_sha256'][:12]}/VERIFICATION.json).

## Published evidence

| Check | Verified result |
|---|---|
| HF revision | `{v['revision']}` |
| Contract | `{v['contract']}` |
| Direct publication |174 logical files /173 selected objects /24,341,191 bytes |
| CSV inventory |60 direct CSV files /42,217 rows, including source/schedule and repeated seam/progress paths |
| Native evidence |65 uncompressed finite NPY arrays; original unit1 and latest unit4 chains verified |
| Exact recovery |96 scientific files identical after actual HF restore/continuation; model/moments/counters/RNG checks pass |
| Devices | Actual Tesla T4 ×2, Torch2.11.0+cu128, Python3.13.15 |
| Interactive gallery |32 predicted-mask cases; actual script/filter controls and PNG payloads checked |
| Ownership | Released |
| Earlier experiments |1,502 context-fit,79 context-review and37 readiness manifest entries unchanged |

The CSV row count includes the retained unit1 seam in addition to the latest
four-unit history and duplicate source metadata paths. These are retained
engineering views/copies, not42,217 independent photos. Only existing training
images entered the disposable optimizer checks; no measured candidate fit ran.

```mermaid
sequenceDiagram
  participant A as Four-update direct fixture
  participant H as Hugging Face
  participant B as Fresh numeric worker
  A->>A: Save complete update1 + keep latest update4
  A->>H: Publish direct CSV / NPY / native optimizer state
  H->>B: Download verified update1
  B->>B: Restore optimizer / RNG / next update
  B->>B: Continue updates2–4
  B->>A:96 scientific files match
```

## What we learned

The new geometry/texture/error paths run deterministically on the real two-T4
setup, including native/CSV restore. GPU0 owns the trainable modules; GPU1 owns
the frozen anchor. Parameter counts remain355,879 trainable /1,728,038 total.
These are engineering facts, not an accuracy or physical-wear improvement.

![Actual engineering device memory](tread_two/nb16_report/figures/NB16_Resources.png)

The visible wait was budget handling and completed normally. Saved headers
actually advertised API1,000 and resolvers5,000 per300seconds; NB16's local API
cap96 per300seconds was substantially more conservative. NB17 keeps the96/hour
commit guard, uses conservative defaults until a header is observed, then at
most80% of that observed bucket with fixed ceilings. Server backoff/remaining
still governs. [HF's separate bucket/action policy](https://huggingface.co/docs/hub/en/rate-limits).
The completed NB16 transport itself is unchanged.

All four VIC/covariance/masked-Huber losses independently reconstruct from CSV
projector values and raw native targets/predictions/weights within FP32 reduction
tolerances. The largest unreduced-sum difference is0.000347; normalized objectives
agree within the declared tolerance. Scalar-conversion warnings are diagnostic
warnings, with no failed check or saved error; later logging detaches values.

The14.334 GB forecast in NB16 explicitly excludes full40-family/staging acceptance.
`measured_fit_ready=false` was correct. It is not evidence of a failed notebook.

## Next implementation

The isolated `tread_two/geo_pilot/` branch implements the full40-family measured
pipeline, bounded native prefixes/shards, original validation/selection,
train-only mean predictor, all epoch checkpoints and selected/final diagnostics.
It preserves the completed NB16 runtime and notebook. New disposable storage/
resume acceptance and actual live checks must pass before NB17's first measured
update. Current delivery status is recorded in [GEO pilot](tread_two/geo_pilot/README.md).

No new photos, independent tyres, mileage classifier, depth/safety claim,
workstation edit or rewrite of the completed ATLAS/NB16 evidence is introduced.
'''
(ROOT/'phase3NB16Completion.md').write_text(text,encoding='utf-8')
print(json.dumps({'document':'phase3NB16Completion.md','verified_files':174,'figure':str(fig/'NB16_Resources.png')}))
