# TREAD-TWO / GEO — implemented preparation

<!-- NB16_DONE_STATUS -->
**Current update,1November2026:** [NB16 is complete and HF-verified](../../phase3NB16Completion.md); no rerun. NB17 is delivered in the isolated geo_pilot branch. Earlier pending/preparation statements below are dated history.
<!-- END_NB16_DONE_STATUS -->

**1 November 2026. NB16 is ready for Kaggle; actual T4/HF execution is pending.**
The user approved the geometry-first plan and requested notebooks. This package
implements the next prerequisite: a real-model engineering gate. The measured
60-epoch experiment has not started. No additional photographs are requested.

| Entry | Purpose / status |
|---|---|
| [NB16 notebook](../../notebooks/phase3_NB16_TREAD_TWO_GEO_Preparation_And_Contract_Gate.ipynb) | Self-contained Kaggle preparation; run first |
| [Runbook](RUNBOOK.md) | Exact settings, cells, output, safe stop and recovery |
| [Implementation status](IMPLEMENTATION_STATUS.md) | Actual modules, tensor/parameter counts and remaining fit work |
| [Validation](VALIDATION_AND_RECOVERY.md) | Mathematical checks, complete-update rollback and evidence boundaries |
| [Retention scope](CSV_AND_NATIVE_SCOPE.md) | Actual engineering CSV/native records versus the approved full-fit schema |
| [Master plan](../../phase3TreadTwoGeometryPlan.md) | Original approved geometry-first scientific design |
| [Local proof](validation/DELIVERY.json) | CPU/offline evidence, not an actual two-T4/HF claim |

```mermaid
flowchart LR
  P[Approved design] --> R[Implemented model and engineering store]
  R --> L[Local numerical / restored-state checks]
  L --> K[Run NB16 on two T4s]
  K --> H[Verify published HF receipt independently]
  H --> F[Finish full 40-family shard integration and freeze NB17]
  F --> T[One measured 60-epoch fit]
  T --> V[NB18 saved-evidence review]
```

There are **355,879 trainable / 1,728,038 total parameters** including the frozen
1,372,159-parameter selected TREAD-ONE anchor. Geometry, texture and experimental
error-estimation gradients are isolated. No mileage classifier is fitted here.

NB16 has three code cells: private transport/source extraction; all actual checks
and direct HF publication; receipt and interactive gallery. It embeds 27 files
and stays below 1 MB. Readable source is in `runtime/`; the notebook is generated
from that source. Once the user executes NB16, preserve its exact source/notebook
identity and inspect its outputs rather than regenerating it.

NB17 and NB18 remain reserved next-stage names. The approved acceptance sequence
requires actual two-T4/HF evidence before measured fitting; no empty placeholder,
simulated pass or auto-starting training notebook is supplied. Both workstations
and all completed classifier experiments stay preserved.
