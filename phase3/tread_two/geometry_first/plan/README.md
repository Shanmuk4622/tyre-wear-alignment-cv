# TREAD-TWO / GEO — complete design plan

<!-- NB16_DONE_STATUS -->
**Current update,1November2026:** [NB16 is complete and HF-verified](../../../phase3NB16Completion.md); no rerun. NB17 is delivered in the isolated geo_pilot branch. Earlier pending/preparation statements below are dated history.
<!-- END_NB16_DONE_STATUS -->

<!-- NB16_STATUS -->
**Implementation update, 1 November 2026:** [NB16 preparation is delivered](../../../phase3NB16Preparation.md). Local model/recovery checks pass; actual Kaggle/HF and full measured-fit integration remain pending. The design below is the original planning snapshot.
<!-- END_NB16_STATUS -->

**Status: design complete; implementation, numerical/resource gates and measured results pending.** User requested the full geometry-first plan on 1 November 2026. This is a new objective under TREAD-TWO; the completed ATLAS classifier series and both workstations remain preserved.

| Read in order | Purpose |
|---|---|
| [Project plan](../../../phase3TreadTwoGeometryPlan.md) | Goal, evidence, deliverables, stages and acceptance |
| [Architecture](ARCHITECTURE.md) | Exact branch shapes, gradients, geometry/texture/error heads and loss math |
| [Experiment](EXPERIMENT.md) | Data, deterministic schedule, augmentations, selection and evaluation |
| [Retention](RETENTION.md) | Complete draft CSV fields, native raw tensors, budgets and explicit gaps |
| [Operations](OPERATIONS.md) | Kaggle cells, two-GPU ownership, recovery/HF/stop/resource behavior |
| [Implementation](IMPLEMENTATION.md) | File ownership, dependency order, actual test cases and next notebooks |
| [Risks/decisions](RISKS_AND_DECISIONS.md) | Failure controls, comparison limits, unresolved measurements |
| [PLAN.json](PLAN.json) | Chosen design values, status and named engineering thresholds |
| [SCHEMAS.json](SCHEMAS.json) | Every planned CSV family/field/key/cadence; executable dtypes pending freeze |
| [BUDGET.json](BUDGET.json) | Shape-based raw byte totals and provisional CSV/staging allowances; actual fixture still required |
| [SOURCES.json](SOURCES.json) | Pinned local evidence hashes and consulted primary sources |

No model implementation, training notebook, optimizer update, new inference, HF write or workstation change is claimed by these design documents. The next implementation deliverable is the planned NB16 engineering/contract gate; it will be a Kaggle .ipynb, not a pasted training script. NB17 is the single measured fit only after its gates pass. A successful new model is a hypothesis to test, not a result.

[Planning checks](../validation/PLAN_VALIDATION.json) cover source preservation, metadata/shape arithmetic and schedule feasibility only; model math and actual T4/resource checks are pending.
