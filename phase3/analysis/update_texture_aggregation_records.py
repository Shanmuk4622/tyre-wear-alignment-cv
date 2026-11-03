"""Current navigation for completed saved-aggregation investigation; no old report edits."""
import re
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def prepend(name,body):
    body=re.sub(r'\b(all|both|same|preserves|measured|restores|retains|to|for|epoch|update|from|uses|about|and|after|before|view)(?=\d)',r'\1 ',body)
    body=re.sub(r'(?<=\d)(FP64|MB|GB|images|native|decoder|projector)\b',r' \1',body)
    p=R/name;old=p.read_text(encoding='utf-8');tag='TEXTURE_AGGREGATION_20261102'
    block='<!-- '+tag+' -->\n'+body.rstrip()+'\n<!-- END_'+tag+' -->'
    if '<!-- '+tag+' -->' in old:new=re.sub(re.escape('<!-- '+tag+' -->')+r'.*?'+re.escape('<!-- END_'+tag+' -->'),lambda _:block,old,flags=re.S)
    else:first,rest=old.split('\n',1);new=first+'\n\n'+block+'\n'+rest
    if new!=old:p.write_text(new,encoding='utf-8',newline='\n')
notice='''## Latest — saved texture aggregation investigation complete

**2 November 2026:** [Full investigation](phase3TextureAggregationInvestigation.md)
traces actual TRACE local/context features, nine predeclared aggregations,
two recorded views and unchanged decoder weights across the existing570 images.
No image/geometry inference, optimizer updates, new photos or workstation edits.
Frozen decoder/projector readouts are explicitly counted; this is not a
zero-computation review or a newly trained candidate.

One average preserves61.14% of local validation advantage over the position
control; a train-position template plus five residual moments preserves81.76%.
Orientation/contrast suffer disproportionately. Averaging preferentially
removes within-component variation, while context already changes diversity.
Invertible scaling and mask placement can inflate rank without useful new
texture information. Original gates/selections, including selected=None, stay intact.

[Interactive explorer](texture_aggregation_20261102/Aggregation_Explorer.html) ·
[Methods and direct-file dictionary](texture_aggregation_20261102/METHODS.md).
New raw/derived CSV and FP64 NPY evidence is local, not HF-published. Large
native/detail tables are ignored by Git; compact summaries/figures remain visible.
The supported next design question is incremental local spatial-residual utility
beyond global/position controls. No new architecture/objective or fit is selected here.'''
notice=notice.replace('existing570','existing 570').replace('preserves61','preserves 61').replace('preserves81','preserves 81')
for name in ['AGENTS.md','README.md','phase3Progress.md','phase3Understanding.md','phase3NextSteps.md','phase3OpenItems.md']:prepend(name,notice)
records={
'phase3Decisions.md':'''## Aggregation investigation decisions — 2 November 2026

| ID | Decision | Evidence / boundary |
|---|---|---|
| P3-D187 | Freeze nine saved-feature interventions before primary calculation | Same final epoch, original roles/views/support, no new candidate search |
| P3-D188 | Count frozen-head readouts explicitly |656,640 decoder-token /1,140 projector-vector readouts; zero image/geometry forwards or optimizer updates |
| P3-D189 | Separate local-decoder utility from global-vector diversity | Actual source graph, native bridge replay and fixed-head counterfactuals |
| P3-D190 | Preserve original checkpoint selection and gate | GEO/TRACE selected=None; diagnostic alternatives never installed |
| P3-D191 | Declare mask association follow-up as exploratory | Separately pinned addendum after observing high masked-context rank |
| P3-D192 | Use an invertible scaling control for the rank statistic | Original coordinates recover to1e-12; no new information or gate relaxation |
| P3-D193 | Prioritize local contrast/orientation residual utility in the next design question | Evidence supports a question, not a trained objective or architecture claim |
| P3-D194 | Retain full direct CSV/native derivatives locally with explicit Git/HF boundaries | Large predictions/detail kept out of Git; new derivative HF publication not claimed |
''',
'phase3SessionLog.md':'''## 2 November 2026 — authorized local-texture aggregation investigation

1. Froze the primary nine-variant protocol, then fetched/reused249 pinned native/metadata bodies,66,722,671 bytes, using public reads only.
2. Verified all saved pooling/projector/decoder bridges and every original local/masked prediction replay. Compared unchanged decoders after weighted means, band/column means, spatial/random modes and train-only templates/residuals.
3. Retained every counterfactual prediction plus image/channel/tile losses, all feature spectra and covariance/energy details. No image encoder/geometry forward or neural training.
4. Measured61.14% local validation utility retention for one average and81.76% for position-template +five residual moments; identified orientation/contrast losses and disproportionate low-group detail loss.
5. Distinguished context/aggregation concentration, rank scale-dependence and exploratory mask-layout association. Reconciled all channel/role/component reductions and retained original selected=None.
6. Added scientific Markdown/methods, two figures and an interactive component/tile explorer. New derivatives local; source evidence, notebooks, workstations and previous reports preserved.
''',
'phase3EvidenceRegister.md':'''## TRACE aggregation investigation evidence

| Evidence | Scope |
|---|---|
| [Primary protocol](texture_aggregation_20261102/PROTOCOL.json) | Nine fixed interventions declared before primary results |
| [Input verification](texture_aggregation_20261102/INPUT_VERIFICATION.json) |249 full-body checked native/metadata objects; pinned source |
| [Numerical bridges](texture_aggregation_20261102/BRIDGE_VERIFICATION.csv) | Six original source/readout bridges including all native decoder predictions |
| [Role utility](texture_aggregation_20261102/UTILITY_SUMMARY.csv) /[components](texture_aggregation_20261102/COMPONENT_UTILITY.csv) | Both branches, original roles/components/views and all nine interventions |
| [Descriptor utility](texture_aggregation_20261102/CHANNEL_UTILITY_SUMMARY.csv) | Contrast/orientation damage and exact channel-to-image reconciliation |
| [Covariance](texture_aggregation_20261102/SPATIAL_COVARIANCE.csv) /[spaces](texture_aggregation_20261102/FEATURE_SPACES.csv) | Exact mean-mode/spatial decomposition and scale-dependent ranks |
| [Exploratory mask addendum](texture_aggregation_20261102/MASK_ADDENDUM_PROTOCOL.json) | Mask-plan alignment is association, not physical evidence |
| [Methods/dictionary](texture_aggregation_20261102/METHODS.md) | All native/detail grains, units, caveats and local/publication boundary |
''',
'phase3DocumentationInventory.md':'''## Local texture aggregation documentation

| Document | Purpose |
|---|---|
| [Investigation findings](phase3TextureAggregationInvestigation.md) | Architecture branch trace, all comparisons, mechanisms, implications and limits |
| [Methods and data dictionary](texture_aggregation_20261102/METHODS.md) | Projection equations, frozen-head replay, weighting, schemas and retention |
| [Package index](texture_aggregation_20261102/README.md) | Source/result navigation and local/HF status |
| [Interactive explorer](texture_aggregation_20261102/Aggregation_Explorer.html) | Role/component/view/branch/intervention controls and canonical tile-loss maps |
''',
'phase3RepositoryMap.md':'''## Saved aggregation investigation locations

| Path | Ownership / scope |
|---|---|
| `texture_aggregation_20261102/` | New protocols, compact CSV summaries, receipts and offline explorer |
| `texture_aggregation_20261102/native/`, `detail/` | Large derived predictions/features and full CSV detail; local/Git-ignored |
| `texture_aggregation_20261102/figures/` | Curated generated scientific charts, Git-visible |
| `analysis/*texture_aggregation*.py`, `analysis/texture_mask_confounds.py` | Public pinned retrieval, frozen-token algebra, validation and documentation; no new training |
| Existing NB24 mirror | Source evidence read-only; additional pinned native arrays downloaded into its already ignored remote cache |
''',
'phase3DataRetentionPolicy.md':'''## Aggregation-analysis retention — 2 November 2026

The new saved-feature assay retains all18 FP64 counterfactual prediction arrays,
per-image/channel/tile losses, feature representations/eigenspectra, covariance
matrices, train-only template/scaling parameters and full provenance as direct
CSV/NPY/JSON. Primary derivatives are about769MB, with no archive envelope.

Native/detail and large image/channel tables remain local and Git-ignored;
compact summaries, source and figures are retained for review. They have **not**
been uploaded to HF in this analysis. Original source science is already pinned
on HF. [Exact methods and file grains](texture_aggregation_20261102/METHODS.md).
''',
'phase3DocumentationConflicts.md':'''## Aggregation assay scope — 2 November 2026

| Wording to avoid | Exact interpretation |
|---|---|
| No model computation occurred | No image/geometry forward or optimizer update; frozen decoder/projector readouts are counted explicitly |
| Pooled collapse contradicts useful reconstruction | Original local/masked decoders use spatial tokens, bypassing the global projector |
|82% retention means82% accuracy | It is a retained loss advantage over the train-position control; no classification evaluated |
|95.85% of wear information was lost | This is projected-away component-centered token variance under a fixed-mass diagnostic, not semantic information |
| Higher masked rank or scaled rank fixes representation | Mask layout and invertible coordinate scaling can inflate the statistic; original rule stays unchanged |
| The best counterfactual is a selected new model | Same frozen decoder, potentially off-manifold features, no new head training or checkpoint selection |

[Investigation](phase3TextureAggregationInvestigation.md) preserves original reports,
selection and existing-photo/deferred-workstation scope. Mask association is an
explicit post-hoc addendum. New derivative HF publication remains unperformed.
'''}
for name,body in records.items():prepend(name,body)
p=R/'.gitignore';old=p.read_text(encoding='utf-8');tag='# Large saved-token aggregation derivatives; keep compact evidence visible.'
if tag not in old:
    old=old.rstrip()+'\n\n'+tag+'\n/texture_aggregation_20261102/native/\n/texture_aggregation_20261102/detail/\n/texture_aggregation_20261102/IMAGE_LOSSES_*.csv\n/texture_aggregation_20261102/CHANNEL_LOSSES_*.csv\n/texture_aggregation_20261102/SPATIAL_ENERGY.csv\n!/texture_aggregation_20261102/figures/*.png\n'
    p.write_text(old,encoding='utf-8',newline='\n')
print('Investigation linked in current records; full direct local evidence retained and ignored appropriately.')
