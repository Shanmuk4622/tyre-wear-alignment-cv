"""Write verified NB12 findings and the isolated r3 protocol/engineering handoff."""
import json,sys,textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tread_two/context_validation';SOURCE=ROOT/'tread_two/context_pilot'
sys.path.insert(0,str(SOURCE))
from common import read,sha,atomic
from schemas import SCHEMAS
v=read(ROOT/'tread_two/nb12_completion/036132254087/VERIFICATION.json')
local=read(OUT/'LOCAL_VALIDATION.json');math=read(OUT/'MATH_VALIDATION.json');package=read(OUT/'PACKAGE_VALIDATION.json');account=read(OUT/'CSV_ACCOUNTING.json')
assert all(r['status']=='passed' for r in (v,local,math,package,account))
assert v['files']==63 and all(r['maximum_numeric_difference']==0 for r in v['table_comparisons'])
def write(name,text):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(textwrap.dedent(text).strip()+'\n',encoding='utf-8')

# A measured structural diagnostic, not an accuracy plot or fabricated learning curve.
figures=OUT/'figures';figures.mkdir(exist_ok=True)
df=pd.read_csv(OUT/'layout_invariance.csv');fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
colors={'reverse_bands':'#147d92','reverse_columns':'#dc8347'}
for i,(variant,rows) in enumerate(df.groupby('variant',sort=False)):
    axes[0].scatter(range(1,9),rows.logit_max_abs_delta/1e-8,label=variant.replace('_',' '),s=60,marker=('o','x')[i],color=colors[variant])
axes[0].set(xlabel='Training photo (one per known training tyre)',ylabel='Maximum absolute logit change (×10⁻⁸)',title='Frozen LC discards these layout changes',xticks=range(1,9))
axes[0].grid(alpha=.2);axes[0].legend(frameon=False)
axes[1].axis('off');axes[1].text(.03,.95,'What this check establishes',va='top',fontsize=15,weight='bold',color='#147d92')
axes[1].text(.03,.81,'8 existing training tyres · 16 rearrangements\n24 frozen-head calls · no new dense forwards\n0 optimizer updates · 0 held-out forwards\n\nMaximum logit change: 5.96 × 10⁻⁸\n\nIt motivates a position-aware correction.\nIt does not identify the cause of low-class errors\nor predict whether the correction will help.',va='top',fontsize=12,linespacing=1.55)
fig.suptitle('ATLAS-CONTEXT r3: evidence behind the next hypothesis',fontsize=16)
fig.savefig(figures/'layout_invariance.png',dpi=160);plt.close(fig)

write('phase3NB12Completion.md',r'''
# NB12 completed — ATLAS-LC saved-evidence review

**30 October 2026. The supplied executed NB12 is complete and independently HF-verified. No rerun is needed.** Its four cells finished without errors. The review adds derived evidence; it performs zero model forwards and zero training updates. The original 60-epoch NB11 fit is preserved.

## Publication receipt

| Check | Verified result |
|---|---|
| Executed notebook | `0361322540878593f7e16066bd28c7d21899b01576936da33f1c389582937980` |
| Embedded delivery | All 39 payload files match; all four code cells completed |
| Pinned HF revision | `77d03303a9d42f89bfd3fbc567faedf86b09a855` |
| Review contract | `7e2856500bbd1a9dd76e0057d82642dd7601cb9e23d02f9f52afebc784c4af7e` |
| Direct publication | 63 files including manifest; all 62 object sizes and SHA-256 values checked |
| Verified payload | 112,895,999 bytes |
| Scientific tables | 40 CSVs / 380,595 rows |
| Visual review | Four PNG figures; 1,442 endpoint/probe cases and 18 reference cards |
| Independent table comparison | Every table's rows/columns/text/numbers checked; maximum numeric difference 0 at retained precision |
| Original fit preservation | All 1,350 NB11 manifest entries unchanged at the review revision |
| Writer | Released ownership |
| Local audit | Public reads only; no credential files read and no remote writes |

[Machine verification](tread_two/nb12_completion/036132254087/VERIFICATION.json), [input inspection](tread_two/nb12_completion/036132254087/INSPECTION.json), [published manifest](tread_two/nb12_completion/036132254087/REMOTE_MANIFEST.json), [HF tree at verified revision](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/tree/77d03303a9d42f89bfd3fbc567faedf86b09a855/phase3/tread-two/atlas-lc-r2/review/7e2856500bbd1a9dd76e0057d82642dd7601cb9e23d02f9f52afebc784c4af7e).

The remote review used Python 3.13.15 / NumPy 2.1.3 / pandas 2.3.3; local reconstruction used Python 3.11.15 / NumPy 2.4.6 / pandas 3.0.3. Exact retained-precision table agreement was checked rather than assumed across those environments. The review reads 696 saved scientific inputs. Exhaustive verification of the original 54,336,368 raw rows belongs to the separate [NB11 audit](phase3NB11Completion.md), not to a claim that NB12 recomputed every raw family.

```mermaid
flowchart LR
    A[NB11 completed fit\n60 epochs / 2940 updates] --> B[NB12 saved-data review\nzero forwards / zero fitting]
    B --> C[40 derived CSVs\n4 figures / 1442 cases]
    C --> D[Independent pinned-HF audit\nall published bytes verified]
    D --> E[Close NB12\nretain limitations]
    E --> F[Separate r3 context hypothesis\nNB13 accuracy unmeasured]
```

## What the completed evidence says

The local-contrast revision solved a particular appearance-sensitivity problem in the saved probes. It did not solve low-class recognition. At both selected and final endpoints, all 62 low images still become mid and all 41 high images remain high. There are no mid ground-truth endpoint images, so observed-class balanced accuracy averages only low/high recall.

| Historical endpoint metric | v1 selected | LC selected, epoch 7 | LC final, epoch 60 |
|---|---:|---:|---:|
| Observed-class balanced accuracy | 50.00% | 50.00% | 50.00% |
| Accuracy | 39.806% | 39.806% | 39.806% |
| Low recall | 0/62 | 0/62 | 0/62 |
| High recall | 41/41 | 41/41 | 41/41 |
| Three-class NLL, lower is better | 1.178296 | 0.961273 | 1.369872 |
| Brier, lower is better | 0.711430 | 0.568355 | 0.854731 |
| ECE, 10 equal-width bins | 0.162204 | 0.425969 | 0.538923 |

LC's selected NLL/Brier are better; selected ECE is worse. These results support distinct conclusions about probability loss and raw-score calibration. They do not justify promotion. Validation selected epoch 7 / update 343 using the declared BA→NLL→earliest rule, with validation BA 56.25% and NLL 0.934841. Final weights must not replace it based on a test preference.

![Completed fit, endpoint confusion and selection evidence](tread_two/lc_analysis_report/figures/fit_and_confusion.png)

All six predeclared probe types produced zero argmax flips at both LC endpoints. Their scores still move: final gamma 1.2 changed a class probability by up to 0.03686; selected top-patch occlusion by up to 0.00987. Grayscale stability follows from the representation by construction. Small perturbation sensitivity and useful discrimination are different questions.

![Probe behavior and reference affinities](tread_two/lc_analysis_report/figures/probes_and_affinities.png)

The sampling evidence is also stronger than v1: all 386 training queries were drawn, with 23,520 draws / 47,040 correlated view observations. The largest image exposure was 423 draws, versus 1,948 in v1. Full source-SHA exclusion blocked 4,882 view-reference rows, equivalent to 2,441 paired draws. Better coverage and correct self exclusion did not remove cohort/identity shortcuts or create more independent tyres.

![Paired-view losses and exposure](tread_two/lc_analysis_report/figures/paired_learning.png)

Use the [40-table dictionary](tread_two/lc_analysis_report/CSV_INDEX.md), [methods](tread_two/lc_analysis_report/METHODS.md), and [interactive failure explorer](tread_two/lc_analysis_report/ATLAS_Failure_Explorer.html) for the full evidence. The large explorer/row tables are published on HF and excluded from Git; compact summaries and curated figures stay versioned.

## Next hypothesis and decision

The previous classifiers aggregate patch/reference evidence through a shared local encoder and equal-band pooling. That calculation cannot distinguish arbitrary reordering of bands or columns within a band when the corresponding local features/support move together. A new training-only check used one saved photo from each of eight known training tyres: 16 rearrangements changed LC logits by at most 5.96×10⁻⁸. This identifies a representational limitation, not the cause of the endpoint failures.

![Training-only layout diagnostic](tread_two/context_validation/figures/layout_invariance.png)

The next isolated candidate is **ATLAS-CONTEXT r3 / NB13**. It freezes the original validation-selected LC epoch 7 and learns a bounded position-aware residual. Initial predictions match LC exactly. [Architecture](phase3TreadTwoContextArchitecture.md), [frozen protocol](phase3TreadTwoContextProtocol.md), [CSV schema](phase3TreadTwoContextEvidenceSchema.md), [engineering checks](phase3TreadTwoContextValidation.md), [Kaggle instructions](tread_two/context_pilot/README.md).

Its accuracy is unmeasured. No new photos, repeated old fit, test-selected base, or workstation change was introduced. The old historical test has already informed development and remains development evidence, not an independent confirmation set. Proxy classes are not physical wear, depth or safety labels.
''')

write('phase3TreadTwoContextArchitecture.md',r'''
# TREAD-TWO / ATLAS-CONTEXT r3 — spatial correction with an auditable prior

**Implemented for NB13; measured accuracy is pending Kaggle execution.** This is one new candidate on the existing cohort. It preserves the selected LC model and changes the hypothesis that spatial layout is available to classification. The [NB12 findings](phase3NB12Completion.md) motivate it; improved recognition is not established.

## Why this architecture follows the evidence

| Observation | Design consequence | What remains uncertain |
|---|---|---|
| LC is stable to saved appearance probes but still misses every endpoint low image | Keep its representation/prior; add capacity for information it discards | Texture layout may still be insufficient or confounded |
| Local encoder + equal-band sum discards band order | Explicit 2D token positions and attention across all 32 tiles | Architecture sensitivity does not imply learned useful layout |
| Small cohort and prior overfitting | Freeze the entire selected LC branch; bounded residual with zero initial head and magnitude penalty | Additional 231,555 parameters can still overfit identity/artifacts |
| Reference evidence alone no longer explains the full prediction | Retain base logits, per-patch correction, global bias correction, combined signed evidence and attention separately | Attention is not a causal physical explanation |
| User wants one practical, deeply logged run | Same seed/budget, sampler, recovery and Kaggle workflow | Actual T4×2/live-HF proof is still an automatic gate |

The layout check used eight existing training-photo packets and no held-out forward, new dense inference or optimizer update. [CSV](tread_two/context_validation/layout_invariance.csv), [receipt](tread_two/context_validation/LAYOUT_PROOF.json).

```mermaid
flowchart TD
    A[Existing RGB tyre image\n512 x 384] --> B[Frozen TREAD-ONE dense branch]
    B --> C[Masks / boxes / points\n512 x 256 tread atlas]
    C --> D[Exact transformed local-contrast tiles\n32 x 4 x 64 x 64]
    D --> E[Frozen selected LC epoch 7\n18 literal reference patches]
    E --> F[Base class logits\nreference evidence / SHA exclusion]
    D --> G[Trainable warm-start patch encoder\n32 x 96 tokens]
    G --> H[Fixed 2D positions + coverage embedding]
    H --> I[Two context blocks\n4 heads / 96 dimensions]
    I --> J[Zero-initialized head\nbounded weighted tile correction]
    F --> K[Combined low / mid / high logits]
    J --> K
    L[Bounded learned global bias] --> K
    K --> M[Direct CSV evidence + native recovery\nbase / correction / combined accounting]
```

## Frozen prior and trainable components

The prior is the exact LC validation-selected epoch 7 from protocol `ae918ce787775192ce38031936cbfd8285b51b72fbbedb1da570c9271018eb28`, original fit revision `3d5ae39602c3b25caa6af65c1e8d45f53a08e1c8`. Its native checkpoint is 7,040,141 bytes with SHA-256 `f6156aa560aeb880317c6acd745ec1b287aa615fd76656411059033b6a5959fd`. [Pinned input](tread_two/context_pilot/BASE_INPUT.json).

Every base parameter and buffer remains frozen in evaluation mode, including the dense anchor, reference encoder, weights, temperature and bias. The worker checks the full base state after each committed unit and requires no base gradients. The context encoder starts from a separate exact copy of the LC patch encoder; it can learn without changing that prior. No test-picked epoch or alternative bank is loaded.

| Component | Definition |
|---|---|
| Atlas | Eight bands × four columns; each tile 64×64; frozen geometry |
| Four LC channels | Masked 9×9 local luminance standardization, std floor 0.02, clip ±3 then divide by 3; Sobel x/y/magnitude |
| Context patch encoder | Warm-start LC encoder; 96D token per tile |
| Position | Fixed 96D row/column Fourier features at 0.1 scale; 24 frequencies per sine/cosine pair, frequencies `2**linspace(0,3,24)` |
| Coverage | Learned 2→96 linear map of valid fraction and pooling mass |
| Each of two blocks | LN96 → QKV96→288 → four 24D heads → projection96→96 and residual; LN96 → FF96→192→96 with GELU and residual |
| Attention | Explicit FP32 QK/softmax/value multiplication; invalid keys masked at −10,000; all-empty support uses a dummy valid key |
| Correction head | LN96 then bias-free Linear96→3, zero-initialized; `2*tanh(head(token))*pooling_mass` |
| Global correction | Three trainable zero-initialized parameters through `0.5*tanh` |
| Trainable / total | **231,555 / 1,684,988**; frozen base 1,453,433 |
| Execution | FP32 throughout, no dropout/BatchNorm in context attention, no SDPA/flash path, no DDP |

## Exact prediction accounting

Let `alpha_p` be the equal-valid-band patch mass, `a_pr` the frozen reference affinity, `m_r` the full-SHA permission, `s_r=6/n_allowed(class_r)` and `w_cr` the frozen signed weight. Then:

```text
base_evidence[p,c] = alpha[p] * sum_r(a[p,r] * w[c,r] * m[r] * s[r])
delta[p,c]         = alpha[p] * 2 * tanh(head(context_token[p])[c])
bias_delta[c]      = 0.5 * tanh(global_parameter[c])
logit[c]           = base_bias[c] + bias_delta[c]
                     + sum_p(base_evidence[p,c] + delta[p,c])
```

The context branch can use all valid tokens to compute a tile's correction, so the combined tile value accounts for logit mass rather than proving the tile independently caused the decision. On a supported atlas, each class's summed spatial correction is bounded by ±2 and global correction by ±0.5. With no support, tile correction is zero; the finite global correction remains. Initial head/global parameters are zero, so initial logits and patch evidence are bit-identical to selected LC.

The native spatial projection deterministically scatters combined signed evidence through the frozen atlas grid and conserves class-logit mass with bias/unsupported mass handled separately. Original-image projections are retained for identity selected/final endpoints. Structural permutations retain source-patch mapping; they are not projected as if they were natural new photographs.

## Paired objective

```text
L = mean_draws[0.5*(CE_A + CE_B)
               + 0.1*JS(P_A,P_B)
               + 0.05*sum_p(alpha_p*(1-cos(context_A[p],context_B[p])))
               + 0.02*0.5*(mean_class(total_correction_A²)
                           + mean_class(total_correction_B²))]
```

Both query-only views use deterministic seed-addressed brightness/contrast/gamma factors independently uniform in [0.8,1.2]. The frozen dense result is shared; both branches consume the same transformed LC tiles. Reference patches are not independently appearance-augmented. Both CE terms, JS, contextual cosine distance, correction penalty and total are retained per pair and update. Patch distances and mass support independent reconstruction. No extra test loss, label-conditioned bank or teacher label enters this objective.

## Diagnostics and interpretation

The six inherited diagnostics are grayscale-both, brightness 0.8/1.2, gamma 0.8/1.2 and top signed-patch occlusion. Two structural probes reverse eight-band order or reverse four columns within every band, moving local features/support/reference evidence together while fixed position codes remain in place. They preserve the frozen pooled prior. Their deltas test learned positional dependence, not realistic acquisition robustness. The old image thumbnail is only an identity reference for these rearrangements.

Reference cards describe the frozen LC case branch. They cannot explain the learned context term by themselves. The explorer explicitly shows base logits and total correction; CSVs retain the tile decomposition. Attention summaries/matrices are internal behavior diagnostics and must not be called physical-wear ground truth or causal explanations.

[Protocol](phase3TreadTwoContextProtocol.md) · [Evidence dictionary](phase3TreadTwoContextEvidenceSchema.md) · [Engineering proof](phase3TreadTwoContextValidation.md) · [Run guide](tread_two/context_pilot/README.md).
''')

write('phase3TreadTwoContextProtocol.md',r'''
# ATLAS-CONTEXT r3 — frozen one-seed protocol

**State: implemented and locally checked; actual NB13 T4×2/live-HF preflight, 60-epoch fit and accuracy pending.** The local engineering contract is not the Kaggle measured-fit contract. Actual environment and device identities are included in each contract. [Recipe](tread_two/context_pilot/RECIPE.json), [32-file frozen inventory](tread_two/context_pilot/FILES.json).

## Cohort and boundaries

| Item | Fixed policy |
|---|---|
| Inputs | Existing expanded Kaggle `shanmuk4622/tire-dataset-prepared-phase2`, read-only |
| Images | 570 total: 386 train, 81 validation, 103 historical test |
| Known original training tyres by class | Low/mid/high: 1/3/4; lab source groups have separate identity limitations |
| Labels | Low/mid/high mileage proxies, not measured wear, depth, alignment or safety |
| Bank | Same NB10 18 distinct training-photo patches, six/class; no new reference selection |
| Query eligibility | All 386 train images; query's full original source SHA blocks its own reference before augmentation |
| Reference normalization | Multiply each permitted reference by `6/n_allowed` for its reference class; never use query target to choose references |
| Sampling | Uniform class then domain; within domain group weight ∝√n, then image uniform; replacement explicit |
| New acquisition | None; additional observations are correlated draws/views, not additional independent tyres |
| Held-out status | Validation and historical test have no mid targets; prior test inspections make this a development study |

No workstation or completed experiment is changed. The r1 dense branch and entire validation-selected LC prior remain frozen. The new context encoder/attention/residual is the only measured optimization target. Source, objective, bank or recipe changes after starting require compatibility review and a new experiment identity, never a silent resume under an altered implementation.

## Fit and selection

| Setting | Value |
|---|---|
| Candidate budget | One seed, seed 1; 60 epochs; 49 batch-8 updates/epoch = 2,940 updates |
| Draw/view budget | 392 replacement draws/epoch; 23,520 draws and 47,040 paired-view observations |
| Trainable/total parameters | 231,555 / 1,684,988 |
| Optimizer | AdamW lr 0.001, weight decay 0.0001, betas (0.9,0.999), eps 1e-8; foreach/fused disabled |
| Schedule | Cosine over 2,940 completed updates; no warmup |
| Gradients | Global norm clip 1.0; finite checks and every trainable tensor's gradient summary |
| Precision | FP32, disabled scaler still serialized; dense microbatch 1 |
| Two GPUs | GPU0 training; GPU1 immutable-copy evaluation at barriers; one measured model, no extra seed/DDP |
| Validation | Every epoch; observed-ground-truth-class BA, NLL, group/class metrics and complete per-image evidence |
| Selection | Maximum validation observed-class BA, then minimum three-class NLL, then earliest epoch |
| Weights | Keep all 60 epoch weights plus complete completed-update recovery states; no early stopping |
| Endpoints | Validation-selected and final, with all 103 historical endpoint images each |
| Diagnostics | Eight variants at both endpoints; never used to select a checkpoint |

The eight variants are listed in the recipe. Grayscale is representation-invariant. Reverse-band/column variants are deliberate token-layout interventions and preserve the frozen prior; no synthetic photograph or new physical sample is claimed. The frozen initial baseline is already known, so no additional measured baseline fit is needed.

## Recovery, publication and ownership

```mermaid
stateDiagram-v2
    [*] --> VerifyInputs
    VerifyInputs --> Preflight
    Preflight --> RestoreVerifiedState: dual GPU / live HF / storage passed
    RestoreVerifiedState --> TrainUpdate
    TrainUpdate --> AtomicLocalCommit: optimizer completed
    AtomicLocalCommit --> TrainUpdate: next cursor
    AtomicLocalCommit --> ImmutableSnapshot: 30 min / major stage / catchable stop
    ImmutableSnapshot --> HFCommit: deduplicate / bounded batches / CAS owner
    HFCommit --> TrainUpdate: verified revision
    TrainUpdate --> RestoreVerifiedState: restart / replay incomplete update
    TrainUpdate --> Complete: selected + final + diagnostics + report
    Complete --> [*]: verified publication / owner released
```

Native recovery includes model, optimizer, scheduler, disabled scaler, Python/NumPy/CPU/CUDA RNG, deterministic epoch orders, next draw cursor, scientific CSV row cursors, partial metrics, artifact registry, configuration and environment/source/data hashes. Atomic CSV/native cursor checks roll back uncommitted suffixes; an incomplete update is replayed, not credited as complete. Rerunning unchanged restores verified progress and skips completed work.

HF target is dataset repository `Shanmuk4622/tyre-wear-study` under **`phase3/tread-two/atlas-context-r3/fit/<contract>`**. Engineering preflight has a separate namespace. The selected LC checkpoint is fetched from its original pinned native object and hash/size/protocol/epoch verified. Kaggle `HF_TOKEN` Secret is the only write credential source. No local credential probing or `.env` fallback.

Local commits occur every completed optimizer update. Normal remote snapshots occur every 30 minutes, after major stages, and on catchable Stop/error/SIGINT/SIGTERM. Stop triggers a flush attempt subject to server Retry-After/reset backoff. Forced kernel death cannot run cleanup or publish unsent updates. Remote recovery can only guarantee the last verified publication; preserve local files while a flush is pending.

The inherited transport has five-minute request buckets, a separate hourly commit guard, shared local accounting, authenticated/cached reads, unchanged-object deduplication, bounded 80-object commits, CAS revisions, owner fencing and ambiguous-success reconciliation. These address the user's historical approximately 128/hour concern without presenting it as a universal HF quota. Server limits still govern. No live HF success is claimed from offline fixtures.

## Storage planning and retention

Local real-model fixture extrapolation projects **14,372,641,128 bytes (14.37 GB)** conservative peak against a **16,000,000,000-byte** logical guard within the assumed 20 GB output area. This includes projected direct CSV/native artifacts, two-copy snapshot allowance and a 2 GB staging reserve. The runtime separately requires 2 GiB of free space. Native epoch weights, image maps and dense caches are scaled separately; an epoch checkpoint is not multiplied by the endpoint image count. Full selected/final attention is budgeted as 1,687,552 direct CSV values/rows.

This is a planning estimate, not guaranteed capacity. Actual Kaggle preflight remeasures fixture bytes/free disk and must pass before fitting. Live capacity guards remain active; extra 1 TB scratch is not assumed. Dataset inputs are read directly from the attached Kaggle mount. Do not solve capacity pressure by dropping promised scientific evidence or ZIP compression. Save a catchable failure and pending publication details for review.

## Acceptance after actual execution

1. Preserve executed NB13 and compare all 33 embedded files against delivery.
2. Pin final HF revision/contract; verify all direct object bytes/SHA, 27 schemas/counts and native recovery/owner state.
3. Confirm all 60 epochs / 2,940 updates, all 386 exposed queries, reference self exclusion and no frozen-base drift.
4. Reconstruct paired objective including context penalty, base+correction logits, attention summaries and selected/final mass conservation.
5. Review validation selection, low recall, NLL/Brier/ECE, group failures, context magnitudes, appearance/layout probes and identity/artifact shortcuts. Report failures and unavailable mid recall explicitly.
6. Decide another saved-data review or closure from evidence. No automatic workstation promotion, extra seed, test-picked checkpoint or external acquisition.

[Kaggle instructions](tread_two/context_pilot/README.md) · [Schema](phase3TreadTwoContextEvidenceSchema.md) · [Local proof and limits](phase3TreadTwoContextValidation.md).
''')

schema_intro=r'''
# ATLAS-CONTEXT r3 — direct CSV and native evidence dictionary

**Freeze before NB13 measured training.** More telemetry is a standing requirement. This revision keeps every prior scientific family and adds context corrections, attention summaries and complete endpoint attention. [Executable schema](tread_two/context_pilot/schemas.py) is the exact column authority. Tables are published directly on HF, never hidden inside ZIP/NPZ/database artifacts.

## Keys, units and planned observations

`run` is the contract's 12-character prefix; the full immutable SHA is in CONTRACT.json. `sample` joins predictions/support/affinities/corrections/attention; `image_index` joins the source registry and full source SHA. `parameter_update` identifies the completed model state used for an observation. Train predictions are pre-update. Epoch weights/native states supply the numeric model state. Timings/resources vary across recovery and are excluded only from deterministic-equality tests.

| Observation source | Count |
|---|---:|
| Training paired views | 47,040 |
| Per-epoch validation | 4,860 |
| Selected + final identity endpoint | 206 |
| Eight diagnostic variants × two endpoints × 103 images | 1,648 |
| **Total planned raw observations** | **53,754** |
| Paired draws | 23,520 |
| Endpoint attention entries: 206 × 2 layers × 4 heads × 32 queries × 32 keys | 1,687,552 |

Logits, signed evidence and corrections use logit units. Affinity/attention/probability/mass are dimensionless. Dense coordinates use declared resized image/atlas systems; boxes are half-open. Point errors are image-width fractions, not physical distance. Blank metrics denote unavailable denominators; zero remains zero. Mid recall in held-out partitions is unavailable.

## Families and complete columns

'''
descriptions={
'source_images':'570-image identity, role, domain, group, hash, point kind and bank/query registry.',
'bank_cases':'Literal reference provenance and risk/review policy.', 'bank_pixels':'Every reference RGB value, long format.',
'predictions':'Every observation: three logits/scores, loss, uncertainty, prior anchor output, identity, support, variant, checkpoint and permissions.',
'sample_exposures':'Every sampled view: exact probability, factors, seed, group and occurrence; distinguish paired views from draws.',
'affinities':'All 32×18 patch/reference cosines and affinities per observation, including blocked references; join case_rules before accounting.',
'patch_support':'Coverage and exact pooling mass of each tile.', 'patch_evidence':'Combined frozen-reference plus context signed class contributions, 32×3 per observation.',
'coefficients':'Frozen signed reference coefficients/temperature plus combined base+context bias at recorded parameter updates.',
'reference_embeddings':'Full frozen reference embeddings at retained evaluation/update endpoints.',
'train_updates':'Loss components, lr, clipping/finite status, frozen-base gradient count, learned-state hashes and bounded timing/resource counters.',
'parameter_gradients':'Every trainable tensor: count/norm/mean/min/max/finite; not full gradient tensors.',
'dense_pixels':'Frozen dense pixel probabilities on the declared audit grid; native full maps retained.',
'atlas_rows':'Native-to-atlas geometry/support provenance.', 'dense_regions':'Mask/box intersections, counts, coordinates and metrics.',
'dense_points':'Point predictions/truth/kind, errors and unavailable reasons.',
'metrics':'Cohort/domain/group/class summaries with denominators and unavailable reasons.',
'confusion':'Complete class confusion counts at reported levels.', 'calibration_bins':'Ten equal-width confidence bins with sufficient statistics.',
'selection':'Every validation checkpoint decision, BA/NLL/tie rule and native artifact.',
'explanation_checks':'Selected/final class-wise signed mass conservation and unsupported mass.',
'case_rules':'All 18 full-source permissions/reference-class normalization scales per observation.',
'view_pairs':'Per-draw pair loss components, disagreement and both observation keys; includes residual_l2.',
'view_patch_consistency':'Each pair/tile contextual cosine distance and pooling mass.',
'context_evidence':'Every observation/tile: source-patch mapping and three weighted spatial logit corrections.',
'context_summary':'Every observation: base prediction/NLL/three logits, three bias corrections, total correction squared and eight attention entropy/max pairs.',
'endpoint_attention':'Full selected/final identity attention: layer/head/query/key/value, including unsupported query rows; normalized valid-key support.'}
lines=[schema_intro, f'**{len(SCHEMAS)} families / {sum(len(c) for c in SCHEMAS.values())} declared columns** (column names repeat across families).\n']
for name,columns in SCHEMAS.items():
    lines.append(f'### `{name}` — {len(columns)} columns\n\n{descriptions[name]}\n\n`'+ '`, `'.join(columns)+'`\n')
lines.append(r'''
## Reconstructing the added measurements

Join each context_evidence row to predictions/sample and patch_support. Deltas already include pooling mass; do not multiply by mass again. Combined patch_evidence minus context_evidence gives the frozen case-branch contribution. Subtract context_summary.bias_delta from coefficients' combined bias to recover base bias. Sum patches plus combined bias to reproduce each logit. The independently checked engineering maximum residual is 8.28×10⁻⁸.

`correction_l2` is mean over three classes of `(sum_patch(delta)+bias_delta)²`. A pair's `residual_l2` is the mean of its two view values. Add `0.02*residual_l2` to paired CE/JS/contextual-distance terms. Context attention matrices have shape [2,4,32,32]. Summaries weight each query row by pooling mass; entropy is divided by `log(max(2, valid_key_count))`. Empty support has zero weighted summaries. Full matrices include all query rows even if their output has zero pooling mass.

## Native state, visuals and explicit limits

| Retained directly | Purpose |
|---|---|
| Native completed-update state | Exact model/optimizer/scheduler/scaler/RNG/order/cursor/partial-stat recovery |
| All 60 native epoch weights | Preserve full training trajectory without refitting |
| Selected/final native image captures | Base/context embeddings, affinities, combined evidence, correction, bias, full attention, atlas grid, original dense outputs and signed projections |
| Immutable JSON/source manifests | Configuration, source/data/environment/recipe IDs, schemas, original identities, object hashes and row cursors |
| Lifecycle/HTTP records | Stop/errors, retry/backoff, publication/ownership/revisions, pending progress |
| PNG + interactive HTML | Curves; endpoint/error/source filtering; literal reference cards; signed 32-tile evidence and explicit frozen-base/context accounting |

**Declared gaps:** no full per-update attention matrices, no full per-update activations/embeddings/gradient tensors, no complete profiler trace or power telemetry, no genuine physical wear/depth/alignment/safety targets, no new independent tyre cohort and no held-out mid targets. Attention summaries are retained every observation; full attention is retained at both identity endpoints. Engineering fixture resource readings are not measured-fit timings. Future derived summaries must not invent unsaved values.

The reporter only reads retained state and marks engineering examples prominently. Reference thumbnails explain the frozen case branch; context evidence must accompany any complete prediction explanation. Conservation does not establish causality or clinical/physical usefulness. All data/protocol/cohort caveats remain attached to exported evidence.

[CSV-only verification](tread_two/context_validation/CSV_ACCOUNTING.json) · [Storage policy](phase3TreadTwoContextProtocol.md) · [Run instructions](tread_two/context_pilot/README.md).
''')
write('phase3TreadTwoContextEvidenceSchema.md','\n'.join(lines))

write('phase3TreadTwoContextValidation.md',f'''
# ATLAS-CONTEXT r3 — engineering verification and pending checks

**Local engineering checks passed. No measured context accuracy exists yet.** Every Python check ran through `phase3/run_cv.cmd` in `cv_conda`, without replacing CUDA/Torch or dependencies. Actual dual T4/live-HF preflight remains required on Kaggle.

| Check | Actual evidence | Boundary |
|---|---|---|
| [Layout premise](tread_two/context_validation/LAYOUT_PROOF.json) | Eight known training tyres, 16 rearrangements; max logit delta 5.96×10⁻⁸ | No proof layout explains endpoint errors |
| [Model math](tread_two/context_validation/MATH_VALIDATION.json) | 231,555 trainable / 1,684,988 total; initial logits/evidence bit-equal to LC; frozen base unchanged after engineering update; finite gradients/empty support | Disposable checks, not a measured fit |
| Context expressiveness | Nonzero disposable-engineering band-order response: 0.0002171993 logit units | Shows ability to respond, not learned useful recognition |
| [Recovery and storage](tread_two/context_validation/LOCAL_VALIDATION.json) | Four actual batch-8 updates; uninterrupted, paused, abrupt-native rollback and fresh-process snapshot restore agree | Single local CUDA device; zero network requests |
| Native equality | Model, optimizer, scheduler, scaler, RNG, all deterministic orders/cursors and nine native artifacts agree | Timings/resources intentionally not equal |
| CSV equality | All 27 scientific families, including context and endpoint attention, agree | Engineering fixture contains 104 observations |
| [CSV-only accounting](tread_two/context_validation/CSV_ACCOUNTING.json) | 312 logits; 32 pair losses; four update losses; 32,768 attention values reconstructed | Zero model forwards or optimizer updates in this audit |
| Accounting precision | Max logit residual {account['maximum_logit_residual']:.3g}; max pair loss residual {account['maximum_pair_loss_residual']:.3g}; max attention row-sum error {account['maximum_attention_rowsum_residual']:.3g} | Retained FP32/CSV precision |
| [Fresh notebook package](tread_two/context_validation/PACKAGE_VALIDATION.json) | 33 payload files extracted; actual one-update worker matches reference native state and all 27 CSVs; 16 view rows | Kaggle execution not claimed |
| Interactive controls | Actual generated report JavaScript tested with a synthetic DOM: endpoint/error/empty/source filters, four cases, 18 references, separate base/correction fields | No native browser rendering claim |
| Capacity estimate | {local['actual_model_proof']['storage_projection']['conservative_peak_output_bytes']:,} bytes conservative peak < 16 GB guard | Actual Kaggle bytes/free disk remeasured automatically |

The 17 offline recovery/publication checks cover four transaction fault boundaries, CSV rollback/tamper rejection, source-contract drift, native/snapshot recovery without archives/databases, immutable snapshot isolation, remote fast-forward, bounded 80-object batches, ambiguous success, pinned-LFS hash verification, deduplication, owner fencing/release and shared budget/server-delay behavior. These are transport fixtures, not live-HF success claims.

## Scope of engineering examples

Recovery uses two validation examples as explicitly labeled fixtures and a two-epoch/two-update-per-epoch recipe. These engineering outputs are isolated from the future measured namespace and do not enter checkpoint selection or accuracy reporting. Train-only math/layout checks use existing cached train packets. The delivery is a single new measured candidate; no old fit or extra measured seed was run locally.

The preflight is intentionally short and disposable. Its full attention matrix accounting, frozen-base assertions and raw-byte extrapolation are exercised before a long run. No test expectation requires improved class accuracy.

## Deliverable and preservation

- [NB13](notebooks/phase3_NB13_TREAD_TWO_ATLAS_CONTEXT_Resumable_Pilot.ipynb): five code cells, 33 embedded files, notebook SHA `{package['notebook_sha256']}`.
- [Frozen source inventory](tread_two/context_pilot/FILES.json): 32 source/asset entries; FILES.json is the 33rd notebook payload file.
- Executed NB08–NB12, v1/r2/gate/review sources, both workstations, earlier paired/r1 runtimes and original Markdown baselines are preservation targets in the final handoff check.
- Git ignores large fixtures/native outputs/HF mirrors and keeps only four explicitly allowed new reference-input arrays, small receipts, current notebook and curated proof figure.

## Pending actual execution

The user must run the new notebook on Kaggle T4×2 with the existing expanded input, Internet and HF_TOKEN. Only that run can establish actual dual-device/live-HF recovery, real storage footprint, final model metrics, learned layout response and final ownership release. Save the executed notebook. Then audit the pinned direct publication before another experiment or workstation decision.

[Architecture](phase3TreadTwoContextArchitecture.md) · [Protocol](phase3TreadTwoContextProtocol.md) · [Evidence dictionary](phase3TreadTwoContextEvidenceSchema.md) · [Run guide](tread_two/context_pilot/README.md).
''')

write('tread_two/context_pilot/README.md',r'''
# NB13 / ATLAS-CONTEXT r3 — Kaggle run and recovery guide

**One new candidate; measured training has not started locally.** NB12 is already HF-verified and needs no rerun. This separate model freezes selected LC epoch 7 and learns a position-aware correction. Its accuracy is unknown. Existing tyre data only; workstation work remains deferred.

## Settings and execution

1. Upload [phase3_NB13_TREAD_TWO_ATLAS_CONTEXT_Resumable_Pilot.ipynb](../../notebooks/phase3_NB13_TREAD_TWO_ATLAS_CONTEXT_Resumable_Pilot.ipynb) to Kaggle.
2. Select **GPU T4 ×2** and enable **Internet**.
3. Attach exactly one **expanded** `shanmuk4622/tire-dataset-prepared-phase2` dataset directly from Kaggle. Keep the input read-only; do not copy its images into working output or attach a ZIP instead.
4. Enable Kaggle Secret **HF_TOKEN** with write access to `Shanmuk4622/tyre-wear-study`. Do not paste or print the token.
5. Begin with at least **12 GiB free** working output. Preflight must also pass its measured peak estimate below **16 GB** and the actual free-space guards. The assumed output budget is 20 GB; extra scratch is not assumed.
6. Choose **Run All**, in order. Do not bypass failed input/recovery/storage checks. Keep the executed notebook after completion.

| Code cell | Work | What to expect |
|---|---|---|
| 1 | Verify/extract 33 embedded files; establish launcher | Source compatibility checks; no measured fitting |
| 2 | `prepare` | Verify data/roles/references, devices, upstream dense and exact selected LC weights; create immutable contract |
| 3 | `preflight` | Offline transaction tests + actual dual-T4/live-HF restoration fixture + measured capacity projection |
| 4 | `train` | Restore or fit one seed for 60 epochs / 2,940 updates, select by validation, final/selected evaluation, eight diagnostics, publish report |
| 5 | Inspect receipt/curves/explorer | Display last verified remote revision, selected/final report and interactive HTML link |

GPU0 fits; GPU1 evaluates an immutable copy at barriers. This is one FP32 model, batch 8 with dense microbatch 1. The notebook pins `huggingface_hub==0.36.2` only; it does not install Torch/CUDA. Source/assets are embedded, so a separate code dataset upload is unnecessary. Upstream weights are downloaded once from pinned HF objects and SHA/size/protocol checked. No local workstation dependency is required.

## Resume and stopping

Work directory: `/kaggle/working/tread_two_atlas_context_r3`. Normal local commits happen after each completed optimizer update; HF flushes normally every 30 minutes and at major stages. Use Kaggle Stop while keeping the session/files alive: the launcher forwards SIGINT, waits for atomic state and attempts an immediate flush subject to HF server delay. Forced kernel/session death cannot publish unsent updates.

Restart by using this **unchanged** notebook with the same dataset/settings and Run All. It reconstructs the exact frozen sources, restores a verified compatible remote/native state, replays an incomplete update and resumes the next completed-update cursor. It never silently restarts a whole completed epoch. Completed compatible preflight/fit stages are verified and skipped. A different environment/device/source contract is not silently treated as the old run.

Target: `Shanmuk4622/tyre-wear-study`, namespace `phase3/tread-two/atlas-context-r3/fit/<contract>`. Engineering preflight is separate. Never run simultaneous writers to the same contract or delete/override a live owner lease. Keep LAST_REMOTE.json, lifecycle/HTTP logs, `run/console.log`, native states and CSVs while a publication is pending. All science is published as direct CSV/native/JSON/code/PNG/HTML files; no ZIP/NPZ/database uploads.

## Troubleshooting

| Symptom | Action |
|---|---|
| Last training cell returns normally after report/owner release | Expected completion; inspect receipt, save executed notebook; no restart needed |
| No valid prepared input or checksum mismatch | Attach the declared expanded Kaggle version exactly once; preserve error details; do not edit checksums to continue |
| Missing HF_TOKEN / permission failure | Enable the authorized Secret and repository write access; never paste a credential in code/output |
| Rate limit / Retry-After / commit delay | Keep session alive; inherited shared budget/backoff/reconciliation handles retries; avoid manual repeated pushes |
| Active owner / stale writer rejection | Preserve identity and logs; do not start another writer or erase ownership records |
| Source/environment/contract mismatch | Preserve old state and stop dependent fitting; report the mismatch for a compatibility review |
| Storage projection or free-space guard fails | Preserve state/receipt; report actual bytes; do not drop CSVs, disable guards or assume extra scratch |
| Unexpected stop with no traceback | Inspect completion phase, LAST_REMOTE.json and published manifest before assuming failure; completed work skips fitting |
| Exception/forced loss | Keep traceback and executed notebook; recovery starts from last verified publication, plus compatible surviving local completed updates |

## Outputs and review

Expect all 60 epoch weights, completed-update recovery, 27 direct scientific CSV families, 53,754 planned observations, full selected/final endpoint attention and native image evidence. Per-observation attention summaries are always retained; full per-update attention/activation/gradient tensors are explicitly not retained. The explorer filters selected/final cases, errors, empty support and source identity, and separates frozen LC logits from context correction. Reference cards explain only the frozen case branch.

After the notebook finishes, return the executed NB13 for independent pinned-HF verification. No improved accuracy, model promotion or workstation change is claimed beforehand. Low/mid/high are mileage proxies; no physical wear/depth/safety conclusion is supported.

## Source and resource map

| Files | Responsibility |
|---|---|
| `RECIPE.json`, `BASE_INPUT.json`, `FILES.json`, `spec.py` | Frozen intent, exact LC input, source bytes/hashes and environment-specific contract |
| `model_atlas.py`, `frozen_lc.py`, `frozen_anchor.py`, `lc_features.py` | New spatial correction; preserved LC/dense calculations; exact local-contrast transform |
| `data_atlas.py`, `legacy_data.py`, `phase2_training_data.py`, `geometry.py` | Prepared inputs, immutable splits, deterministic replacement draws, source SHA and atlas coordinates |
| `assets/*.npy`, `assets/prototype_bank.csv`, `assets/sampling_plan.csv` | Literal frozen reference inputs and query/group provenance; these are not trained context weights |
| `objective.py`, `worker.py` | Paired loss, one completed update, validation/selection/endpoints, diagnostics and frozen-base assertions |
| `schemas.py`, `telemetry.py` | Every direct CSV column, prediction/evidence/attention accounting and metric denominators |
| `store.py`, `common.py` | Atomic CSV/native recovery, hashes, RNG and capacity guards |
| `coordinator.py` | Kaggle stages, child stop forwarding, ownership, 30-minute/major/stop snapshot scheduling |
| `publisher.py`, `evidence_hub.py`, `rate_budget.py` | Direct HF transport, CAS snapshots, retry/backoff, request accounting and fencing |
| `acceptance.py`, `report.py` | Real-model disposable recovery/storage gate; saved-evidence curves and interactive explorer |
| `UPSTREAM.json`, `VENDORED.json`, `LC_RULES.json` | Inherited provenance and fixed representation/reference rules |

The 32-file inventory covers runtime/assets, not this explanatory README. Do not run scaffold scripts over an active/completed experiment. The new source folder is isolated from both completed models and the workstations.

[Architecture](../../phase3TreadTwoContextArchitecture.md) · [Frozen protocol](../../phase3TreadTwoContextProtocol.md) · [Complete CSV dictionary](../../phase3TreadTwoContextEvidenceSchema.md) · [Local checks](../../phase3TreadTwoContextValidation.md) · [NB12 findings](../../phase3NB12Completion.md).
''')

current='''
<!-- NB12_CONTEXT_UPDATE_20261030 -->
## Current — NB12 verified; NB13 context pilot ready

**30 October 2026. NB12 is complete and independently HF-verified:** all 63 direct files, 40 CSVs / 380,595 rows, four figures and 1,442 review cases checked. All 1,350 original NB11 manifest entries are unchanged; ownership is released. No NB12 or earlier fit rerun is needed. [Completion and insights](phase3NB12Completion.md).

Appearance stability improved, but selected/final LC endpoint BA remains 50% with low recall 0/62; selected NLL improved and ECE worsened. A train-only layout check confirms that the old shared-local/equal-band pooling ignores band/column reordering. This is a structural limitation, not proof of the cause of those errors.

**Next: [NB13 / ATLAS-CONTEXT r3](notebooks/phase3_NB13_TREAD_TWO_ATLAS_CONTEXT_Resumable_Pilot.ipynb).** One new seed-1/60-epoch candidate freezes selected LC epoch 7 and learns a bounded position-aware correction. Initial logits match LC exactly. Local math/recovery/fresh-package/CSV accounting pass; actual T4×2/live-HF preflight, measured fit and accuracy remain pending. Use T4×2, Internet, existing expanded Kaggle input and HF_TOKEN, then Run All. [Run guide](tread_two/context_pilot/README.md).

Keep all 27 CSV families, native recovery/all epoch weights, separate base/context evidence and full selected/final attention. Per-update matrices are not retained; per-observation summaries are. Existing tyre photos only, no workstation changes or completed-source rewrites. Earlier dated pending instructions below are historical and superseded by this block.

[Architecture](phase3TreadTwoContextArchitecture.md) · [Protocol](phase3TreadTwoContextProtocol.md) · [Schema](phase3TreadTwoContextEvidenceSchema.md) · [Engineering proof](phase3TreadTwoContextValidation.md).
<!-- END_NB12_CONTEXT_UPDATE -->
'''
import re
def prepend_current(name,body):
    p=ROOT/name;s=p.read_text(encoding='utf-8')
    s=re.sub(r'\n<!-- NB12_CONTEXT_UPDATE_20261030 -->.*?<!-- END_NB12_CONTEXT_UPDATE -->\n','\n',s,flags=re.S)
    s=re.sub(r'\n<!-- NB11_COMPLETE_UPDATE_20261030 -->.*?<!-- END_NB11_COMPLETE_UPDATE -->\n','\n',s,flags=re.S)
    first,rest=s.split('\n',1);p.write_text(first+'\n\n'+body.strip()+'\n\n'+rest.lstrip(),encoding='utf-8')
for name in ['README.md','AGENTS.md','phase3Progress.md','phase3NextSteps.md','phase3OpenItems.md','phase3Understanding.md','phase3ModelRationaleAndLogging.md','phase3DataRetentionPolicy.md','phase3TreadTwoFindings.md']:
    prepend_current(name,current)

evidence='''
<!-- NB12_CONTEXT_UPDATE_20261030 -->
## NB12 closed / r3 evidence map — 30 October 2026

| Record | Verified scope |
|---|---|
| [NB12 completion](phase3NB12Completion.md) | Executed notebook, 63 direct files, 40 CSVs / 380,595 rows, owner released |
| [Pinned publication audit](tread_two/nb12_completion/036132254087/VERIFICATION.json) | Every object byte/SHA; exact table agreement; original 1,350-entry fit unchanged |
| [Layout diagnostic](tread_two/context_validation/LAYOUT_PROOF.json) | Eight train tyres / 16 interventions, no fitting/held-out forwards |
| [r3 math](tread_two/context_validation/MATH_VALIDATION.json) | Zero-initialized identity, frozen base, finite/empty support and layout expressiveness |
| [r3 recovery](tread_two/context_validation/LOCAL_VALIDATION.json) | All 27 CSV/native state equality, four batch-8 updates, 17 offline checks and 14.37 GB planning estimate |
| [CSV-only accounting](tread_two/context_validation/CSV_ACCOUNTING.json) | 104 observations / 312 logits / 32 pairs / 32,768 attention values; zero forwards |
| [NB13 package](tread_two/context_validation/PACKAGE_VALIDATION.json) | 33 extracted files, one actual update; all scientific/native state match |
| [Source inventory](tread_two/context_pilot/FILES.json) | 32 frozen inputs; new namespace, earlier studies unchanged |

Actual NB13 dual T4/live-HF/fit/accuracy are pending. Engineering receipts cannot substitute for them. Earlier pending NB12 notes below are historical.
<!-- END_NB12_CONTEXT_UPDATE -->
'''
prepend_current('phase3EvidenceRegister.md',evidence)
inventory='''
<!-- NB12_CONTEXT_UPDATE_20261030 -->
## NB12 completion / ATLAS-CONTEXT documentation — 30 October 2026

| Document | Purpose |
|---|---|
| [NB12 completion](phase3NB12Completion.md) | Actual pinned-HF proof, findings, figures and why the next hypothesis changes |
| [Context architecture](phase3TreadTwoContextArchitecture.md) | Frozen prior, tokens/positions/attention, bounds, objective and explanation accounting |
| [Context protocol](phase3TreadTwoContextProtocol.md) | Existing cohort, exact fit/selection/recovery/publication and capacity boundaries |
| [Context evidence dictionary](phase3TreadTwoContextEvidenceSchema.md) | All 27 families and every column, units, joins, cadence and declared gaps |
| [Context validation](phase3TreadTwoContextValidation.md) | Actual local tests, numerical residuals, preserved studies and pending Kaggle checks |
| [NB13 run guide](tread_two/context_pilot/README.md) | T4×2/input/Secret/cells, resume, stopping, output and troubleshooting |

Current action is NB13, not a repeat of NB08–NB12. Earlier dated pending statements are historical. Git policy now includes ten small reference-input arrays in three frozen packages; new large context fixtures stay ignored.
<!-- END_NB12_CONTEXT_UPDATE -->
'''
prepend_current('phase3DocumentationInventory.md',inventory)
prepend_current('phase3RepositoryMap.md',inventory)
p=ROOT/'phase3NextSteps.md';s=p.read_text(encoding='utf-8');s=s.replace(s.split('\n',1)[0],'# Next steps — run the isolated ATLAS-CONTEXT pilot',1);p.write_text(s,encoding='utf-8')
extra=current.replace('(phase3','(../phase3').replace('(notebooks/','(../notebooks/').replace('(tread_two/','(')
prepend_current('tread_two/README.md',extra)

def append_once(name,marker,body):
    p=ROOT/name;s=p.read_text(encoding='utf-8');s=re.sub(r'\n<!-- '+marker+r' -->.*?<!-- END_'+marker+r' -->\n','\n',s,flags=re.S)
    p.write_text(s.rstrip()+'\n\n<!-- '+marker+' -->\n'+textwrap.dedent(body).strip()+'\n<!-- END_'+marker+' -->\n',encoding='utf-8')
append_once('phase3Decisions.md','NB12_R3_DECISIONS','''
## NB12 closure and isolated spatial-context candidate — 30 October 2026

| ID | Decision | Evidence / constraint |
|---|---|---|
| P3-D095 | Close NB12, preserve executed payload and original fit | All 63 direct files/40 tables match; owner released; all 1,350 fit entries unchanged |
| P3-D096 | Treat stable probes and lower NLL as narrower findings | Endpoint low recall remains 0/62 and selected ECE worsens; no model promotion |
| P3-D097 | Test a separate position-aware residual, not another old-fit seed | Training-only proof exposes order invariance; not proof of the cause of endpoint failure |
| P3-D098 | Freeze original validation-selected LC epoch 7; zero-initialize bounded context head | Exact native SHA pinned; initial predictions/evidence match; base remains unchanged |
| P3-D099 | Retain 27 CSV families and full selected/final attention | Every view gets corrections and attention summaries; full per-update matrices explicitly omitted |
| P3-D100 | Deliver NB13 after local math/recovery/package/CSV tests | Actual T4×2/live-HF/storage/accuracy pending Kaggle; one new candidate only |
| P3-D101 | Preserve Git hygiene alongside new source inputs | Four narrow new NPY exceptions; fixtures/native/HF mirrors ignored; no commit/push |

[Findings](phase3NB12Completion.md), [architecture](phase3TreadTwoContextArchitecture.md), [protocol](phase3TreadTwoContextProtocol.md).
''')
append_once('phase3SessionLog.md','NB12_R3_SESSION','''
## 30 October 2026 — NB12 independently verified; NB13 implemented

User reported completed NB12 and authorized verification/documentation/next work. Archived executed notebook 036132254087… without regeneration; all four code cells and 39 payload files pass. Public HF revision 77d03303… contains 63 direct files, 40 CSVs/380,595 rows, four figures and 1,442 cases. Verified every payload byte/size/SHA and exact retained-precision tables; original NB11 1,350 manifest entries unchanged, owner released. No local credentials or remote writes.

Review interpretation: stable appearance probes, better selected NLL/Brier, worse ECE and unchanged low→mid endpoint failures. No promotion. Train-only structural check used eight known tyres, 24 cached head calls/16 permutations and zero optimizer/held-out/dense forwards; confirms pooling order invariance only.

Implemented isolated context r3 from selected LC epoch7: frozen full prior; warm-start trainable patch encoder; fixed 2D positions/coverage; two 96D four-head attention blocks; zero-initialized bounded spatial/global correction. One new seed/60epoch budget; same existing dataset/sampler/recovery/transport. Added three CSV families, 53,754 planned observations, full selected/final attention and every-observation summaries/corrections. All 27 CSV families and exact native state pass real four-update interruption/recovery checks. CSV-only audit reconstructs 312 logits,32 paired losses and32,768 attention entries; extracted notebook one-update comparison passes. Storage planning peak14.37GB; actual Kaggle guard remains required.

Delivered NB13 and six substantive Markdown documents with diagrams, full tables, exact pins, findings/visuals, schema/gaps, runtime ownership and run/resume/troubleshooting. Updated central indexes, decisions, retention, environment/scope context and Git policy. Existing photos only; both workstations, completed source packages/executed notebooks and original documentation baselines preserved. Local math/disposable engineering updates are not a measured context run. Actual T4×2/live-HF/60epoch metrics pending; next action is unchanged NB13 Run All on Kaggle, then pinned publication audit.
''')

for name in ['phase3TreadTwoLCArchitecture.md','phase3TreadTwoLCProtocol.md','phase3TreadTwoLCValidation.md','phase3NB11Completion.md']:
    p=ROOT/name;s=p.read_text(encoding='utf-8');first,rest=s.split('\n',1)
    note='> Current status, 30 October 2026: NB11 and NB12 are complete/HF-verified. This document records the preserved LC r2 study. The new separately frozen candidate is [ATLAS-CONTEXT r3](phase3TreadTwoContextArchitecture.md); [NB12 completion](phase3NB12Completion.md) records actual publication and interpretation.\n'
    if note not in s:p.write_text(first+'\n\n'+note+'\n'+rest.lstrip(),encoding='utf-8')
for name in ['tread_two/lc_pilot/README.md','tread_two/lc_review/README.md']:
    p=ROOT/name;s=p.read_text(encoding='utf-8');first,rest=s.split('\n',1)
    note='> Completed-study guide: NB11 and NB12 are HF-verified. No rerun is needed. See [NB12 findings](../../phase3NB12Completion.md) and the new [NB13 guide](../context_pilot/README.md). Earlier run instructions below describe the preserved completed delivery.\n'
    if note not in s:p.write_text(first+'\n\n'+note+'\n'+rest.lstrip(),encoding='utf-8')

# Keep the existing user's Git preparation and all seven index-only removals intact.
for p in [ROOT/'AGENTS.md',ROOT/'phase3DocumentationInventory.md',ROOT.parent/'docs/GIT_PUSH_CHECK.md']:
    s=p.read_text(encoding='utf-8').replace('Six small','Ten small').replace('Six explicitly','Ten explicitly').replace('six deliberate','ten deliberate').replace('six frozen arrays','ten frozen arrays').replace('The six arrays','The ten arrays').replace('20 ignored / 20 retained','22 ignored / 27 retained').replace('20 ignored and 20 retained','22 ignored and 27 retained')
    if p.name=='GIT_PUSH_CHECK.md':s=s.replace('`phase3/tread_two/lc_pilot/assets/` are deliberate','`phase3/tread_two/lc_pilot/assets/` and\n`phase3/tread_two/context_pilot/assets/` are deliberate')
    p.write_text(s,encoding='utf-8')

atomic(OUT/'DOCUMENTATION.json',dict(status='written',new_documents=6,scientific_families=len(SCHEMAS),declared_columns=sum(len(c) for c in SCHEMAS.values()),figures=['figures/layout_invariance.png'],measured_accuracy=None))
print('Wrote NB12 closure, context architecture/protocol/schema/validation/run guide, visual proof and current trackers.')
