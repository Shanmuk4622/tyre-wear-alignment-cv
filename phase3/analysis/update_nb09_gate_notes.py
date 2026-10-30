"""Synchronize current NB09 completion and separately completed local NB10 gate."""
from pathlib import Path
import sys
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
MARK='<!-- NB09_GATE_UPDATE_20261030 -->'
blocks={
'README.md':'''**Latest, 30 October 2026 — NB09 complete / HF-verified; next gate implemented.**
All four NB09 cells and all 51 direct HF files pass the
[completion audit](phase3NB09Completion.md); no rerun is needed.
The [training-only gate](phase3TreadTwoAppearanceGate.md) is implemented and
locally complete on all 386 training images: **1,918,052 scientific CSV rows**,
full native image state and an 18-distinct-photo UNTRAINED proposal.
**Next: [NB10 on T4×2](notebooks/phase3_NB10_TREAD_TWO_Train_Only_Appearance_And_Reference_Gate.ipynb)**
with the existing expanded Kaggle dataset, Internet and HF_TOKEN, then Run All.
[Settings/recovery](tread_two/appearance_gate/README.md). Actual dual-T4/HF gate
execution remains pending; no new fitting or workstation change occurred.
Earlier pending-NB09 statements below are historical local-delivery records.''',
'phase3Progress.md':'''## Latest — NB09 verified; full local appearance gate / NB10 delivered

**30 October 2026.** Preserved the supplied executed NB09; all four cells and
30 embedded files match delivery. Independently checked all 50 published
objects plus manifest, all 30 CSVs / 246,157 rows, exact retained-precision
agreement and released owner. [Completion](phase3NB09Completion.md).
The updated executed NB08 is also preserved: its original 25-file payload and
1,177 published scientific objects remain unchanged; administrative restore
history is distinguished explicitly.

Completed the next gate locally: 386 training images, seven probes / 2,702
predictions, eight CSV families / 1,918,052 rows, all native packets and exact
score reconstruction. An 18-photo proposal passes structural coverage but
leaves two high groups with one query photo. Local-contrast features are less
brightness-sensitive on the declared unclipped subset, not a trained classifier.
[Findings/visuals](phase3TreadTwoAppearanceGate.md). Recovery and packaged-worker
checks pass. Next: actual NB10 dual-T4/HF execution. No fit, new tyre photo,
held-out forward or workstation update was added.''',
'phase3Understanding.md':'''**Current, 30 October:** [NB09 is complete and HF-verified](phase3NB09Completion.md).
The [next training-only gate](phase3TreadTwoAppearanceGate.md) is implemented
and fully run locally, with zero new fitting or held-out queries. It probes
the frozen selected v1 classifier and a deterministic texture/reference
proposal; it does not establish a trained v2 classifier. Next is NB10 on Kaggle
for actual dual-T4/direct-HF gate retention. Existing photos only; both
workstations remain unchanged. Older pending-NB09 statements are historical.''',
'AGENTS.md':'''## Latest superseding status — 30 October 2026

- NB09 completed on Kaggle and is independently HF-verified at `93d7ea34…`: 51 direct files, 30 CSVs / 246,157 rows, released owner. No rerun is needed; [completion](phase3NB09Completion.md).
- Preserve the supplied executed NB09 and updated executed NB08. Do not force old clean-delivery hashes onto executed notebooks. NB08's original 25-file payload and 1,177 published scientific objects remain unchanged; distinguish refreshed administrative/history paths from scientific rewrites.
- The training-only appearance/reference gate is implemented and fully executed locally: 386 train images, 1,918,052 scientific rows, zero fitting or held-out forwards. The 18-photo bank is UNTRAINED and must not be installed in v1; two high groups retain only one query photo. Preserve the repetition/artifact caveats.
- Next is NB10: T4×2, existing expanded Kaggle input, Internet/HF_TOKEN. Distinguish the one-GPU local proof from pending actual dual-T4/live-HF gate execution. A future model needs a separately frozen representation, support/query and repetition policy; do not edit the completed fit or workstations. [Gate findings](phase3TreadTwoAppearanceGate.md).''',
'phase3OpenItems.md':'''## Current — NB10 gate publication

| Item | Actual state / next action |
|---|---|
| NB09 | All four cells / 51 HF files verified; no rerun |
| Training-only gate | Complete locally: 386 images, eight families / 1,918,052 rows, native state and an 18-photo UNTRAINED proposal |
| Actual dual-T4/HF gate | Run NB10 with existing expanded input, Internet and HF_TOKEN |
| Next audit | Preserve executed NB10 and verify the new gate namespace |
| V2 classifier | Not fitted; freeze representation, reference ownership and repetition policy together |
| Proposal caveats | Paint/blur/glare remain; two high groups have one query photo; no automatic installation |
| Photos / workstations | Existing data only; both applications preserved |

[NB09 completion](phase3NB09Completion.md), [gate results](phase3TreadTwoAppearanceGate.md).
Pending-NB09 entries below describe the earlier delivery stage.''',
'phase3NextSteps.md':'''**Current, 30 October 2026:** [NB09 is complete and HF-verified](phase3NB09Completion.md).
No NB08/NB09 rerun is needed.

Open [NB10](notebooks/phase3_NB10_TREAD_TWO_Train_Only_Appearance_And_Reference_Gate.ipynb),
select **T4×2**, attach exactly one expanded
**shanmuk4622/tire-dataset-prepared-phase2**, enable **Internet + HF_TOKEN**,
leave at least **4 GiB free** and choose **Run All**.
[Full run/resume guide](tread_two/appearance_gate/README.md).

The gate is implemented and complete locally, retaining 1,918,052 scientific
rows and full native image state. It uses frozen weights and publishes in a
new namespace, with image-unit recovery. Actual dual-T4/HF gate execution
remains pending. [Findings/tradeoffs](phase3TreadTwoAppearanceGate.md) distinguish
brightness-feature consistency from classification, and reference diversity
from remaining query scarcity. After actual gate review, freeze a new recipe
before any measured v2 fit. No new photos or workstation changes are needed.
The previous NB09 publication instructions below are historical.''',
'tread_two/README.md':'''**Latest, 30 October — [NB09 complete and HF-verified](../phase3NB09Completion.md).**
The next [training-only gate](../phase3TreadTwoAppearanceGate.md) is implemented
and locally complete: 386 images, seven probes, 1,918,052 CSV rows, native
state and an 18-photo UNTRAINED proposal. Next: **[NB10 on T4×2](../notebooks/phase3_NB10_TREAD_TWO_Train_Only_Appearance_And_Reference_Gate.ipynb)**
with existing expanded data; [settings/recovery](appearance_gate/README.md).
No fitting, held-out query, extra photo or installation occurred. Actual
dual-T4/HF gate publication remains pending user execution.''',
'tread_two/review/README.md':'''**Completion update, 30 October:** all four NB09 cells and all 51 HF files are
independently verified. [Completion audit](../../phase3NB09Completion.md).
No rerun is needed. Next is the separate [NB10 gate](../appearance_gate/README.md).
Earlier local-only validation descriptions below retain their original scopes.''',
'phase3TreadTwoFindings.md':'''**Completion / next gate, 30 October:** [NB09 is HF-verified](phase3NB09Completion.md).
The proposed training-only gate is now [implemented and locally complete](phase3TreadTwoAppearanceGate.md).
Next: NB10 for actual dual-T4/HF gate retention; it does not fit or promote a
new classifier. Earlier pending-NB09 wording is historical.''',
'phase3NB08CompletionStatus.md':'''**Latest:** NB09's completed review is [HF-verified](phase3NB09Completion.md).
The next [training-only gate](phase3TreadTwoAppearanceGate.md) is implemented
and locally complete. Current executed NB08/NB09 are archived without
regeneration; original source/checkpoint science is preserved.''',
}
for name,block in blocks.items():
    path=ROOT/name;text=path.read_text(encoding='utf-8')
    if MARK in text: continue
    first,rest=text.split('\n',1)
    if name=='phase3NextSteps.md': first='# Next steps — run the training-only gate with NB10'
    text=first+'\n\n'+MARK+'\n'+block+'\n\n'+rest.lstrip('\n')
    if name=='phase3TreadTwoFindings.md':
        text=text.replace('is to run **NB09 on Kaggle CPU** to preserve the review in HF. No T4 training\nor NB08 rerun is needed.',
            'is now to run **NB10 on T4×2** for the implemented training-only gate. NB09\nis complete and HF-verified; no new fitting or NB08 rerun is needed.')
    path.write_text(text,encoding='utf-8',newline='\n')
print('Updated',len(blocks),'current status/run/instruction notes.')
