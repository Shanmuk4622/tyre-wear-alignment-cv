# GEO postrun batch — all notebooks delivered together

<!-- NB20_VERIFIED_GEO_CLOSED_20261102 -->
## Current — NB20 HF-verified; GEO research cycle complete

**2 November 2026:** NB20 completed both NB18 review and NB19 passport stages.
All 19 payloads match; three cells complete without saved errors. Actual HF
revisions **610bdb22… / abfec8d6…** contain **121 logical files, 51 CSVs /
183,812 rows**. All 119 objects, released owners, 48 reference-comparable tables,
exact vector arrays and actual passport provenance check out. Earlier NB13–NB17
selecting manifests are unchanged. **No NB18/NB19/NB20 rerun needed.**
[Completion and numerical/evidence notes](phase3NB20Completion.md).

The current GEO cycle is closed as **unselected, unpromoted research**. Geometry
passes 60/60 epochs; representation eligibility passes 0/60. The original
27,514,418 CSV rows and native/epoch evidence remain on HF.
[Compact research handoff](phase3TreadTwoGEOResearchClosure.md).

**Recommended next direction:** separately design a texture revision with fixed
local targets and batch-compatible diversity, reusing saved geometry and existing
photos. [Design brief](phase3TreadTwoGEONextResearch.md). This is proposed,
not implemented or trained. No new notebook/fit is required to finish this cycle.
Both workstation implementations remain deferred and preserved.
<!-- END_NB20_VERIFIED_GEO_CLOSED_20261102 -->

**Recommended: run NB20 once with Run All.** It executes NB18, publishes the
review, then executes/publishes NB19 without a manual handoff. The individual
notebooks are alternatives for separate runs; launching all three concurrently
duplicates work and competes for ownership.

| NB | Notebook | Mode | Delivered bytes |
|---|---|---|---:|
| 18 | [phase3_NB18_TREAD_TWO_GEO_Completed_Review.ipynb](notebooks/phase3_NB18_TREAD_TWO_GEO_Completed_Review.ipynb) | review | 178070 |
| 19 | [phase3_NB19_TREAD_TWO_GEO_Evidence_Passport_And_Research_Handoff.ipynb](notebooks/phase3_NB19_TREAD_TWO_GEO_Evidence_Passport_And_Research_Handoff.ipynb) | passport | 178087 |
| 20 | [phase3_NB20_TREAD_TWO_GEO_All_In_One_Postrun.ipynb](notebooks/phase3_NB20_TREAD_TWO_GEO_All_In_One_Postrun.ipynb) | all | 178082 |

## Kaggle setup and sequence

| Setting | Value |
|---|---|
| Accelerator | None / CPU |
| Internet | ON |
| Existing input | Expanded `shanmuk4622/tire-dataset-prepared-phase2` |
| Secret | Enable `HF_TOKEN`, write access to `Shanmuk4622/tyre-wear-study` |
| Fresh free space | At least 13 GB; full pinned input 10.31 GB, shared once |
| RAM | About 8 GB recommended; stream dense CSV and native shards |
| Action | Fresh kernel, preserve compatible session files, **Run All in NB20** |

NB19 also automatically satisfies a missing NB18 dependency. All three files
embed the same 19-file verified runtime package. No hidden
local Python files, earlier notebook variables or new dataset are required.
Compatible NB17 files can be reused through exact hashes and immutable hardlinks;
otherwise the pinned HF content is restored into one shared cache.

```mermaid
sequenceDiagram
  participant K as Kaggle NB20
  participant H as Hugging Face
  participant W as Secret-free CPU worker
  K->>H: Restore pinned NB17 evidence or verify cache
  K->>W: NB18: schemas, metrics, native tensors and existing masks
  W-->>K: Completed source-bound stage receipts + visuals
  K->>H: Publish verified direct NB18 evidence
  K->>W: NB19: capability passport + research handoff
  W-->>K: Model card + CSV limits + interactive passport
  K->>H: Publish NB19; release owner
```

## Retention, recovery and troubleshooting

| Event | Behavior |
|---|---|
| Completed input download | Size/SHA retained; reused on rerun |
| Finished CPU stage | Exact artifact hashes and result receipt retained; skip when matching |
| Interrupted CPU stage | Replay from the preceding complete stage; no optimizer exists |
| Major boundary | Immediate direct HF publication |
| Active run | 30-minute publication cadence, safe request-budget ledger |
| Catchable Stop/error | Terminate numeric worker, attempt immediate completed-evidence/status flush |
| Server limit | Respect Retry-After/reset headers; bounded retries, retain local work |
| Forced process/session death | Cannot upload unsent bytes; restore published stages and available local cache |
| Existing matching completed HF result | Restore and skip; no new analysis or training |
| Wrong data/source/content | Stop at the technical gate; preserve evidence and report the mismatch |
| No selected model | Expected original scientific rejection; continue the research passport |

The original measured snapshot remains immutable. New namespaces are
`phase3/tread-two/geo-r1/review/<postrun-contract>` and
`phase3/tread-two/geo-r1/passport/<postrun-contract>`. Their contract binds the
pinned fit revision, exact new sources and scope. Every stage publishes direct
CSV and appropriate JSON/MD/PNG/HTML/NPY files; no ZIP/NPZ envelope. The original
10.31 GB is not duplicated into a new HF report publication.

The parent reads only Kaggle Secret HF_TOKEN. Numeric workers receive no token.
HF SDK 0.36.2 is private and installed without altering Torch/CUDA/NumPy. One owner
and local lock, immutable object batches and selecting manifests protect writes.
The 96/hour commit guard and server-observed API/resolver windows remain separate;
cross-host account traffic cannot be globally counted by one local ledger.

## Scope and verification status

| Area | Status |
|---|---|
| NB17 completed training and source identity | Independently verified |
| Full local saved-evidence analysis | Passed on actual downloaded NB17 bytes and existing masks |
| Three notebooks: schema, source hashes, fresh extraction | Passed |
| Stage interruption/invalidation and automatic handoff | Passed offline with explicitly injected transport/worker fixtures |
| Actual new Kaggle/HF report publication | Pending user execution |
| Interactive report script controls | Node DOM fixture passed; no claim of full native-browser rendering |
| Physical depth/safety or GEO classification | Not validated / not fitted |
| Extra fit, new photos, workstation changes | None |

[Runtime and operations](tread_two/geo_postrun/README.md),
[delivery checks](tread_two/geo_postrun_validation/DELIVERY_CHECKS.json),
[notebook identities](tread_two/geo_postrun_validation/NOTEBOOKS.json) and
[NB17 findings](phase3NB17Completion.md). The passport records this pilot as
completed, unselected and unpromoted; no automatic next full fit follows it.
