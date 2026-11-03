# GEO implementation and module ownership

<!-- NB16_DONE_STATUS -->
**Current update,1November2026:** [NB16 is complete and HF-verified](../../phase3NB16Completion.md); no rerun. NB17 is delivered in the isolated geo_pilot branch. Earlier pending/preparation statements below are dated history.
<!-- END_NB16_DONE_STATUS -->

**Implemented preparation, not a completed measured trainer.** The original
[architecture](plan/ARCHITECTURE.md), [experiment](plan/EXPERIMENT.md) and
[retention](plan/RETENTION.md) remain the scientific design. This dated record
distinguishes executable work from the next integration milestone.

## Actual parameter inventory

| Group | Parameters | Ownership |
|---|---:|---|
| Frozen original TREAD-ONE | 1,372,159 | GPU1, immutable selected seed3/epoch20 |
| Mask/raw-point refinements | 27,068 | GPU0, geometry loss only |
| Texture encoder/context/projector/decoder | 326,632 | GPU0, SSL loss only |
| Experimental annotation-error head | 2,179 | GPU0, detached observed-error targets |
| Trainable total | **355,879** | Three disjoint AdamW groups, separate norm1 clips |
| Packaged total | **1,728,038** | Below both approved parameter ceilings |

The checkpoint remains `a5b872ed…`, source `d16cd4ee…`. Hooks expose the existing
48×128×96 decoder features, 256D guide features and raw point parameters without
rewriting the original forward/state keys. Zero-initialized residual heads
reproduce original masks/points exactly on the matched local math path.

| Source | Implemented responsibility |
|---|---|
| `data_geo.py` | Exact overlay/source/checksum adapter; original ignore targets; seeded photometric views; no flips |
| `schedule_geo.py` | 386 primary +6 marked padding draws,49×8, unique IDs/batch and≥2 conservative groups |
| `anchor_geo.py` | Strict selected-weight SHA/load, frozen eval, microbatch1 and detached feature transfer |
| `atlas_geo.py` | Detached predicted-support warp, original canonicalizer agreement and replayable row parameters |
| `model_geo.py` | Bounded refinements, patch-local four-channel LC, axial context, VIC projector, masked decoder and detached quality head |
| `objective_geo.py` | Explicit geometry/quality/VIC/masked Huber reductions and human-only point-error targets |
| `store_geo.py` | Finite engineering complete-update transaction, immutable unit CSV/NPY, two rolling native states, verified snapshot |
| `engineering_geo.py` | Real batch8 four-update local/HF seams, independent group clips, native observations and extensive telemetry |
| `checks_geo.py` | Independent loss calculations, actual hidden-RGB test, missing-target/dense confusion checks and crash injection |
| `worker_geo.py` | Separate numeric subprocess without HF_TOKEN |
| `coordinator_geo.py` | Kaggle ownership/status/stop, 30-minute/major publication, verified download/continuation, finite storage report |
| `report_geo.py` | Interactive update filter and 32 predicted-mask cases |
| Nine copied dependencies | Exact frozen anchor/data/geometry/LC/common/HF transport copies; provenance in `VENDORED.json` |

`runtime/FILES.json` binds the new source/config package. Its 40-family schema is
the approved **design proposal**, not a claim that all measured-fit emitters are
already wired. Existing executed source catalogs are untouched.

## Pre-freeze technical clarification

Texture photometric contrast uses a **per-tile mean** before the per-tile 9×9 LC
operation. A global atlas mean would let hidden tile pixels affect visible tile
inputs. The actual hidden-RGB intervention now leaves visible LC values and the
masked-context prediction exactly unchanged. Geometry photometric contrast stays
per-image/per-channel, with brightness→contrast→gamma and clipping at each stage.
This clarification is recorded before a measured fit; no completed experiment
was changed.

## Still required before NB17

| Work | Why it remains a prerequisite |
|---|---|
| Actual NB16 T4/HF receipt and independent audit | CPU/offline proof does not establish real two-GPU/transport behavior |
| All40 CSV/443-field measured emitters | Engineering tables intentionally have a narrower finite schema |
| At-most8-update raw stream→NPY shard sealing/rotation | Engineering uses one immutable NPY per tensor/unit and never claims a scalable 2,940-update layout |
| Full measured storage and staging peak | Current fixture measures body/vector/dense sizes; other40-family paths and copy/hardlink staging are incomplete |
| Train-only mean predictor, G0 baseline and fixed81-image validation | Required original selection/non-collapse rules; not needed for disposable optimizer seams |
| Full60-epoch coordinator and fresh-session HF auto-resume | Engineering fresh-session reruns are separate from the future measured-fit guarantee |
| Epoch/selected/final/intervention/export review | No candidate was trained or selected here |

NB17 is created after these checks/integration, then one seed/60epochs/2,940updates
may be measured. NB18 follows completed retained evidence. No additional
photography, hidden model sweep, production promotion or workstation changes.
