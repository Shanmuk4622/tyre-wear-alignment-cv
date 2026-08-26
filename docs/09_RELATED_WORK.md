# 09 — Annotated Related Work and Novelty Audit

> Extends the Review-1 literature review with the methods research from `10_VISION_TECHNIQUES.md`.
> §6 is the honest novelty grading — read it before writing the Review-2 abstract.

---

## 1. Tyre-domain literature

### 1.1 Visual tyre and tread recognition

**[1] Petrovic et al. (2025) — "Deep learning-based instance segmentation for detection of tire tread area."** *Progress in Engineering Science* 2(2). [doi:10.1016/j.pes.2025.100080](https://doi.org/10.1016/j.pes.2025.100080)

Mask R-CNN with a ResNet-50 backbone, trained on **247 images**, tested on 62, achieving **COCO mAP 0.6081**. HOG features extracted from the segmented tread for condition classification.

*Why it matters:* the closest published viewpoint to ours. It proves a learned tread mask is feasible from a front view. Equally important, the modest dataset and mAP show **tread localisation is not a solved preprocessing step** — it must be evaluated explicitly, not assumed. Our design keeps high-resolution boundary information for shoulders and thin grooves rather than treating segmentation as a throwaway crop.

*How we differ:* SegFormer-B2 with tiled fine-structure passes and boundary + clDice losses; segmentation quality reported as a first-class metric.

**[6] Vivekanandan & Rajeswari (2026) — "Edge-based visual tire wear classification with behavior-aware fusion."** *Measurement* 276:121509. [doi:10.1016/j.measurement.2026.121509](https://doi.org/10.1016/j.measurement.2026.121509)

MobileNetV2 + behaviour-aware fusion, **4,000 images**, multiple manufacturers. Domain adaptation via Adaptive BatchNorm and Deep CORAL raised unseen-brand accuracy **88.2% → 92.4%**; ~44 ms/frame on a Jetson Nano.

*Why it matters:* the important result is not the real-time inference — it is that **brand and tread-design shift produce a measured 4-point generalisation gap.** This is the single most quotable justification for our unseen-brand test split. Domain shift here is documented fact, not a hypothetical risk.

*How we differ:* we report stratified performance by brand/tread family rather than a pooled accuracy, and treat domain adaptation as an ablation (`04_MODEL.md` #15).

### 1.2 Optical tread-depth measurement

**[2] Huber, Preindl & Betz (2022) — "TireEye: Optical On-board Tire Wear Detection."** *PHM Society* 14(1). [doi:10.36001/phmconf.2022.v14i1.3242](https://doi.org/10.36001/phmconf.2022.v14i1.3242)

Wheel-well-mounted device observing a longitudinal groove cross-section. Adaptive Canny recovers the groove outline; **tread-wear indicators provide an in-frame scale reference.** Full-vehicle accuracy **0.57 mm**.

*Why it matters:* the most directly transferable result in this list. It establishes three principles we adopt wholesale — stable camera placement, a constrained measurement region, and a **physical in-frame scale reference**. Our TWI-anchor loss (`04_MODEL.md §3F`) is this idea formalised into a training objective.

*Benchmark:* 0.57 mm on a full vehicle is the number to beat for on-board optical wear measurement.

**[3] Wang et al. (2019) — "Tire tread depth measurement based on machine vision."** *Advances in Mechanical Engineering* 11. [doi:10.1177/1687814019837828](https://doi.org/10.1177/1687814019837828)

Two cameras observing a laser plane intersecting a radial section; laser centreline extracted, transformed to world coordinates, groove concavities identified. **Absolute error < 0.2 mm on two tyres.**

*Why it matters:* demonstrates that accurate depth is a **calibrated 3-D metrology problem**, not an appearance problem. Sets the ceiling: with structured light you get 0.2 mm; without it you should not claim millimetres.

*Caveat to cite honestly:* two tyres is not a generalisation claim.

*How we use it:* structured light as a **training-time teacher** with RGB-only inference — the cheap-deployment story plus a clean ablation (#9).

### 1.3 Surface damage and anomaly segmentation

**[4] Chen et al. (2024) — "Tire Surface Damage Detection Based on Image Processing."** *Sensors* 24(9):2778. [doi:10.3390/s24092778](https://doi.org/10.3390/s24092778)

Multi-scale bilateral filtering, clustering, Harris corners, histogram correlation. Mainly sidewall scratches.

*Why it matters:* an explainable classical baseline. Dark rubber has weak colour differences, and edge-preserving filtering genuinely helps isolate local cuts. In our system these operations generate **candidate regions and interpretable comparisons**, not the final detector.

**[5] Ko et al. (2021) — "Anomaly Segmentation Based on Depth Image for Tire Manufacturing."** *Applied Sciences* 11(21):10376. [doi:10.3390/app112110376](https://doi.org/10.3390/app112110376)

16-bit depth images + DeepLabV3+. Stacking raw depth, histogram-equalised depth and a height heat map improved **mIoU by >7 points**.

*Why it matters:* directly supports our photometric-stereo decision. When a defect is defined by **protrusion, indentation or groove height**, depth-like channels carry evidence RGB texture does not. Their multi-representation stacking is the same idea as our `[RGB | normals | albedo | CLAHE]` input stack.

### 1.4 Vision-based wheel alignment

**[7] Furferi, Governi, Volpe & Carfagni (2013) — "Machine Vision System for Automatic Vehicle Wheel Alignment."** [doi:10.5772/55928](https://doi.org/10.5772/55928)

Stereovision with NIR markers around the sidewall; stereo triangulation → wheel-plane fit → toe and camber in a calibrated vehicle frame. Average difference vs a 3-D scanner: **~0.024° toe, ~0.026° camber**.

*Why it matters:* demonstrates the **correct measurement logic** — alignment angles are properties of a wheel plane relative to a vehicle coordinate system, not the apparent tilt of an image. Our analytic estimator follows this structure exactly.

*Why we cannot match it:* markers, stereo, and precise references. We are target-free and monocular. **State this gap explicitly rather than hoping nobody compares.**

**[8] Xu et al. (2022) — "Automatic and Accurate Vision-Based Measurement of Camber and Toe-In."** *IEEE T-IM* 71. [doi:10.1109/TIM.2022.3216382](https://doi.org/10.1109/TIM.2022.3216382)

Calibrated vision + geometric recognition; emphasises detecting stable circular/wheel structure before computing the angle.

*Why it matters:* reinforces detecting stable geometry first. From our low front view the rim is usually occluded, which is precisely why we substitute **shoulder contours and groove-orientation fields** as the stable structure.

**[9] Shi, Liu & Zappa (2026) — "Flexible Wheel Alignment via APCS-SwinUnet and Point Cloud Registration."** *Metrology* 6(1):4. [doi:10.3390/metrology6010004](https://doi.org/10.3390/metrology6010004)

Rim segmented by a task-specific Swin-UNet, point cloud registered by ICP, rotation matrix → toe and camber. **Errors < 0.1°** in a controlled single-wheel feasibility study.

*Why it matters:* the target-free 3-D direction and the best modern comparator. The authors themselves acknowledge limited diversity and **no direct validation against a commercial aligner** — cite that honestly; it shows even the strongest recent work is a feasibility study, which properly frames our own scope.

*How we differ:* monocular RGB, no depth sensor. RGB-D is our ablation #9, not our baseline.

**[10] Zhang et al. (2023) — "Noncontact tire deformation measurement, Tire-Net."** *Measurement* 215:113034. [doi:10.1016/j.measurement.2023.113034](https://doi.org/10.1016/j.measurement.2023.113034)

Semantic segmentation + sub-pixel edge extraction + physical scale conversion.

*Why it matters:* the clearest precedent for our hybrid split — **deep learning finds the correct region, calibrated analytic vision performs the physical measurement.** This is the design pattern of our whole alignment module.

### Tyre-domain summary

| Study | Approach | Reported result | Relevance |
|---|---|---|---|
| Petrovic 2025 | Mask R-CNN front-view tread | mAP 0.6081, 247 imgs | Same viewpoint; segmentation must be evaluated |
| Vivekanandan 2026 | MobileNetV2 + DA | 88.2% → 92.4% unseen brand | **Brand shift is measured, not hypothetical** |
| Huber 2022 | Wheel-well groove + TWI scale | **0.57 mm** | In-frame scale reference; benchmark |
| Wang 2019 | Stereo + laser plane | **<0.2 mm** (2 tyres) | Depth is 3-D metrology |
| Chen 2024 | Classical filtering | — | Explainable baseline |
| Ko 2021 | Depth + DeepLabV3+ | **+7 mIoU** | Depth channels beat RGB for height defects |
| Furferi 2013 | Stereo NIR markers | **~0.025°** | Correct alignment logic; marker-based |
| Xu 2022 | Calibrated vision geometry | — | Find stable structure first |
| Shi 2026 | Swin-UNet + RGB-D ICP | **<0.1°** | Modern target-free 3-D; still feasibility-stage |
| Zhang 2023 | Tire-Net + sub-pixel edges | — | **Hybrid learned-region + analytic-measurement pattern** |

---

## 2. Method literature adopted from the technique research

### Illumination and photometric stereo

- **[11]** Photometric Stereo-Based Defect Detection for Steel Components, *Sensors* 2022 — [PMC8838491](https://pmc.ncbi.nlm.nih.gov/articles/PMC8838491/). Photometric stereo + convolutional segmentation deployed on **highly specular** industrial surfaces. Establishes the approach on the same specularity problem wet rubber has.
- **[12]** A dataset for surface defect detection on complex structured parts based on photometric stereo, *Scientific Data* 2025 — [doi:10.1038/s41597-025-04454-6](https://www.nature.com/articles/s41597-025-04454-6). Public dataset for method development.
- **[13]** Defect segmentation for multi-illumination quality control, *MVA* 2021 — [link](https://link.springer.com/article/10.1007/s00138-021-01244-z). Multi-angle pseudo-colour imaging; documents that a **static ring light cannot distinguish a stain from a shadowed cavity.** This is the argument for our illumination design in one sentence.
- **[14]** Polarization 3D imaging technology: a review, *Frontiers in Physics* 2023 — [link](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2023.1198457/full). Shape-from-polarisation background; supports cross-polarisation and scopes SfP as future work.

### Thin-structure segmentation

- **[15]** Shit et al., **clDice — a topology-preserving loss for tubular structures** — [arXiv:2003.07311](https://arxiv.org/pdf/2003.07311). Dice on skeletonised predictions, with a topology-preservation guarantee up to homotopy equivalence; **explicitly proposed for crack detection in industrial quality control.** Adopted for sipes and cracks.
- **[16]** Skeleton Recall Loss — [arXiv:2404.03010](https://arxiv.org/html/2404.03010v1). Cheaper connectivity-preserving alternative.

  *Known trade-off to state in the report:* clDice is insensitive to boundary shifts within the structure radius, so it slightly costs boundary precision. We therefore **combine** it with boundary loss rather than replacing it, and ablate it specifically on connectivity metrics.

### Foundation models and annotation efficiency

- **[17]** FS-SAM2: adapting SAM2 for few-shot segmentation via LoRA — [arXiv:2509.12105](https://arxiv.org/html/2509.12105). Meaningful gains with **as few as 50 images per class.**
- **[18]** SAM2 memory propagation for video annotation — throughput **37.8 s/frame → 4.5 s/frame** in a comparable workflow. Directly adopted (`03_DATA.md §5`).
- **[19]** Rethinking Transfer Learning for Industrial Inspection: DINOv3 vs ImageNet — [arXiv:2605.23472](https://arxiv.org/html/2605.23472). **Frozen SSL features give no clear advantage on RGB industrial tasks; fully fine-tuned SSL initialisation is strongest.** This changed our training recipe.
- **[20]** DINOv2 — [arXiv:2304.07193](https://arxiv.org/html/2304.07193v2).

### Monocular depth — a documented negative result

- **[21]** Depth Anything V2 — [arXiv:2406.09414](https://arxiv.org/html/2406.09414v2), and independent robotics evaluation reporting that **"fine details and close-range depths in feature-sparse areas are not represented well."**

  Dark, low-texture, close-range rubber is close to the worst case for this model family. **We run it once as a negative-result ablation and do not build on it.** A documented negative result is a legitimate contribution.

### Core architectures

- **[22]** Xie et al., SegFormer, NeurIPS 2021 — [link](https://research.nvidia.com/labs/lpr/publication/xie2021segformer/)
- **[23]** Woo et al., ConvNeXt V2, CVPR 2023 — [link](https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html)
- **[24]** Roth et al., PatchCore, CVPR 2022 — [link](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)
- **[25]** Sun et al., HRNet, CVPR 2019 — [link](https://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html)
- **[26]** Teed & Deng, RAFT, ECCV 2020 — [arXiv:2003.12039](https://arxiv.org/abs/2003.12039)
- **[27]** Cao et al., CORAL rank-consistent ordinal regression — [arXiv:2111.08851](https://arxiv.org/html/2111.08851v5)
- **[28]** Angelopoulos & Bates, A Gentle Introduction to Conformal Prediction

---

## 3. Public datasets — what they are good for

| Dataset | Size | Labels | Verdict |
|---|---|---|---|
| [Tyre Quality Classification](https://www.kaggle.com/datasets/warcoder/tyre-quality-classification) | 1,854 | good / defective | Binary static close-ups. **Backbone pretraining only** |
| [Tire Texture Image Recognition](https://www.kaggle.com/datasets/jehanbhathena/tire-texture-image-recognition) | 1,028 | cracked / normal | Texture pretraining |
| [Tyre Condition Classification](https://www.kaggle.com/datasets/sameersambhare1/tyre-condition-classification-dataset) | — | new / serviceable / unusable | Closest to ordinal; warm-start the CORAL head |
| [Roboflow tyre datasets](https://universe.roboflow.com/search?q=class%3Atyre) | varies | boxes, some masks | Warm-start the detector |

**Use them for exactly two things:** self-supervised backbone pretraining (labels irrelevant), and warm-starting the detector/segmenter.

**Do not benchmark against them and claim victory.** They are static hand-held photos with binary labels; ours is a calibrated video task with localisation, geometry and uncertainty. Say so plainly in Related Work — that framing *strengthens* the contribution.

---

## 4. Where this project sits

| Axis | Best published | This project |
|---|---|---|
| Front-view tread segmentation | mAP 0.608, 247 imgs [1] | SegFormer-B2, 250–400 tyres, boundary + topology losses |
| On-board optical depth | 0.57 mm [2] | RGB-relative + TWI anchor; metric only with a metric sensor |
| Structured-light depth | <0.2 mm [3] | Ablation ceiling / training teacher |
| Marker-based alignment | ~0.025° [7] | Target-free; camber ~0.3–0.5°, toe screening |
| RGB-D target-free alignment | <0.1° [9] | Monocular; RGB-D as ablation |
| Wear classification | binary / 3-class | **Ordinal + 9-way multi-label + localised heatmap** |
| Uncertainty | rarely reported | Conformal intervals + explicit refusal |

**The defensible positioning:** every prior system does *one* of these well, usually with extra hardware (markers, stereo, lasers, depth cameras) and usually without uncertainty. We do **detailed localised wear recognition and alignment screening together, from one ordinary camera, with calibrated confidence and an honest refusal state.**

Do not claim to beat Furferi or Shi on angular accuracy. Claim a different operating point: **no markers, no stereo, no depth sensor, and a system that knows when it cannot answer.**

---

## 5. Gaps in the literature this project addresses

1. **No public front-view single-tyre video dataset** with tread masks, fine structures, gauge depth, wear patterns, damage and calibrated alignment labels together
2. **Wear work is binary or 3-class**; nobody publishes localised, multi-label, ordinal wear from this viewpoint
3. **Alignment work needs markers, stereo or RGB-D**; monocular target-free single-wheel screening is unexplored
4. **Uncertainty and refusal are almost never reported** in tyre inspection, despite it being a safety task
5. **Circumferential coverage is never quantified** — papers show one view and imply the whole tyre
6. **Photometric stereo is proven in industrial inspection but unpublished on tyres**, despite rubber being the ideal case (low albedo, shape-defined features)
7. **Wear and alignment are never estimated jointly**, so no one can distinguish recent from chronic misalignment

---

## 6. Novelty audit — honest grades

| # | Claim | Verdict |
|---|---|---|
| 1 | Front-view single-tyre video **dataset** with paired wear + alignment labels | ✓✓ **Strong** — no equivalent exists, and it outlives the project |
| 2 | Multi-task **localised** wear + damage instead of binary classification | ✓ **Solid** — a real step beyond [1], [6] |
| 3 | **Partial tread unrolling with explicit coverage reporting** | ✓ **Novel and honest** — nobody quantifies what fraction they actually saw |
| 4 | Hybrid learned-landmark + analytic alignment | ~ **Established pattern** ([10], [8]); novel in *this* viewpoint and single-wheel scope |
| 5 | **Wear ↔ geometry cross-check; recent vs chronic misalignment** | ✓✓ **Strongest claim — nothing comparable found** |
| 6 | RGB vs photometric-stereo vs structured-light vs RGB-D on **one tyre set** | ✓ **Genuinely useful** — no such controlled comparison exists |
| 7 | **Photometric stereo applied to tyre wear** | ✓ **Novel application** of an established industrial method |
| 8 | TWI bar as in-image metric anchor | ~ **Prior art** ([2] TireEye). Novel only as a *training loss term*. Cite Huber |
| 9 | Conformal intervals + explicit refusal | ~ Standard method, new application. Good practice, not a headline |
| 10 | Optional app | ✗ Engineering, not research |

**Review-2 / paper headline order:** 5 → 1 → 6 → 3 → 7. Keep 8, 9 in Methods where they belong; do not lead with them.

---

## 7. Reading priority

**Tier 1 — read fully before Review-2**

1. Huber et al. 2022 (TireEye) — closest measurement precedent, and the TWI scale idea
2. Petrovic et al. 2025 — closest viewpoint precedent
3. Zhang et al. 2023 (Tire-Net) — the hybrid design pattern we follow
4. Shit et al., clDice — the loss upgrade for sipes and cracks
5. Defect segmentation for multi-illumination QC [13] — the illumination argument

**Tier 2 — method grounding**

6. Shi et al. 2026 · 7. Furferi et al. 2013 · 8. Ko et al. 2021 · 9. Vivekanandan & Rajeswari 2026 · 10. Photometric stereo for steel components [11] · 11. DINOv3 vs ImageNet [19] · 12. SegFormer · 13. ConvNeXt V2 · 14. PatchCore · 15. HRNet · 16. CORAL · 17. Conformal prediction intro

**Tier 3 — skim:** Wang 2019, Chen 2024, Xu 2022, RAFT, Depth Anything V2, SAM2/FS-SAM2, Kaggle dataset cards.
