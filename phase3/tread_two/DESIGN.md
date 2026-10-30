# TREAD-TWO / ATLAS — architecture and mathematical contract

**Prototype v0.1, 29 October 2026.** The new classifier is untrained. Numbers
below describe the implemented architecture; training gains are hypotheses.
The primary idea is a geometry-normalized, case-based classification path
attached to an unchanged dense anchor.

**Implementation update:** the [v1 runtime](PILOT_IMPLEMENTATION.md) and
[NB08 notebook](../notebooks/phase3_NB08_TREAD_TWO_ATLAS_Resumable_Pilot.ipynb)
are now delivered with local engineering acceptance. The original prototype
described here remains preserved. V1 uses FP32 and fixed microbatch-one anchor
inference, freezes the deterministic bank after [training-only review](BANK_REVIEW.md),
and implements original-image mass-conserving evidence splatting. Actual Kaggle
and live-HF gates still precede the measured fit.

## 1. Dense anchor and gradient boundary

Use the declared TREAD-ONE seed-3/epoch-20 state, SHA-256
`a5b872ed2e4cbfb3b9782b0df563e2606631744cd0274fee2da42a832f985f65`.
Its exact frozen architecture SHA is
`d16cd4eef75779044dd95900c0531fee474704cf88f523326ee069115347ba09`.
Source and weights are verified before `weights_only=True` loading.

All anchor parameters have `requires_grad=False`. The candidate's `train()`
override explicitly leaves the anchor in eval mode. Dense forward and atlas
construction use `no_grad`, and the new optimizer must contain only the 81,274
trainable classifier parameters. Old classification output is retained only
as a diagnostic; it does not determine the new prediction. The anchor and
grid run FP32 with autocast explicitly disabled, including inside an outer
mixed-precision context. A CPU bfloat16-context check measures zero dense drift.

For the same input, batch size, hardware/math and precision, the mask/point
outputs are the frozen anchor's outputs. The prototype measures a maximum
difference of **0.0** for both. This is a parameter/forward-preservation property,
not a guarantee that FP32 matches historical mixed-precision numbers or that
predictions become physically correct.

## 2. Visible 2D tread atlas

Input is RGB FP32 in [0,1], 512 high × 384 wide. The anchor receives its original
ImageNet normalization. Quarter-grid sigmoid masks use threshold ≥0.5 and the
frozen repeat-by-four reporting rule. Atlas support is the **intersection of
predicted tyre and tread masks**; raw dense outputs remain unchanged.

For source row y, L(y)/R(y) enclose supported pixels. Rows narrower than 16 input
pixels are unsupported. Supported y extent is mapped to 512 atlas rows; each
row's visible L/R span maps to 256 columns. Sampling uses a bilinear RGB grid,
`align_corners=True`; validity is sampled separately with nearest interpolation.
RGB outside valid predicted support is zeroed. Missing rows stay unsupported;
they are not silently replaced by centre pixels or the original full frame.

Clipped transverse rows remain visible partial-tread evidence and are flagged.
Their fraction is recorded; the atlas does not claim a complete tyre cross
section. The map is a **2D image-coordinate normalization**, not calibrated 3D
unrolling, tread depth, circumference or camera-pose reconstruction.

The grid is detached: no classification gradient goes through the sampler or
the dense model. The installed Torch 2.5.1 grid-sampling documentation warns
about CUDA backward nondeterminism; this prototype needs no sampler backward.
Future implementation must preserve that boundary and verify the actual
Kaggle precision/device path before training.

## 3. Local representation

The atlas becomes **32 nonoverlapping 64×64 patches**, eight longitudinal bands
and four lateral positions. All query and reference patches use one shared
96-dimensional encoder:

| Stage | Operation |
|---|---|
| Detail input | RGB normalized by (.5,.25), plus fixed Sobel channel-mean grayscale gradient magnitude |
| Stem | 4→24, 3×3 stride 2, GroupNorm, SiLU |
| Groove block 1 | Parallel depthwise 7×3 / 3×7, concatenate, 1×1 mix to 32 |
| Downsample | 32→48, 3×3 stride 2, GroupNorm, SiLU |
| Groove block 2 | Same directional construction, mix to 64 |
| Embedding | 64→96 stride 2, GroupNorm/SiLU, global pool, L2 normalization |

Directional filters provide an explicit local-pattern bias. They are not
human-labelled groove detectors. The prototype deliberately has no additional
transformer/context module; the encoder plus evidence layer is the concrete
pilot mechanism, rather than an unbounded list of architectural additions.

## 4. Actual-image reference cases

Six cases per class give 18 raw RGB patches. One eligible source photo per
original training tyre provides the bank: **eight photos** from 1 low, 3 mid
and 4 high tyres. Selection is deterministic, training-only, with ≥0.75 patch
validity and evenly spread eligible atlas positions. This validity threshold
is an engineering prototype choice, not a calibrated quality criterion.

Every case retains source image/tyre IDs, original SHA, class provenance,
patch coordinates, predicted support/clipping and anchor hash. Raw reference
pixels are fixed buffers. The same current encoder re-embeds them during the
forward graph, so references are actual inputs to the decision; they are not
nearest images attached afterward to explain unrelated learned vectors.

No learned-prototype projection or optimizer-moment reset is needed. No query
label/tyre ID chooses the bank at inference. The bank is the same for every
query. Query-conditioned class/subject exclusions are avoided because changing
which class's cases are missing could itself leak the target.

The eight bank photos are excluded from the new classifier's **query** pool.
They still participate as bank inputs to the shared encoder. This prevents
an identical source photo being both support and query during fitting. It does
not remove same-tyre dependence. The training-only query pool is 378 images:
66 low, 246 mid (including existing frames), 66 high. Every original training
tyre retains at least one original-photo query, including the two-photo group.

## 5. Signed, auditable class decisions

For query patch i and reference p, normalized embeddings give cosine s(i,p).
With learned bounded temperature τ in (0.05,1):

```text
affinity(i,p) = exp((cosine(i,p) - 1) / tau)
W(c,p) = +2*sigmoid(U(c,p))/6   if reference p has class c
         -2*sigmoid(U(c,p))/6   otherwise
pair_evidence(i,c,p) = alpha(i) * W(c,p) * affinity(i,p)
patch_evidence(i,c)  = sum_p pair_evidence(i,c,p)
logit(c)            = bias(c) + sum_i patch_evidence(i,c)
probabilities       = softmax(logits)
```

Each supported longitudinal band receives equal total pooling mass. Within a
band, alpha is proportional to patch support coverage. Unsupported bands get
zero; an entirely unsupported atlas gives zero evidence and a bias-only raw
prediction, explicitly flagged. Such cases remain in all denominator counts.

Positive/negative reference contributions and class bias reconstruct the
prediction. The prototype's measured accounting residual is **0.0**. Kernel
temperature is a feature-similarity scale, not softmax probability calibration.
Source patch similarity and exact algebra do not establish physical/causal
wear reasoning; geometry itself depends on a wide-receptive-field dense model.

## 6. Evidence display and self-checks

Intended outputs include original pixels, predicted dense maps, canonical atlas,
32-patch class contributions, ranked supporting/opposing reference patches,
source identities and coverage/clipping/support counts. Raw three-class scores
must always be saved. No learned/calibrated confidence or rejection guarantee
is claimed by this prototype.

Patch contributions can be placed in atlas coordinates directly. A future
original-image overlay must use **mass-conserving splatting through the stored
grid** and keep bias separate; simple heatmap resizing does not prove logit
conservation. The original prototype stores atlas mappings and patch evidence;
the delivered v1 pilot now implements the mass-conserving projection, retains
full native selected/final maps and direct numerical residual checks.

Small brightness/view perturbations can later produce a logged stability
diagnostic, with raw and changed predictions both retained. It is not a model
ensemble and cannot be used to hide wrong examples or claim calibrated trust.

## 7. Risks visible in the actual illustration

Some source patches contain paint/marker lines, highlights or weak detail.
Mask support does not prove a patch is a useful groove reference. The real-patch
view makes those possible shortcuts inspectable; it does not automatically
remove them. Before a measured fit, freeze a **training-source-only** bank review
and artifact/colour-sensitivity diagnostic policy. Do not replace examples after
seeing validation/test performance to make an attractive explanation gallery.

The key remaining risk is identity/appearance transfer, especially one low
training tyre. A local case bank can still memorize tyre type, capture marks
or lighting. We are proposing a testable alternative representation, not a
proof that the existing cohort can support reliable physical-wear classification.
The row normalization can also stretch away absolute apparent groove sizes;
its usefulness must be measured rather than assumed.

## Primary references and what is different here

- [Spatial Transformer Networks, Jaderberg et al., 2015](https://proceedings.neurips.cc/paper_files/paper/2015/hash/33ceb07bf4eeb3da587e268d663aba1a-Abstract.html): learned spatial manipulation is established. Our prototype uses a frozen predicted-mask row map rather than a learned localization grid.
- [This Looks Like That / ProtoPNet, Chen et al., 2019](https://proceedings.neurips.cc/paper/2019/hash/adf7ee2dcf142b0e11888e72b43fcb75-Abstract.html): visual part prototypes motivate inspectable evidence. Our decision uses fixed actual raw-image references re-encoded by the shared current metric network.
- [Prototypical Networks, Snell et al., 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html): learned embedding metrics are established. This is not that paper's new-class few-shot protocol or class-mean rule.

TREAD-TWO is a project-specific synthesis and engineering proposal. No formal
novelty, state-of-the-art accuracy or physically meaningful wear claim is made.
