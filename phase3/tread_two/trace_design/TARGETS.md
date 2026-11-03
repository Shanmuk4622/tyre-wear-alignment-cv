# Fixed-target mathematics and information boundaries

**Implemented prototype:** [targets.py](runtime/targets.py), revision
`trace-fixed-target-v2`. [Independent numerical proof](validation/NUMERICAL.json).
No trainable parameter participates in target construction.

## Canonical image and local preprocessing

Use the declared frozen geometry to form the existing 512×256 atlas. Split it
without overlap into 8×4 tiles of 64×64. Predicted tyre∩tread support is binary;
source rows narrower than 16 pixels are unavailable. The original geometry
sampling uses bilinear RGB, nearest support and `align_corners=True`.

For each tile, let `v` be predicted valid support and `g=(R+G+B)/3`. Compute masked
9×9 local means and variances. Define

`z = clip((g−mean9(g*v)/mean9(v)) / max(std9,0.02), −3,3) / 3 * v`.

The variance uses the corresponding masked second moment; its negative numerical
tail is clamped to zero. Horizontal/vertical Sobel filters, divided by eight,
produce `dx,dy` from `z`, then multiply by `v`. Network input channels are
`z,dx,dy,sqrt(dx²+dy²+1e−12)*v`. These operations never cross a tile boundary.

## Six observed fields per tile

| Channel | Exact field before 16×16-cell pooling | Interpretation |
|---|---|---|
| `contrast` | `z` | Signed local normalised intensity |
| `gradient_energy` | `sqrt(dx²+dy²)` | Local image-edge energy |
| `orientation_axis` | `(dx²−dy²)/(dx²+dy²+0.0025)` | Stabilised horizontal/vertical orientation preference |
| `orientation_cross` | `2*dx*dy/(dx²+dy²+0.0025)` | Stabilised diagonal orientation component |
| `relative_curvature` | `clip(abs(laplace(z)/4)/(abs(dx)+abs(dy)+0.05),0,1)` | Bounded second-order image structure |
| `contrast_occupancy` | `1[abs(z)>0.15]` | Fraction of locally contrasted pixels after pooling |

`laplace` uses the four-neighbour kernel with centre −4. Orientation axes refer
to canonical **image coordinates**, not road camber/toe or physical groove depth.
All constants are source-bound design choices, not empirically validated physical
thresholds. Occupancy is a fixed target and needs no derivative.

## Dependency-safe weights and normalisation

Target loss support is predicted support times the complement of the original
annotation ignore mask. Erode it with an 11×11 minimum filter and explicit invalid
padding: each retained target pixel has an entirely valid radius-five dependency
for local contrast followed by a Sobel/Laplacian operation. This also excludes the
outer five pixels of each tile.

Divide each tile into 4×4 non-overlapping cells, each 16×16 pixels. A cell target
is the weighted mean of each of six fields. Keep the exact valid-pixel count;
counts below 64 make all six channel targets unavailable. A zero placeholder
with zero weight is **missing**, not a measured zero target. Targets have shape
`N×32×6×4×4`; weights `N×32×1×4×4`.

Fit weighted population mean/std for each channel on **all 386 train images
only**. Accumulate in FP64, store FP32; standard deviation has floor 0.001.
Normalised targets use `(target−mean)/std`. Fit a separate train-position mean
for every tile/cell/channel, with its exact count and availability. No whitening,
per-role rescaling, label-conditioned statistic or held-out fitting is used.

The complete construction produces **197,632 cells /1,185,792 channel values**
for train. **197,201 cells** are supported in the actual local audit. Full maps,
weights, position controls and one deterministic mask plan are saved directly as
CSV/NPY locally. [Actual target receipt](validation/ACTUAL_TARGETS.json).

## Two views and masking

The proposed appearance factors are deterministic brightness/contrast/gamma in
`[0.9,1.1]` per image/view, with tile-local contrast means. Targets always come
from the clean unaugmented atlas. There are no flips, geometric warps or source
swaps in the first candidate.

For each four-column band, choose one supported tile if at least two exist.
At most eight tiles are hidden. Remove their **raw RGB and observation support**
before preprocessing; substitute a learned mask token only after the masked
encoder. Position and predicted-coverage metadata remain visible. No hidden RGB,
hidden target, ignore mask or clean full-path encoding enters this context path.

Context eligibility requires at least four supported tiles across two bands,
at least two visible supported tiles and some supported hidden target cells.
Record exact reasons when it fails. Covariance/variance terms additionally need
at least **seven eligible rows** for a 6D sample covariance; smaller sets receive
explicitly unavailable diversity diagnostics and no such loss.

## Controls on identical cells

| Control | Inputs permitted | Missingness |
|---|---|---|
| Train channel mean | Six frozen train-channel statistics | No held-out fitting |
| Train position mean | Frozen train tile/cell means | Unsupported train positions excluded explicitly |
| Visible band mean | Clean fixed descriptors of visible columns in the same band | Exclude cells with no visible neighbour support |

Score model and all controls on the same intersection of supported hidden cells.
Retain each control's native coverage, the intersection count, Huber sums/counts
and per-image reductions; require high common coverage before making a utility
claim. Native model loss on its wider support is reported separately. The local
train-only control table uses this common support and is descriptive, not an
estimate of held-out performance.

## Prototype objective

For each view, use Smooth L1 with beta 0.1 on standardised targets. Weight each
cell by its valid-pixel count, average the six channels within each image, then
average available images. Local and masked objectives have separate availability.
Masked loss excludes a whole context-ineligible image. Do not silently replace
this image-mean reduction with a global pixel mean.

`L = L_local + L_masked + 0.1*L_pair + 0.1*L_variance + 0.01*L_covariance`.

`L_pair` is MSE between paired 6D projected vectors on the shared eligible set.
Each view's sample covariance divides by `n−1`. Variance penalty is
`mean(relu(0.1−sqrt(diag(C)+1e−4)))`; covariance penalty is the off-diagonal
squared sum divided by `6*5`. Average the two views' local/masked/diversity terms.
All five components and availability must be retained separately.

These prototype weights are the draft first recipe. Actual task-gradient,
source-coverage, format/storage and two-T4 measurements must precede its frozen
measured protocol. They are not tuned using the inspected test role.

The implementation demonstrates an 8×6 centred matrix with identity covariance
to FP64 roundoff. This removes the specific 64D/rank≤7 conflict; it neither
guarantees six independent semantic factors nor justifies a healthy projector
when its encoder remains collapsed.
