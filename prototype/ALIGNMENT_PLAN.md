# Alignment through visible geometry

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](../docs/CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

Start with existing masks and uncalibrated portrait video. No new training is required.
The first output is camera-relative silhouette geometry, with the derivation drawn
on the actual captured frame. This extends the native workstation.

## Implemented: Shape Compass

The **Shape Compass — explain geometry** button opens a light, native diagram:
selected model tyre mask → largest connected component → pixel covariance/PCA
axis → apparent tilt relative to image vertical. It uses the exact analysed frame,
not the moving source preview. An assumed-reference slider demonstrates camera
roll ambiguity without changing the image or pretending to calibrate it.

Empty, tiny, clipped, fragmented and nearly round silhouettes withhold the angle.
The size/fragmentation/roundness gates are provisional heuristics, not validated
confidence bounds. Largest component selection can still choose a false positive.
PCA describes mask shape, not the wheel axle or wheel plane. Tyre and tread masks
are not verified rim contours. No ellipse-to-toe or ellipse-to-camber conversion.

## Implemented: temporal geometry overlay

**Track geometry on image** is enabled by default. The analysed-frame overlay
shows the current contour, raw gray axis, blue image vertical and, after three
consistent frames, an amber smoothed axis. Original view remains untouched.
The tracker uses doubled-angle smoothing, overlap and angle/shape continuity gates.
Missing/invalid masks, large changes, source/recipe changes, seeks, replay and
long gaps reset acquisition. Gates are provisional heuristics. A stationary wrong
mask can still be stable. The overlay is display-only; saved masks remain raw.

### Future refinements

Track the same wheel region through a short fixed-camera burst. Collect principal
axis, centroid, shape ratio, mask overlap and boundary completeness. Reject abrupt
mask changes and occlusion. Smooth undirected angles with doubled-angle circular
statistics, not ordinary means. Show raw and smoothed traces together, the number
of accepted frames, and mask overlays for every rejected interval.

Compare repeated captures only with the same camera/reference/view. A baseline
records relative change, never declares the wheel correctly aligned. Reset the
baseline when the camera moves. Video consistency measures repeatability; it does
not prove accuracy or remove a stable segmentation error.

## Implemented: geometry hypotheses with visible evidence

The view selector adds **Boundary lines / tread view**, **Rim candidate / side
view**, and **Image edge fitting off**. Boundary mode is the default. It finds
Canny edges near the two mask sides, uses deterministic two-point consensus to
reject outliers, then fits paired lines and a midline. It requires support along
both sides and suppresses a midline when side directions disagree or cross.

Rim mode searches image contours inside an eroded tyre region, fits ellipse
candidates and gates by residual, angular coverage, size and containment. The
largest supported candidate is shown; it can be lettering or another internal
structure, so the operator must visually confirm a rim. This is contour fitting
with rejection gates, not a trained rim detector or guaranteed robust recovery.

**Explain edge fit** shows source → supporting/rejected points → fitted shape.
Green points support a local fit; red points are rejected. Cyan lines show sides;
magenta shows the midline or ellipse candidate. The readable status below the
image explains rejection or reports image-relative fit residuals. Residual is not
alignment uncertainty. Search runs at a bounded 640px longest side and maps back
to the source frame; sliders reuse cached fits. Original view stays untouched.
Saved evidence adds `edge-fit.png` and an `edge_geometry` record including model,
algorithm version, view, points, inliers and fitted geometry.

### Remaining experiments

Use the mask to restrict edge searches in the source image. For front/tread views,
test paired boundary lines and their midline. For side views with an actual visible
rim, test robust ellipse fitting to rim edges, displaying inliers, rejected edges
and fit residuals. Require sufficient boundary coverage. Compare methods and
models on the same frame; disagreement should suppress a combined estimate.

An ellipse is a projected shape descriptor. Circle/plane assumptions, perspective,
lens distortion and pose ambiguities must be resolved before interpreting it as
wheel orientation. Do not fit the thick tyre silhouette and call it a rigid rim.

**Compare boundaries + rim** now shows both hypotheses per model, their support,
residuals and cross-model disagreement. Different region selections suppress
agreement. Current edge hypotheses are per-frame, while silhouette PCA already
has temporal tracking. No combined image-derived alignment estimate is produced.
Temporal edge-fit traces remain a future refinement.

## Implemented: target-assisted calibration and alignment workflow

See `CALIBRATED_ALIGNMENT.md`. The app creates/loads camera profiles, exports two
distinct ChArUco targets, solves both target poses in the same frame, and computes
single-wheel camber/toe in an explicitly defined vehicle reference frame. The
dialog supports reference readings and preserves evidence. Synthetic image and UI
tests pass; real physical accuracy has not been established.

### Remaining physical validation

Add camera calibration, vertical/reference geometry and a verified wheel-plane
relationship for camber. Add a defined vehicle longitudinal reference for toe.
Check against independent reference measurements and repeated target remounts.
Use held-out wheel/vehicle identities rather than splitting adjacent video frames.
Report error and rejection rates before assigning vehicle-specific tolerances.

## Verification

Now: synthetic signed-axis tests, circular/clipped/fragmented/empty mask rejection,
diagram rendering, slider interaction and workstation button integration.
Next: manually reviewed real-video intervals covering stable views, camera motion,
occlusion and mask failures. No alignment accuracy claim from current recordings.

Reference: OpenCV PCA orientation tutorial:
https://docs.opencv.org/4.5.3/d1/dee/tutorial_introduction_to_pca.html

OpenCV contour and ellipse fitting:
https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
