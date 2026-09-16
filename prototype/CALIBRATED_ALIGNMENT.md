# Geometry comparison and calibrated alignment

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](../docs/CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

Launch Tread Station in `cv_conda` as before. No new models or training are needed.
The existing PyTorch pipeline still supplies tyre masks. Geometry and target
calibration use the installed OpenCV 4.11 APIs.

## Compare boundaries and rim

Inspect a frame, optionally **Compare all four** to obtain both region models,
then press **Compare boundaries + rim**. The frozen frame is shown with both
geometric hypotheses. The table reports support, residuals and the appropriate
image measurement for each method/model. Select a model to change both images.

Boundary midline angles and rim ellipse ratios are different observables. They
are not averaged into a wheel angle. Cross-model boundary angles and rim ratios
are compared only when masks overlap sufficiently; disagreement is visible.
Agreement between models is not proof of rim identity or measurement accuracy.
**Save comparison** writes the source image, numerical fits and screen diagram
under `results/geometry-comparisons/`.

## Calibrated alignment workflow

1. Open **Calibrated alignment**, then **Printable targets**. Two different
   ChArUco targets are provided: ground IDs 0–16 and wheel IDs 50–66. Each board
   is 5 × 7 squares, 30 mm per square, with 22 mm markers (DICT_4X4_100).
   Print the HTML at 100%, not fit-to-page. Check square dimensions with a ruler
   and mount targets on flat rigid material. The board area is 150 × 210 mm.
2. Capture at least 10 distinct photos of the ground target with the exact camera
   lens, zoom, orientation and capture mode to be used later. Move the target
   across the image and vary tilt substantially. Keep focus settings repeatable;
   disable variable digital stabilisation/cropping. Prefer 15–25 usable views.
3. Enter a camera/setup name and choose **Calibrate from photos**. Calibration
   runs in a worker so the interface remains responsive. Save the resulting JSON
   profile. Profiles contain intrinsics, distortion, image size, view errors and
   source hashes. RMS ≤1.5 px, view diversity and minimum counts are provisional
   screening gates, not an accuracy certificate. Portrait workstation frames
   must match the calibration orientation/aspect. Uniform resizing is supported;
   cropping, rotation or a changed lens needs the corresponding new calibration.
4. Put the ground target flat on a verified level reference plane. Its printed
   RIGHT (+board X) points vehicle-forward; printed UP (-board Y) points
   vehicle-left; its visible face (-board Z) points up. This establishes a
   right-handed vehicle frame: X forward, Y left, Z up. Ground-target heading
   error directly biases toe. A floor that is merely assumed level is insufficient.
5. Mount the separate wheel target rigidly parallel to a verified wheel plane,
   using a suitable fixture. A board resting on tyre rubber, an unverified clamp,
   wheel/rim runout or a tilted mounting plate introduces angle error. Fixture
   calibration/runout compensation is not implemented; these must be checked
   independently. Normal-direction sign is resolved using the selected wheel side.
6. Keep BOTH targets visible in the SAME frame. Choose left/right, confirm the
   setup, then **Measure workstation frame** or **Open target photo**. The dialog
   accepts raw target photos without running a tyre model. **Follow analysed
   video frames** evaluates new workstation analysis frames at a bounded cadence.
   Losing either target withholds the result; an old world reference is never
   reused across frames or camera motion.
7. Axes on the photo are target-local XYZ (red/green/blue), not arrows inferred
   from tyre texture. The camera pose of the ground target defines the vehicle
   frame; the wheel target supplies the wheel-plane normal. Positive camber means
   top outward; positive toe-in means wheel-forward direction toward the vehicle
   centre. Results are single-wheel target-assisted research measurements.
8. Enter independent reference camber/toe values when available to display signed
   errors. Record wheel/vehicle ID, fixture, pressure/load and reference instrument
   in the notes. **Save alignment evidence** stores the frame, target-axis overlay,
   full camera profile, pose residuals, outputs and entered reference values under
   `results/alignment/`. When the exact workstation frame was used, its available
   boundary/rim hypotheses are retained as separate supporting evidence.

No target-free 3-D angle is inferred from a silhouette or ellipse. The calibrated
path connects the inspection workstation to measured reference geometry; it does
not turn the existing mask axes into calibration data. No universal vehicle
tolerance, pass/fail decision, caster, thrust angle or full-vehicle alignment is
reported. Printed-target flatness, heading, mounting, camera calibration and lens
stability remain error sources even when reprojection residuals are small.

## Verification and remaining physical validation

`check_alignment.py` renders known target poses and recovers signed camber/toe
on both sides, estimates intrinsics from rendered calibration photos, checks
missing/ambiguous targets, duplicate photos and aspect changes, and exercises the
native dialog/reference controls and evidence export. `check_geometry_comparison.py`
tests method separation and cross-model region disagreement. Existing video smoke
checks exercise both fit hypotheses on the supplied clips using GPU model results.

These tests validate software geometry and UI behavior. The supplied recordings
do not contain this calibrated target setup. Independent fixture/rack experiments,
repeat mounting trials and a physical camera calibration are still needed to
quantify real alignment error. Numerical display precision is not claimed accuracy.

References: [OpenCV ChArUco](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html)
and [planar pose estimation](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html).
