# Video export controls and visible angles — 21 September 2026

Restart Tread Station. Beneath the download buttons, choose **Video export FPS** from 1 to 60 (fractional values supported, default 10). Click **Download video**. A higher rate requires more model evaluations; output remains silent. The duration stays within one output frame of the source. Low-rate sources repeat frames when exporting at a higher rate.

**Include video information panel** is enabled by default. It adds a fixed-height dark footer below the image: time, export FPS, active region model, mileage-proxy prediction and model score, tread tilt, raw upper/middle/lower widths, and learned-point review status. Model scores are not calibrated probabilities of physical wear. Turning the panel off exports the overlay without that footer. Image pixels are not cropped to make space. Each export snapshots its controls, so changing FPS during a job affects only the next export.

Learned overlays and the inspection panel now show signed tread centreline tilt in degrees relative to the image vertical. It uses the three paired-boundary midpoints, and the displayed smoothed points if active. Positive means the top leans right; negative means left. Crossed, missing or nonfinite points withhold the angle. Camera perspective can change this value; it is not wheel camber/toe. Raw widths in the footer remain explicitly raw when display smoothing is active.

The calibrated-target alignment bench now prints its camber/toe values directly on the measured preview and saved image. Its calibration, two-target visibility and setup-confirmation requirements are unchanged. The ordinary video exporter does not reuse a single calibration measurement across unrelated frames or print unmeasured physical alignment angles.

## Checks

- Synthetic 5/30-FPS sources exported at 1, 7.5, 24 and 60 FPS; every decoded frame count, rate, timestamp sampling and duration bound checked.
- Invalid, out-of-range and nonfinite rates rejected; signed/vertical/invalid point-angle cases checked.
- Footer canvas is constant despite changing text; original image pixels remain intact above the footer.
- Existing export checks passed: default 10 FPS, PNG view fidelity, failure cleanup and background cancellation.
- Real GPU inspection plus separate CPU export passed at 12.5 FPS with the installed Phase 2 models, learned overlays and information panel. Changing the control to 6 FPS mid-export did not change the 12.5-FPS job or completion message. Encoded output: four frames, 360 pixels wide, portrait image plus footer.
- Synthetic calibrated alignment tests passed signed camber/toe on both sides, two targets, missing-target/setup rejection, UI and evidence.

Evidence: `workstation/phase2_video_controls_tests.json`, `workstation/phase2_video_export_preview.png`, `workstation/phase2_video_export_preview.mp4`. Source backups: `workstation/backup-video-controls/`.
