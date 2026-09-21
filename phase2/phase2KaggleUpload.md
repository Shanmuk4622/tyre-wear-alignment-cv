# Upload and verify Phase 2

> Completed: the dataset is uploaded and NB00 passed on Kaggle. Its HF report at `3da9c33dc5d1ca9e57d2c85ec87bdcd4f6fcf876` was independently verified. Do not repeat these upload/preflight steps. Continue with `phase2RunNotebooks.md`; the old remaining-work list below is superseded by the runnable training revision.

## Dataset upload

1. Create a new Kaggle dataset titled **Tire Dataset Prepared phase2**. Keep the existing dataset unchanged.
2. Upload `releases/Tire_Dataset_Prepared_phase2_v1.zip` (777,988,937 bytes; approximately 742 MiB). The ZIP contains the original clean images and labels plus all new labeled frames; no second dataset is needed for this source release.
3. Choose visibility and a license appropriate to your data rights, then create the dataset. Kaggle supports ZIP uploads; see its [dataset guide](https://www.kaggle.com/docs/datasets).
4. After processing, check that the expanded `phase2_dataset_v1` tree contains `VERSION.json`, `SHA256SUMS.txt`, `manifests`, images, annotations and masks.

ZIP SHA-256:

`07c7aa41e84135f578a8cf1d8a1ec42a947b3d5062cbcfa935732799c15ab836`

## Run the supplied notebook

1. Import `notebooks/phase2_NB00_Dataset_Preflight.ipynb` into Kaggle.
2. Attach the new dataset through Add Input. Use CPU for this notebook; save GPU time for training.
3. Enable Internet. Enable the Kaggle secret `HF_TOKEN`, with write access to the existing HF dataset repository `Shanmuk4622/tyre-wear-study`. Never paste the token into a cell.
4. Run All. It verifies the pinned release, all image/mask files and checksums without copying the dataset into `/kaggle/working`.
5. Save the notebook outputs. The report is retained in `/kaggle/working/phase2_preflight` and uploaded in one commit to a unique `phase2/source-v1/…/preflight/…` path. The notebook verifies that commit by downloading the small report at its exact revision.

A transient HF failure uses bounded retries and server backoff. If publication alone fails, retry `publish_report(report)` while the session is alive. A catchable stop attempts to save/publish a partial report; a forced session termination cannot guarantee a final upload. No training is performed by NB00.

## Remaining work before training

- Match video1, Video2 and Video3 to the original 40k, 70k and 90k tyre sessions, then freeze the physical-tyre split. The videos are different tyres, but their exact assignments remain unknown.
- Review the 45 tread/tyre containment disagreements using `review/label_audit/phase2_labels.html` if available, or the contact sheets in that directory. Preserve raw labels; make any corrections as a new annotation revision. The current release includes explicit ignore masks. Stock YOLO export needs resolved conflicts or a verified ignore-aware trainer.
- Accept/correct the 912 new geometry proposals before treating them as HRNet truth. Existing 120 human point sets are reused; old images need no new labeling.
- Build and run the dual-T4 resource/resume smoke notebook, then the model-specific training notebooks described in `phase2TrainingContract.md`. Verify interrupted versus uninterrupted training before long runs.

The ZIP is ready for upload. It is a source release with deliberately unassigned splits, not permission to run the old training notebooks on a new folder layout.
