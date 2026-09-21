# Switch YOLO to three sessions

The observed error was `Run already leased: yolo26m-combined-seed1`. This is a duplicate-owner protection, not a damaged checkpoint.

1. Stop all older YOLO copies, including any experimental worker copies. Wait for their final HF upload. The original session has been confirmed stopped/uploaded. NB02, NB03 and NB05 can continue.
2. Import each of the three supplied files into a separate Kaggle notebook. Use each file once:
   - Worker0 / Seed1
   - Worker1 / Seed2
   - Worker2 / Seed3
3. Attach `shanmuk4622/tire-dataset-prepared-phase2`, enable Internet and `HF_TOKEN`, select GPU, and Run All.
4. These migration copies have `TAKE_OVER=True` once, to reclaim leftover ownership after stopping previous writers. Set it back to False after successful startup for future runs. Never use takeover against a session that is still running.
5. Look for `assigned jobs: 1` and the correct seed ID. One GPU trains that seed. Existing checkpoints keep their original protocol and paths; no re-uploaded dataset or retraining from epoch zero is required for seeds with saved state.

At the read-only audit, seeds 1/2 had 23 completed epochs, with 4,628/4,627 saved optimizer updates. Seed 3 had no published training checkpoint. A newer final upload may contain more progress. Each seed targets 60 epochs.

Three sessions run the three seeds concurrently; they do not divide one seed across three notebooks. Relative to the original two-GPU session, this adds a third concurrent seed, not a threefold speedup per seed. Other models use different run IDs and do not need to stop.

The failed input notebook was archived and left unchanged. Only notebook configuration and explanatory text differ in these copies; all embedded runtime sources, model settings and run IDs are identical.
