"""Generate the post-Stage-A notebooks without touching executed NB00--NB05.

`build_notebooks.py` owns the shared notebook helpers and embedded library.
This file owns NB06--NB10 because those notebooks consume public HF artifacts
and have a different, evidence-gated execution order.
"""


def build_later(ctx):
    md, code, write_nb = ctx["md"], ctx["code"], ctx["write_nb"]
    bootstrap_cell = ctx["bootstrap_cell"]
    SESSION_CELL, DATA_CELL, FINISH_CELL = (
        ctx["SESSION_CELL"], ctx["DATA_CELL"], ctx["FINISH_CELL"])

    def session(stage):
        return SESSION_CELL.replace("stage='a'", f"stage='{stage}'")

    # ------------------------------------------------------------------ NB07
    # NB07 intentionally runs before NB06. Accuracy cannot identify the three
    # Stage-B architectures on this data; the XAI gate publishes that choice.
    c = [md(r"""# NB07 — Stage-A evidence gate (run this **before NB06**)

Stage A is complete, but accuracy alone cannot choose the three architectures
for the expensive technique sweep: two folds contain a known same-tyre warning,
and each validation fold contains only four sessions. This notebook therefore
selects Stage B on **area-normalised tread/tyre evidence**, after rejecting CAM
methods that fail faithfulness or weight-randomisation checks.

If neither candidate CAM passes for one architecture, that architecture is
recorded as `excluded_no_faithful_cam` and the screen continues. The threshold
is never relaxed after seeing a failure; at least five valid architectures are
required before the three-seed confirmation step.

Checkpoint identity is checked before attribution. A saved tensor signature
that disagrees with the declared architecture is published as
`excluded_checkpoint_arch_mismatch` and the remaining screen continues. This
quarantines the nine historical `convnextv2_s` run ids whose public checkpoints
are actually ResNet-18, without relabelling them or letting them enter Stage B.

It downloads one checkpoint at a time from the public Hugging Face dataset into
`/kaggle/temp`, deletes each local copy after use, and publishes:

* `tables/xai_evidence_all.csv`
* `tables/xai_faithfulness.csv`
* `tables/stage_b_selection.csv` — the only file NB06 accepts

The screen uses fold 1 (the only fold without the known cross-fold tyre flag),
seed 1 for every architecture, then confirms the five best screens with seeds
2 and 3. No accuracy value enters the top-three rule.
"""),
         code(bootstrap_cell()),
         md("## 1 — Session, data, and verified masks"), code(session("xai")),
         code(DATA_CELL),
         code(r'''assert ANN_ROOT is not None, "annotations/ not found in the attached dataset"
ANN = tl.ensure_annotations(DATA_ROOT, ann_root=ANN_ROOT,
                            work_dir=sess.stage_dir / "annotations")

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "grad-cam"], check=True)
print("annotation fingerprint is verified by NBT1; XAI dependencies ready")
'''),
         md("## 2 — Lock the hypotheses before reading XAI results"),
         code(r'''import json, time
from pathlib import Path
from huggingface_hub import hf_hub_download

analysis_dir = Path(sess.stage_dir) / "analysis"
analysis_dir.mkdir(parents=True, exist_ok=True)
hyp_path = analysis_dir / "hypotheses.json"
try:
    remote = hf_hub_download(tl.HF_REPO_DEFAULT, "analysis/hypotheses.json",
                             repo_type="dataset", token=None,
                             local_dir=str(Path(sess.stage_dir) / "public_pull"))
    HYPOTHESES = json.loads(Path(remote).read_text())
    print("Existing preregistration found on public HF; it remains locked.")
except Exception:
    HYPOTHESES = {
      "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      "H1": "TER_norm predicts cross-fold macro-F1 stability better than validation accuracy.",
      "H2": "High SAR/DmgAR predicts larger recall loss when markings/damage are masked.",
      "H3": "Fine-grained architectures (bilinear / attention-bilinear) have higher TER_norm than plain classifiers at matched accuracy.",
    }
    hyp_path.write_text(json.dumps(HYPOTHESES, indent=2))
    sess.uploader.enqueue(hyp_path, "analysis/hypotheses.json", force=True)
    sess.push_now("XAI hypotheses preregistered")
print(json.dumps(HYPOTHESES, indent=2))
'''),
         md("## 3 — Fixed checkpoint plan"),
         code(r'''import pandas as pd
STAGE_A_ARCHS = [a for a in tl.ZOO if a != "resnet18"]
SCREEN_RUNS = [f"a-{a}-base-f1-s1" for a in STAGE_A_ARCHS]
print(f"screen: {len(SCREEN_RUNS)} checkpoints (fold 1, seed 1)")
print("confirmation: seeds 2 and 3 for the five highest TER_norm screens")
print("checkpoints are public and are pulled one at a time; HF_TOKEN is used only for result pushes")
'''),
         md("## 4 — Faithfulness-select a CAM method, then screen every architecture"),
         code(r'''import gc, shutil, torch, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
XAI_REVISION = "2026-08-30-r3"
pull_root = Path(sess.stage_dir) / "xai_pull"
table_dir = Path(sess.stage_dir) / "tables"
example_dir = analysis_dir / "xai_examples"
table_dir.mkdir(parents=True, exist_ok=True); example_dir.mkdir(parents=True, exist_ok=True)

def pull_checkpoint(run_id):
    rel = f"runs/{run_id}/checkpoints/ckpt_best.pt"
    return Path(hf_hub_download(tl.HF_REPO_DEFAULT, rel, repo_type="dataset",
                                token=None, local_dir=str(pull_root)))

def sample_validation(fold, per_class=20):
    _, va = tl.load_split(DATA_ROOT, fold)
    return (va.sort_values("image_id").groupby("proxy_label", group_keys=False)
              .head(per_class).reset_index(drop=True))

def probs_for(logits, head):
    return tl.CoralHead.probs(logits) if head == "coral" else logits.softmax(1)

XAI_NUMERIC_COLUMNS = (
    "predicted_class", "ter", "ter_norm", "bar", "sar", "dmgar", "edi",
    "tread_area_frac", "peak_in_tread",
)

def coerce_evidence_numeric(frame):
    """Normalise local and resumed CSVs at the persistence boundary.

    tyrelib uses the literal ``NA`` for a region metric that is undefined.
    A single such value makes pandas load the whole column as ``object``;
    groupby.mean then tries to add strings and floats. Undefined measurements
    are missing numeric observations, so coerce them to NaN before filtering,
    ranking, saving a new per-run file, or combining sessions.
    """
    out = frame.copy()
    for column in XAI_NUMERIC_COLUMNS:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out

_mixed_numeric_selftest = coerce_evidence_numeric(pd.DataFrame({
    "arch": ["a", "a"], "ter_norm": [1.25, "NA"], "bar": [0.2, "NA"],
}))
assert _mixed_numeric_selftest.ter_norm.dtype.kind in "fc"
assert abs(_mixed_numeric_selftest.groupby("arch").ter_norm.mean().iloc[0] - 1.25) < 1e-12
del _mixed_numeric_selftest

class CheckpointCompatibilityError(RuntimeError):
    def __init__(self, ck, cfg, actual_arch, reason):
        super().__init__(reason)
        self.ck = ck
        self.cfg = cfg
        self.actual_arch = actual_arch

def load_run(run_id):
    ck = pull_checkpoint(run_id)
    st = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = st["config"]
    actual_arch = tl.infer_checkpoint_architecture(st["model"])
    if actual_arch != "unknown" and actual_arch != cfg["arch"]:
        raise CheckpointCompatibilityError(
            ck, cfg, actual_arch,
            f"declared {cfg['arch']}, but tensor signature is {actual_arch}"
        )
    try:
        model = tl.build_model(cfg["arch"], 3, pretrained=False,
                               head=cfg["head_type"], img_size=cfg["input_resolution"])
        model.load_state_dict(st["model"], strict=True)
    except Exception as e:
        raise CheckpointCompatibilityError(
            ck, cfg, actual_arch,
            f"checkpoint cannot reconstruct {cfg['arch']}: {type(e).__name__}: {e}"
        ) from e
    return ck, cfg, model.to(dev).eval()

def choose_method(model, cfg, sub):
    methods = (["gradcam", "layercam"] if cfg["arch"] in tl.IS_TRANSFORMER
               else ["gradcam", "hirescam"])
    tf = tl.build_transforms(cfg["input_resolution"], False,
                             cfg.get("preprocessing", "raw"))
    rows = []
    # Sanity uses two real images and the predicted class as the target.
    xs = []
    for r in sub.head(2).itertuples():
        xs.append(tf(Image.open(DATA_ROOT / r.relative_path).convert("RGB")))
    batch = torch.stack(xs).to(dev)
    with torch.no_grad(): pred = probs_for(model(batch), cfg["head_type"]).argmax(1).tolist()
    targets = [tl.ClassProbabilityTarget(k, cfg["head_type"]) for k in pred]

    for method in methods:
        cam, tag = tl.make_cam(model, cfg["arch"], method)
        if cam is None:
            rows.append({"method": method, "tag": tag, "sanity_delta": np.nan,
                         "insertion_auc": np.nan, "deletion_auc": np.nan}); continue
        try:
            sanity = tl.randomisation_sanity(model, cfg["arch"], batch, method, targets)
            ins, dele = [], []
            for r in sub.head(6).itertuples():
                x = tf(Image.open(DATA_ROOT / r.relative_path).convert("RGB")).unsqueeze(0).to(dev)
                with torch.no_grad(): target = int(probs_for(model(x), cfg["head_type"]).argmax(1))
                sal = cam(input_tensor=x,
                          targets=[tl.ClassProbabilityTarget(target, cfg["head_type"])])[0]
                ins.append(tl.insertion_deletion(model, x, sal, target, steps=8,
                                                  mode="insertion", head_type=cfg["head_type"]))
                dele.append(tl.insertion_deletion(model, x, sal, target, steps=8,
                                                   mode="deletion", head_type=cfg["head_type"]))
            rows.append({"method": method, "tag": tag, "sanity_delta": float(sanity),
                         "insertion_auc": float(np.mean(ins)),
                         "deletion_auc": float(np.mean(dele))})
        except Exception as e:
            rows.append({"method": method, "tag": f"{tag}; {type(e).__name__}: {e}",
                         "sanity_delta": np.nan, "insertion_auc": np.nan,
                         "deletion_auc": np.nan})
        finally:
            del cam; torch.cuda.empty_cache()
    d, chosen = tl.cam_method_gate(rows, sanity_threshold=0.05,
                                   revision=XAI_REVISION)
    if chosen is None:
        print(f"\nEXCLUDE {cfg['arch']}: no CAM method passed the locked "
              f"randomisation + faithfulness gate\n{d.to_string(index=False)}")
        return d, None
    return d, chosen

def existing_evidence(run_id, need_faith=False):
    """Resume analysis from public HF; a stopped Kaggle session loses nothing."""
    try:
        ep = hf_hub_download(tl.HF_REPO_DEFAULT, f"runs/{run_id}/xai/evidence.csv",
                             repo_type="dataset", token=None, local_dir=str(pull_root))
        old_ev = coerce_evidence_numeric(pd.read_csv(ep))
        if "xai_revision" not in old_ev or not old_ev.xai_revision.eq(XAI_REVISION).all():
            return None, None, None
        old_faith = None; old_method = None
        if need_faith:
            fp = hf_hub_download(tl.HF_REPO_DEFAULT, f"runs/{run_id}/xai/faithfulness.csv",
                                 repo_type="dataset", token=None, local_dir=str(pull_root))
            old_faith = pd.read_csv(fp)
            if ("xai_revision" not in old_faith or
                    not old_faith.xai_revision.eq(XAI_REVISION).all()):
                return None, None, None
            selected = (old_faith.selected if old_faith.selected.dtype == bool else
                        old_faith.selected.astype(str).str.lower().eq("true"))
            chosen = old_faith[selected]
            old_method = str(chosen.iloc[0].method) if len(chosen) else None
        print("RESUME XAI", run_id, f"({len(old_ev)} rows already on HF)")
        return old_ev, old_faith, old_method
    except Exception:
        return None, None, None

def checkpoint_exclusion(run_id, err):
    """Publish an invalid-checkpoint result and let the remaining screen run."""
    cfg, actual = err.cfg, err.actual_arch
    status = ("excluded_checkpoint_arch_mismatch" if actual != "unknown" else
              "excluded_checkpoint_incompatible")
    print(f"\nEXCLUDE {cfg['arch']}: {err}")
    xdir = Path(sess.stage_dir) / "runs" / run_id / "xai"
    xdir.mkdir(parents=True, exist_ok=True)
    faith = pd.DataFrame([{
        "run_id": run_id, "arch": cfg["arch"], "method": "NONE",
        "tag": str(err), "sanity_delta": np.nan, "insertion_auc": np.nan,
        "deletion_auc": np.nan, "faithfulness": np.nan,
        "passes_sanity": False, "passes_faithfulness": False,
        "xai_revision": XAI_REVISION, "selected": False,
        "gate_status": "failed", "checkpoint_inferred_arch": actual,
    }])
    ev_out = pd.DataFrame([{
        "run_id": run_id, "arch": cfg["arch"], "fold": cfg["fold"],
        "seed": cfg["seed"], "image_id": "", "proxy_label": "",
        "predicted_class": np.nan, "method": "NONE",
        "xai_revision": XAI_REVISION, "xai_status": status,
        "checkpoint_inferred_arch": actual,
        **{k: np.nan for k in ("ter", "ter_norm", "bar", "sar", "dmgar",
                               "edi", "tread_area_frac", "peak_in_tread")},
    }])
    ev_out.to_csv(xdir / "evidence.csv", index=False)
    faith.to_csv(xdir / "faithfulness.csv", index=False)
    sess.uploader.enqueue(xdir / "evidence.csv",
                          f"runs/{run_id}/xai/evidence.csv", force=True)
    sess.uploader.enqueue(xdir / "faithfulness.csv",
                          f"runs/{run_id}/xai/faithfulness.csv", force=True)
    sess.push_now(f"XAI checkpoint exclusion {run_id}")
    with __import__("contextlib").suppress(Exception): err.ck.unlink()
    torch.cuda.empty_cache(); gc.collect()
    return ev_out, faith, None

def evidence_run(run_id, method=None, select_method=False):
    try:
        ck, cfg, model = load_run(run_id)
    except CheckpointCompatibilityError as e:
        return checkpoint_exclusion(run_id, e)
    sub = sample_validation(cfg["fold"])
    faith = None
    if select_method:
        faith, method = choose_method(model, cfg, sub)
        faith.insert(0, "run_id", run_id); faith.insert(1, "arch", cfg["arch"])
    xdir = Path(sess.stage_dir) / "runs" / run_id / "xai"; xdir.mkdir(parents=True, exist_ok=True)
    if select_method and method is None:
        # A failed explanation gate excludes this architecture from TER-based
        # selection; it is a measured result, not a fatal notebook error and
        # not permission to relax the preregistered threshold after seeing it.
        ev_out = pd.DataFrame([{
            "run_id": run_id, "arch": cfg["arch"], "fold": cfg["fold"],
            "seed": cfg["seed"], "image_id": "", "proxy_label": "",
            "predicted_class": np.nan, "method": "NONE",
            "xai_revision": XAI_REVISION,
            "xai_status": "excluded_no_faithful_cam",
            **{k: np.nan for k in ("ter", "ter_norm", "bar", "sar", "dmgar",
                                   "edi", "tread_area_frac", "peak_in_tread")}}])
        ev_out.to_csv(xdir / "evidence.csv", index=False)
        faith.to_csv(xdir / "faithfulness.csv", index=False)
        sess.uploader.enqueue(xdir / "evidence.csv",
                              f"runs/{run_id}/xai/evidence.csv", force=True)
        sess.uploader.enqueue(xdir / "faithfulness.csv",
                              f"runs/{run_id}/xai/faithfulness.csv", force=True)
        sess.push_now(f"XAI gate exclusion {run_id}")
        del model; torch.cuda.empty_cache(); gc.collect()
        with __import__("contextlib").suppress(Exception): ck.unlink()
        return ev_out, faith, None
    cam, tag = tl.make_cam(model, cfg["arch"], method)
    if cam is None: raise RuntimeError(f"{run_id}: {tag}")
    tf = tl.build_transforms(cfg["input_resolution"], False,
                             cfg.get("preprocessing", "raw"))
    rows = []; example = None
    for r in tqdm(list(sub.itertuples()), desc=run_id, leave=False):
        mask = tl.load_mask(ANN, r.image_id, r.image_kind)
        x = tf(Image.open(DATA_ROOT / r.relative_path).convert("RGB")).unsqueeze(0).to(dev)
        with torch.no_grad():
            pr = probs_for(model(x), cfg["head_type"]); target = int(pr.argmax(1))
        sal = cam(input_tensor=x,
                  targets=[tl.ClassProbabilityTarget(target, cfg["head_type"])])[0]
        rows.append({"run_id": run_id, "arch": cfg["arch"], "fold": cfg["fold"],
                     "seed": cfg["seed"], "image_id": r.image_id,
                     "proxy_label": r.proxy_label, "predicted_class": target,
                     "method": tag, "xai_revision": XAI_REVISION, "xai_status": "ok",
                     **tl.evidence_metrics(sal, mask)})
        if example is None: example = (r, sal, mask)
    if example is not None and select_method:
        import matplotlib.pyplot as plt
        r, sal, mask = example
        img = np.asarray(Image.open(DATA_ROOT / r.relative_path).convert("RGB"))
        import cv2
        heat = cv2.resize(sal.astype("float32"), (img.shape[1], img.shape[0]))
        # The HF repo is public. Publish only the evidence-bearing tyre crop,
        # not workshop or vehicle background irrelevant to the experiment.
        ys, xs = np.where(tl.region_tyre(mask))
        if len(xs):
            pad = max(2, int(.05 * max(ys.max()-ys.min(), xs.max()-xs.min())))
            y0,y1=max(0,ys.min()-pad),min(mask.shape[0],ys.max()+pad+1)
            x0,x1=max(0,xs.min()-pad),min(mask.shape[1],xs.max()+pad+1)
            img, heat = img[y0:y1, x0:x1], heat[y0:y1, x0:x1]
        fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(img); ax.imshow(heat, cmap="jet", alpha=.45)
        ax.set_title(f"{cfg['arch']} — {method}"); ax.axis("off"); fig.tight_layout()
        fig.savefig(example_dir / f"{cfg['arch']}.png", dpi=140); plt.close(fig)
    ev_out = coerce_evidence_numeric(pd.DataFrame(rows))
    ev_out.to_csv(xdir / "evidence.csv", index=False)
    sess.uploader.enqueue(xdir / "evidence.csv", f"runs/{run_id}/xai/evidence.csv", force=True)
    if faith is not None:
        faith.to_csv(xdir / "faithfulness.csv", index=False)
        sess.uploader.enqueue(xdir / "faithfulness.csv",
                              f"runs/{run_id}/xai/faithfulness.csv", force=True)
    sess.maybe_push(f"XAI checkpoint {run_id}")
    del model, cam; torch.cuda.empty_cache(); gc.collect()
    with __import__("contextlib").suppress(Exception): ck.unlink()
    return ev_out, faith, method

screen_frames, faith_frames, METHOD = [], [], {}
for rid in SCREEN_RUNS:
    ev_, fa_, method_ = existing_evidence(rid, need_faith=True)
    if ev_ is None:
        ev_, fa_, method_ = evidence_run(rid, select_method=True)
    screen_frames.append(ev_); faith_frames.append(fa_)
    if method_ is not None: METHOD[ev_.arch.iloc[0]] = method_

screen = coerce_evidence_numeric(pd.concat(screen_frames, ignore_index=True))
faith = pd.concat(faith_frames, ignore_index=True)
if "xai_status" not in screen: screen["xai_status"] = "ok"
screen["xai_status"] = screen.xai_status.fillna("ok")
screen_ok = screen[(screen.xai_status == "ok") & screen.ter_norm.notna()].copy()
excluded_archs = sorted(set(screen.arch) - set(screen_ok.arch))
screen_rank = screen.groupby("arch").agg(
    n_total=("ter_norm", "size"), n_valid=("ter_norm", "count"))
rank_values = screen_ok.groupby("arch").agg(
    ter_norm=("ter_norm", "mean"), bar=("bar", "mean"))
screen_rank = screen_rank.join(rank_values, how="inner")
screen_rank["coverage"] = screen_rank.n_valid / screen_rank.n_total.clip(lower=1)
screen_rank = screen_rank.sort_values(["ter_norm", "bar"], ascending=[False, True])
if len(screen_rank) < 5:
    raise RuntimeError(f"only {len(screen_rank)} architectures have a faithful CAM; "
                       "need at least five before seed confirmation")
SHORT5 = list(screen_rank.head(5).index)
print("\nfaithfulness-gate exclusions:", excluded_archs or "none")
print("\nconfirmation shortlist:", SHORT5)
'''),
         md("## 5 — Confirm the shortlist across all three seeds and publish the gate"),
         code(r'''confirm_frames = []
for arch in SHORT5:
    for seed in (2, 3):
        rid = f"a-{arch}-base-f1-s{seed}"
        ev_, _, _ = existing_evidence(rid, need_faith=False)
        if ev_ is None:
            ev_, _, _ = evidence_run(rid, method=METHOD[arch], select_method=False)
        confirm_frames.append(ev_)

ev = coerce_evidence_numeric(pd.concat(screen_frames + confirm_frames,
                                        ignore_index=True))
if "xai_status" not in ev: ev["xai_status"] = "ok"
ev["xai_status"] = ev.xai_status.fillna("ok")
faith.to_csv(table_dir / "xai_faithfulness.csv", index=False)
ev.to_csv(table_dir / "xai_evidence_all.csv", index=False)

ok_ev = ev[ev.xai_status == "ok"].copy()
coverage = ok_ev.groupby("arch").agg(
    n_images=("image_id", "size"), n_valid=("ter_norm", "count"))
valid_ev = ok_ev[ok_ev.ter_norm.notna()].copy()
summary = (valid_ev.groupby("arch").agg(
    seeds=("seed", "nunique"),
    ter_norm=("ter_norm", "mean"), ter_sd=("ter_norm", "std"),
    bar=("bar", "mean"), sar=("sar", "mean"), dmgar=("dmgar", "mean"))
    .join(coverage, how="left")
    .sort_values(["ter_norm", "bar"], ascending=[False, True]).reset_index())
summary["xai_status"] = "ok"
summary["xai_coverage"] = summary.n_valid / summary.n_images.clip(lower=1)
if excluded_archs:
    excluded_status = (screen.loc[screen.arch.isin(excluded_archs),
                                  ["arch", "xai_status"]]
                       .drop_duplicates("arch").set_index("arch")["xai_status"])
    excluded = pd.DataFrame({"arch": excluded_archs, "seeds": 0, "n_images": 0,
                             "n_valid": 0,
                             "ter_norm": np.nan, "ter_sd": np.nan, "bar": np.nan,
                             "sar": np.nan, "dmgar": np.nan,
                             "xai_coverage": 0.0,
                             "xai_status": [excluded_status.get(a, "excluded")
                                            for a in excluded_archs]})
    summary = pd.concat([summary, excluded], ignore_index=True)

# Only the five architectures confirmed with all three seeds can win.
eligible = summary[(summary.seeds == 3) & summary.xai_status.eq("ok")].copy()
assert len(eligible) >= 3, "fewer than three architectures survived three-seed confirmation"
TOP3 = list(eligible.head(3).arch)
summary["eligible"] = summary.seeds.eq(3) & summary.xai_status.eq("ok")
summary["selected_top3"] = summary.arch.isin(TOP3)
summary["selection_revision"] = XAI_REVISION
summary["selection_rule"] = "top TER_norm among five seed-confirmed screens; BAR tie-break; accuracy excluded"
summary.to_csv(table_dir / "stage_b_selection.csv", index=False)

for name in ("xai_faithfulness.csv", "xai_evidence_all.csv", "stage_b_selection.csv"):
    sess.uploader.enqueue(table_dir / name, f"tables/{name}", force=True)
sess.uploader.enqueue_dir(example_dir, "analysis/xai_examples", force=True)
sess.push_now("Stage-B XAI selection locked")

print(summary.round(4).to_string(index=False))
print("\nLOCKED Stage-B architectures:", TOP3)
assert len(TOP3) == 3
'''),
         md("## 6 — Finish and verify the published selection"),
         code(r'''from huggingface_hub import HfApi
files = set(HfApi().list_repo_files(tl.HF_REPO_DEFAULT, repo_type="dataset"))
for rel in ("tables/xai_evidence_all.csv", "tables/xai_faithfulness.csv",
            "tables/stage_b_selection.csv"):
    print("yes" if rel in files else "MISSING", rel)
assert "tables/stage_b_selection.csv" in files, "selection push not visible; do not start NB06"
sess.finish()
''')]
    write_nb("NB07_XAI.ipynb", c)

    # ------------------------------------------------------------------ NB06
    c = [md(r"""# NB06 — Stage B technique sweep (run **after NB07**)

This notebook has no fallback architecture list. It downloads the locked XAI
selection from the public HF dataset and stops if that file is absent. That
gate prevents an arbitrary accuracy ranking from launching hundreds of runs.
The completed public `2026-08-30-r3` gate selected `regnety016`,
`densenet121`, and `resnet50`; cell 2 still re-downloads and verifies the
selection and its raw evidence so the notebook never relies on this prose.

**Memory-repair revision v5 (2026-08-31):** public telemetry proved the kernel
deaths were host-RAM exhaustion, not GPU OOM. The earlier ROI-only diagnosis was
incomplete: full-frame arms also accumulated process RSS. Large temporary
checkpoint/upload arenas were being left mapped in the long-lived Python
process. v5 serialises one full checkpoint per epoch (the best file is an atomic
snapshot, not a second serialisation), releases free Linux arenas after saves,
loads and HF commits, and stops the whole worker after a clean 88% RAM pause.
The ROI crop/loader repairs remain. The locked architectures, batch size,
resolution, optimiser, and 60-epoch scientific recipe are unchanged.

**Epoch-history repair v6 (2026-09-01):** v5 added a commit-policy telemetry
field while two 60-epoch runs still had v4 CSV headers. Positional append wrote
178 values under 177 headings, so pandas stopped resume with `ParserError` even
though both checkpoints were intact. v6 recognises that exact revision marker,
inserts a blank for the older rows, verifies every row width, and atomically
rewrites the canonical table. Future epoch writes merge by column name and can
expand the header safely. Unknown drift raises without modifying the source
file; no epoch or metric is silently dropped.

**CUDA/scheduler/commit repair revision (2026-08-31):** two independent public
RegNetY-16GF ROI attempts failed on their first batch in the same cuDNN grouped
convolution with `CUDNN_STATUS_EXECUTION_FAILED` / `misaligned address`, while
each T4 held only about 1.1 GB. RegNet therefore keeps the exact same model,
384px input, batch 32, AMP, and optimiser but uses the conservative contiguous
(NCHW) cuDNN path instead of `channels_last`; all other architectures keep the
Stage-A layout. A fatal CUDA context now stops the session only after publishing
the error, rather than cascading into unrelated runs. Fresh work is also kept
with its static owner: an absent run is no longer mislabelled as a dead worker
and stolen by all four accounts at once. Work stealing is now opt-in and is
disabled here. An ordinary static-owner claim is batched into the 30-minute HF
cycle; it no longer burns one immediate commit before every model. A paused run
ends the training cell instead of cascading through dozens of one-epoch runs.

Before planning hours of work, cell 2 runs one exact dual-T4 RegNet
forward/backward/optimizer step in an **isolated child process** and publishes
the log. If the Kaggle CUDA image still rejects the conservative profile, only
the child process is poisoned and NB06 stops before claiming a training run.

**Stop every older v4/v5 NB06 copy before starting v6.** For four Kaggle copies set
only `ACCOUNT` to `acct1`, `acct2`, `acct3`, and `acct4`; leave
`NUM_WORKERS=4`. `WORKER_ID` is derived automatically. Cell 4 reads the current
public HF state: completed runs are skipped and every partial run resumes from
its published `ckpt_last.pt`. No architecture or completed epoch is discarded.

Every arm below changes one declared factor relative to the Stage-A recipe.
The tyre-ROI arm runs first. Unsupported values fail before training rather
than becoming silent no-op experiments. Until the tyre groups are re-cut, the
sweep runs **fold 1 only** (three seeds): folds 0 and 2 carry the known
cross-fold-tyre warning and are already saturated, so spending two-thirds of
the Stage-B budget there cannot measure a technique effect.
"""), code(bootstrap_cell()),
         md("## 1 — Session and data"), code(session("b")), code(DATA_CELL),
         md("## 2 — Load the locked Stage-B selection"),
         code(r'''import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download

try:
    p = hf_hub_download(tl.HF_REPO_DEFAULT, "tables/stage_b_selection.csv",
                        repo_type="dataset", token=None,
                        local_dir=str(Path(sess.stage_dir) / "public_pull"))
except Exception as e:
    raise RuntimeError("NB07 has not published tables/stage_b_selection.csv. "
                       "Run the corrected NB07 first; NB06 is intentionally blocked.") from e
SEL = pd.read_csv(p)
assert ("selection_revision" in SEL and
        SEL.selection_revision.eq("2026-08-30-r3").all()), \
       "stale Stage-B selection; rerun the corrected NB07"
flag = (SEL.selected_top3 if SEL.selected_top3.dtype == bool else
        SEL.selected_top3.astype(str).str.lower().eq("true"))
TOP3 = list(SEL.loc[flag, "arch"])
assert len(TOP3) == 3, f"expected exactly three locked architectures, got {TOP3}"
elig = SEL.loc[flag, "eligible"]
elig = (elig if elig.dtype == bool else elig.astype(str).str.lower().eq("true"))
assert elig.all() and \
       SEL.loc[flag, "seeds"].astype(int).eq(3).all(), \
       "selection is not three-seed confirmed; rerun NB07"
assert SEL.loc[flag, "xai_status"].eq("ok").all(), \
       "a selected architecture is not XAI-valid"

# Independently verify the gate from the raw public evidence rather than
# trusting only the summary booleans. This also reports the true denominator;
# an older NB07 table counted only valid rows in both n_images and n_valid.
ep = hf_hub_download(tl.HF_REPO_DEFAULT, "tables/xai_evidence_all.csv",
                     repo_type="dataset", token=None,
                     local_dir=str(Path(sess.stage_dir) / "public_pull"))
EV = pd.read_csv(ep)
EV["ter_norm"] = pd.to_numeric(EV.ter_norm, errors="coerce")
raw_selected = EV[EV.arch.isin(TOP3) & EV.xai_status.eq("ok")].copy()
raw_coverage = raw_selected.groupby("arch").agg(
    seeds=("seed", "nunique"), n_images=("image_id", "size"),
    n_valid=("ter_norm", "count"))
raw_coverage["coverage"] = raw_coverage.n_valid / raw_coverage.n_images.clip(lower=1)
assert set(raw_coverage.index) == set(TOP3), "selected evidence rows are missing"
assert raw_coverage.seeds.astype(int).eq(3).all(), \
       "selected raw evidence does not contain all three seeds"
assert raw_coverage.n_valid.gt(0).all(), "selected architecture has no valid TER_norm evidence"
print(SEL.round(4).to_string(index=False))
print("\nlocked architectures:", TOP3)
print("\nraw selected-evidence coverage:\n", raw_coverage.round(4).to_string())
tl.assert_zoo_ok(TOP3)
print("\nKaggle CUDA runtime layouts (model/config unchanged):")
for arch in TOP3:
    print(f"  {arch:14s} {tl.training_memory_format(arch)}")

# Fatal CUDA launch errors poison a process, so prove the repaired RegNet
# profile in a disposable child before any run is claimed. This is an exact
# dual-GPU training step at the registered batch/resolution, not a one-image
# CPU construction check.
import subprocess, sys
from pathlib import Path
CUDA_PROBE = r"""
import sys, torch, tyrelib as tl
arch, res, bs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dev = torch.device("cuda")
assert torch.cuda.device_count() >= 2, "NB06 requires Kaggle dual T4"
fmt_name = tl.training_memory_format(arch)
fmt = torch.contiguous_format if fmt_name == "contiguous" else torch.channels_last
torch.backends.cudnn.benchmark = fmt_name == "channels_last"
torch.manual_seed(20260831); torch.cuda.manual_seed_all(20260831)
model = tl.build_model(arch, 3, pretrained=False, head="coral",
                       img_size=res).to(dev).to(memory_format=fmt)
model = torch.nn.DataParallel(model).train()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.05)
scaler = tl._grad_scaler(dev)
x = torch.randn(bs, 3, res, res).to(dev, non_blocking=True).to(memory_format=fmt)
y = torch.arange(bs, device=dev) % 3
opt.zero_grad(set_to_none=True)
with tl._autocast(dev):
    loss = tl.CoralHead.loss(model(x), y)
scaler.scale(loss).backward(); scaler.unscale_(opt)
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
scaler.step(opt); scaler.update(); torch.cuda.synchronize()
print(f"CUDA_SMOKE_PASS arch={arch} res={res} batch={bs} devices=2 "
      f"layout={fmt_name} cudnn_benchmark={torch.backends.cudnn.benchmark} "
      f"loss={float(loss.detach()):.6f} safety={tl.CUDA_SAFETY_REVISION}")
"""
for arch in sorted(set(TOP3) & set(tl.CUDA_CONTIGUOUS_ARCHS)):
    spec = tl.ZOO[arch]
    result = subprocess.run(
        [sys.executable, "-c", CUDA_PROBE, arch, str(spec["res"]), str(spec["bs"])],
        cwd=str(Path.cwd()), text=True, capture_output=True, timeout=600)
    probe_log = (result.stdout + "\n" + result.stderr).strip() + "\n"
    probe_path = Path(sess.stage_dir) / "preflight" / f"cuda_{arch}_{ACCOUNT}.txt"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    probe_path.write_text(probe_log)
    sess.uploader.enqueue(probe_path,
                          f"preflight/cuda_profiles/{probe_path.name}", force=True)
    sess.push_now(f"CUDA profile preflight {arch} {ACCOUNT}")
    print(probe_log)
    if result.returncode:
        raise RuntimeError(
            f"isolated CUDA training smoke failed for {arch}; log published to "
            f"preflight/cuda_profiles/{probe_path.name}. No run was claimed."
        )
'''),
         md("## 3 — Verify masks once for the ROI control"),
         code(r'''assert ANN_ROOT is not None, "annotations are required for roi_tyre"
ANN = tl.ensure_annotations(DATA_ROOT, ann_root=ANN_ROOT,
                            work_dir=sess.stage_dir / "annotations")
MASK_ROOTS = {"clean_mask_root": str(ANN["clean_masks"]),
              "propagated_mask_root": str(ANN["propagated_masks"])}
'''),
         md("## 4 — Build and validate the one-factor arms"),
         code(r'''# Base = CORAL, session-balanced, native resolution, full frame,
# ImageNet/SSL initialisation, full fine-tune, raw colour, wd=.05, lr=3e-4.
FACTOR_ARMS = {
  "roi_tyre": dict(roi_mode="tyre_crop", **MASK_ROOTS),
  "head_ce": dict(head_type="ce", loss_name="cross_entropy"),
  "sampler_uniform": dict(sampler_name="uniform"),
  "sampler_classweighted": dict(sampler_name="class_weighted"),
  "res224": dict(input_resolution=224),
  "res512": dict(input_resolution=512, batch_size=16),
  "transfer_random": dict(pretrained=False),
  "ft_frozen": dict(finetune_depth="frozen"),
  "wd_low": dict(weight_decay=0.01),
  "prep_clahe": dict(preprocessing="clahe"),
  "prep_gray": dict(preprocessing="grayscale"),
  "lr_low": dict(lr_initial=1e-4),
}

cfgs = []
for factor, overrides in FACTOR_ARMS.items():
    for arch in TOP3:
        if factor == "res224" and int(tl.ZOO[arch]["res"]) == 224:
            print(f"STRUCTURAL SKIP: {arch} already uses 224; no no-op res224 arm")
            continue
        if factor == "res512" and arch in tl.FIXED_224:
            print(f"STRUCTURAL SKIP: {arch} is fixed-window 224; no res512 arm")
            continue
        arm = sess.configs([arch], (1,), (1, 2, 3),
                           technique=factor, **overrides)
        for cfg in arm: tl.validate_config(cfg)
        cfgs.extend(arm)

roi_cfgs = [x for x in cfgs if x["technique"] == "roi_tyre"]
other_cfgs = [x for x in cfgs if x["technique"] != "roi_tyre"]
run_ids = [x["run_id"] for x in cfgs]
est = tl.estimate_phase(run_ids, num_workers=NUM_WORKERS)
print(f"{len(run_ids)} runs; ~{est['total_gpu_hours']:.0f} GPU-hours total")
print(tl.shard_report(run_ids, max(1, NUM_WORKERS)).to_string(index=False))

# Hugging Face is the authority. Print one compact live audit before any
# worker starts, then run_all refreshes it again at execution time.
sess.inventory.refresh(run_ids, verbose=False)
hf_done = [r for r in run_ids if sess.inventory.state(r) == "completed"]
hf_resume = [r for r in run_ids if sess.inventory.state(r) == "resumable"]
hf_absent = [r for r in run_ids if sess.inventory.state(r) == "absent"]
hf_failed = [r for r in run_ids
             if sess.inventory.status.get(r, {}).get("status") == "failed"]
print("\nPUBLIC HF Stage-B progress:",
      f"completed={len(hf_done)} resumable={len(hf_resume)} "
      f"absent={len(hf_absent)} prior_failed={len(hf_failed)}")
for rid in hf_done:
    print("  COMPLETE", rid)
for rid in hf_resume:
    print("  RESUME  ", rid, "from epoch", sess.inventory.epoch(rid))
for rid in hf_failed:
    st = sess.inventory.status[rid]
    print("  RETRY   ", rid, "after", st.get("error_type", "recorded failure"),
          "at epoch", sess.inventory.epoch(rid))
assert len(hf_done) + len(hf_resume) + len(hf_absent) == len(run_ids)
'''),
         md("## 5 — Run the shortcut-removing ROI control first"),
         code("""# steal_stale=False keeps FRESH work with its static owner (that is what
# stopped four accounts grabbing the same run at a simultaneous start).
# takeover_when_idle=True lets a worker that has finished its own shard pick up
# what is left, one run at a time, through a two-phase claim -- so no GPU sits
# parked while another account still has twenty runs. See docs/05 Bug 24.
roi_summaries = sess.run_all(roi_cfgs, title='Stage B — ROI control first',
                             steal_stale=False, takeover_when_idle=True)"""),
         md("## 6 — Run the remaining one-factor arms"),
         code("""summaries = sess.run_all(other_cfgs, title='Stage B — remaining OFAT arms',
                         steal_stale=False, takeover_when_idle=True)"""),
         md("## 7 — Effects relative to the matching Stage-A base runs"),
         code(r'''import numpy as np, pandas as pd
base_ids = [f"a-{a}-base-f1-s{s}" for a in TOP3 for s in (1,2,3)]
A = sess.aggregate_remote(base_ids, verbose=False)
B = sess.aggregate_remote(run_ids, verbose=False)
if len(A) and len(B):
    base = A.set_index(["arch", "fold", "seed"])["best_val_f1_macro"]
    rows = []
    for r in B[B.status == "completed"].itertuples():
        key = (r.arch, int(r.fold), int(r.seed))
        if key in base.index:
            rows.append({"arch": r.arch, "factor": r.technique, "fold": r.fold,
                         "seed": r.seed, "f1": r.best_val_f1_macro,
                         "delta_vs_stage_a": r.best_val_f1_macro - float(base.loc[key])})
    E = pd.DataFrame(rows)
    print(E.groupby(["arch", "factor"]).agg(
        n=("delta_vs_stage_a", "size"), mean_delta=("delta_vs_stage_a", "mean"),
        sd=("delta_vs_stage_a", "std")).round(4).to_string())
    out = Path(sess.stage_dir) / "tables" / "stage_b_effects.csv"
    E.to_csv(out, index=False); sess.uploader.enqueue(out, "tables/stage_b_effects.csv", force=True)
    sess.push_now("Stage B effects updated")
else:
    print("No completed Stage-B arms yet; rerun this cell after progress accumulates.")
'''),
         md("## 8 — Final public verification"), code(FINISH_CELL)]
    write_nb("NB06_StageB_OFAT.ipynb", c)

    # ------------------------------------------------------------------ NB08
    c = [md(r"""# NB08 — Causal shortcut stress tests

The shuffled-label control runs on all three folds and is a hard gate: if its
mean macro-F1 is above 0.45, the notebook stops before interpreting any other
result. The remaining tests use the three architectures locked by NB07, all
three seeds on fold 1, and every validation image.

Public checkpoints are downloaded one at a time into `/kaggle/temp` and removed
after evaluation, so a fresh Kaggle session needs no local state from NB07.
"""), code(bootstrap_cell()),
         md("## 1 — Session, data, masks, and locked model list"),
         code(session("stress")), code(DATA_CELL),
         code(r'''import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download

assert ANN_ROOT is not None, "annotations required for stress interventions"
ANN = tl.ensure_annotations(DATA_ROOT, ann_root=ANN_ROOT,
                            work_dir=sess.stage_dir / "annotations")
try:
    p = hf_hub_download(tl.HF_REPO_DEFAULT, "tables/stage_b_selection.csv",
                        repo_type="dataset", token=None,
                        local_dir=str(Path(sess.stage_dir) / "public_pull"))
    SEL = pd.read_csv(p)
    assert ("selection_revision" in SEL and
            SEL.selection_revision.eq("2026-08-30-r3").all())
    flag = (SEL.selected_top3 if SEL.selected_top3.dtype == bool else
            SEL.selected_top3.astype(str).str.lower().eq("true"))
    TOP3 = list(SEL.loc[flag, "arch"])
except Exception as e:
    raise RuntimeError("Run NB07 first; its locked top-three selection is missing.") from e
assert len(TOP3) == 3
print("stress-test architectures:", TOP3)
'''),
         md("## 2 — Three-fold shuffled-label control (must remain at chance)"),
         code(r'''import numpy as np
ctrl_cfgs = [sess.config("resnet18", f, 1, technique="shufflectl_r2",
                         stage="stress", max_epochs=12, input_resolution=224,
                         batch_size=32, warmup_epochs=1) for f in (0,1,2)]
ctrl_ids = [x["run_id"] for x in ctrl_cfgs]

_original_load_split = tl.load_split
def _shuffled(root, fold):
    tr, va = _original_load_split(root, fold)
    rng = np.random.default_rng(20260830 + int(fold))
    tr = tr.copy(); tr["proxy_label"] = rng.permutation(tr.proxy_label.values)
    return tr, va

tl.load_split = _shuffled
try:
    sess.run_all(ctrl_cfgs, title="three-fold shuffled-label control")
finally:
    tl.load_split = _original_load_split

C = sess.aggregate_remote(ctrl_ids, verbose=False)
if len(C) != 3 or not set(C.status).issubset({"completed"}):
    raise RuntimeError(f"control incomplete ({len(C)}/3 complete); rerun NB08 to resume it")
mean_ctl = float(C.best_val_f1_macro.mean())
print(C[["run_id", "best_val_f1_macro", "best_val_qwk"]].to_string(index=False))
print(f"\nmean shuffled-label macro-F1 = {mean_ctl:.3f}")
if mean_ctl >= 0.45:
    raise RuntimeError("SHUFFLED-LABEL CONTROL FAILED. Stop: the pipeline can recover labels from leakage.")
print("PASS — shuffled labels remain at chance")
'''),
         md("## 3 — Define interventions"),
         code(r'''import numpy as np
from PIL import Image

def _resize_like(other, shape, nearest=False):
    h, w = shape
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(other).resize((w, h), mode))

def apply_intervention(img, mask, kind, other_bg=None):
    out = img.copy()
    if kind == "bg_blank":
        out[tl.region_background(mask)] = 128
    elif kind == "bg_swap":
        other_img, other_mask = other_bg
        other = _resize_like(other_img, mask.shape)
        omask = _resize_like(other_mask.astype("uint8"), mask.shape, nearest=True)
        bg = tl.region_background(mask)
        obg = tl.region_background(omask)
        # Never paste another tyre into this image's background. Use real
        # pixels only where both masks call the location background; fill any
        # remaining band with the donor background's robust colour.
        fill = np.median(other[obg], axis=0) if obg.any() else np.array([128,128,128])
        out[bg] = np.asarray(fill, dtype=np.uint8)
        valid = bg & obg
        out[valid] = other[valid]
    elif kind == "tyre_crop":
        ys, xs = np.where(tl.region_tyre(mask))
        if len(xs):
            pad = max(2, int(.05 * max(ys.max()-ys.min(), xs.max()-xs.min())))
            y0,y1=max(0,ys.min()-pad),min(mask.shape[0],ys.max()+pad+1)
            x0,x1=max(0,xs.min()-pad),min(mask.shape[1],xs.max()+pad+1)
            out = out[y0:y1, x0:x1]
    elif kind in ("mask_marking", "mask_damage"):
        region = tl.region_marking(mask) if kind == "mask_marking" else tl.region_damage(mask)
        source = (tl.region_tread(mask) if kind == "mask_marking" else tl.region_tyre(mask)) & ~region
        fill = np.median(out[source], axis=0) if source.any() else np.array([128,128,128])
        out[region] = np.asarray(fill, dtype=np.uint8)
    elif kind == "grayscale":
        g = np.round(out.mean(2, keepdims=True)).astype(np.uint8); out = np.repeat(g, 3, 2)
    return out

INTERVENTIONS = ["none", "bg_blank", "bg_swap", "tyre_crop",
                 "mask_marking", "mask_damage", "grayscale"]
print(INTERVENTIONS)
'''),
         md("## 4 — Evaluate all selected architectures and seeds on fold 1"),
         code(r'''import gc, torch, pandas as pd
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STRESS_REVISION = "2026-08-30-r1"
pull_root = Path(sess.stage_dir) / "stress_pull"

def existing_stress(run_id):
    try:
        p = hf_hub_download(tl.HF_REPO_DEFAULT, f"runs/{run_id}/stress/results.csv",
                            repo_type="dataset", token=None, local_dir=str(pull_root))
        d = pd.read_csv(p)
        return d if "stress_revision" in d and d.stress_revision.eq(STRESS_REVISION).all() else None
    except Exception:
        return None

def eval_run(run_id):
    rel = f"runs/{run_id}/checkpoints/ckpt_best.pt"
    ck = Path(hf_hub_download(tl.HF_REPO_DEFAULT, rel, repo_type="dataset",
                              token=None, local_dir=str(pull_root)))
    st = torch.load(ck, map_location="cpu", weights_only=False); cfg = st["config"]
    model = tl.build_model(cfg["arch"], 3, pretrained=False, head=cfg["head_type"],
                           img_size=cfg["input_resolution"])
    model.load_state_dict(st["model"]); model = model.to(dev).eval()
    _, va = tl.load_split(DATA_ROOT, cfg["fold"]); va = va.sort_values("image_id")
    bg_rows = list(va.drop_duplicates("session_group").itertuples())
    backgrounds = [(r.session_group,
                    np.asarray(Image.open(DATA_ROOT/r.relative_path).convert("RGB")),
                    tl.load_mask(ANN, r.image_id, r.image_kind)) for r in bg_rows]
    tf = tl.build_transforms(cfg["input_resolution"], False, cfg.get("preprocessing", "raw"))
    rows = []
    for kind in INTERVENTIONS:
        Y, PR = [], []
        for i, r in enumerate(va.itertuples()):
            mask = tl.load_mask(ANN, r.image_id, r.image_kind)
            img = np.asarray(Image.open(DATA_ROOT/r.relative_path).convert("RGB"))
            donors = [b for b in backgrounds if b[0] != r.session_group]
            assert donors, "background swap requires at least two validation sessions"
            donor = donors[i % len(donors)]
            other = (donor[1], donor[2])
            arr = apply_intervention(img, mask, kind, other)
            x = tf(Image.fromarray(arr)).unsqueeze(0).to(dev)
            with torch.no_grad(), tl._autocast(dev): logits = model(x)
            probs = (tl.CoralHead.probs(logits.float()) if cfg["head_type"] == "coral"
                     else logits.float().softmax(1))
            PR.append(probs[0].cpu().numpy()); Y.append(tl.C2I[r.proxy_label])
        PR = np.asarray(PR); Y = np.asarray(Y); pred = PR.argmax(1)
        met, _ = tl.classification_report_dict(Y, pred, PR, "")
        rows.append({"run_id": run_id, "arch": cfg["arch"], "fold": cfg["fold"],
                     "seed": cfg["seed"], "intervention": kind, "n": len(Y),
                     "stress_revision": STRESS_REVISION,
                     **{k: met[k] for k in ("f1_macro","acc","qwk","recall_low","recall_high")}})
    del model; torch.cuda.empty_cache(); gc.collect()
    try: ck.unlink()
    except Exception: pass
    return rows

stress_ids = [f"a-{a}-base-f1-s{s}" for a in TOP3 for s in (1,2,3)]
rows = []
for rid in tqdm(stress_ids, desc="checkpoints"):
    old = existing_stress(rid)
    if old is not None:
        print("RESUME STRESS", rid); rows.extend(old.to_dict("records")); continue
    part = pd.DataFrame(eval_run(rid)); rows.extend(part.to_dict("records"))
    rd = Path(sess.stage_dir)/"runs"/rid/"stress"; rd.mkdir(parents=True,exist_ok=True)
    part.to_csv(rd/"results.csv",index=False)
    sess.uploader.enqueue(rd/"results.csv",f"runs/{rid}/stress/results.csv",force=True)
    sess.maybe_push(f"stress checkpoint {rid}")
S = pd.DataFrame(rows)
out_dir = Path(sess.stage_dir) / "tables"; out_dir.mkdir(parents=True, exist_ok=True)
S.to_csv(out_dir / "stress_tests.csv", index=False)
sess.uploader.enqueue(out_dir / "stress_tests.csv", "tables/stress_tests.csv", force=True)
sess.push_now("stress tests complete")

piv = S.pivot_table(index="arch", columns="intervention", values="f1_macro")
delta = piv.sub(piv["none"], axis=0).drop(columns="none").round(4)
print("mean delta macro-F1 vs original\n")
print(delta.to_string())
'''),
         md("## 5 — Finish"), code(r'''sess.finish()
print("NB08 complete; tables/stress_tests.csv is queued and flushed.")
''')]
    write_nb("NB08_StressTests.ipynb", c)

    # ------------------------------------------------------------------ NB09
    c = [md(r"""# NB09 — Ensembles, calibration, TTA, and conformal uncertainty

This notebook downloads only the selected Stage-A predictions/checkpoints from
the public HF dataset. It implements the four operations it claims: seed and
architecture ensembles, horizontal-flip TTA, temperature scaling on a disjoint
split, and conformal prediction sets on a second disjoint split.
"""), code(bootstrap_cell()),
         md("## 1 — Session, data, and locked architectures"),
         code(session("ens")), code(DATA_CELL),
         code(r'''import pandas as pd, numpy as np, torch
from pathlib import Path
from huggingface_hub import hf_hub_download

pull_root = Path(sess.stage_dir) / "ensemble_pull"
try:
    sp = hf_hub_download(tl.HF_REPO_DEFAULT, "tables/stage_b_selection.csv",
                         repo_type="dataset", token=None, local_dir=str(pull_root))
    SEL = pd.read_csv(sp)
    assert ("selection_revision" in SEL and
            SEL.selection_revision.eq("2026-08-30-r3").all())
    flag = (SEL.selected_top3 if SEL.selected_top3.dtype == bool else
            SEL.selected_top3.astype(str).str.lower().eq("true"))
    TOP3 = list(SEL.loc[flag, "arch"])
except Exception as e:
    raise RuntimeError("Run NB07 first; no locked top-three selection exists.") from e
assert len(TOP3) == 3
print("ensemble architectures:", TOP3)
'''),
         md("## 2 — Pull the 27 prediction files; reconstruct any derivable gap"),
         code(r'''from PIL import Image
import gc

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_or_rebuild_predictions(rid):
    """A best checkpoint is primary evidence; predictions are reproducible.

    The public audit found one historical run whose final metrics and best
    checkpoint exist but whose predictions parquet did not land. Reconstruct
    that derived file if any selected run has the same gap, publish it, and
    continue instead of making the ensemble depend on a missing upload.
    """
    rel = f"runs/{rid}/per_sample/predictions.parquet"
    try:
        return Path(hf_hub_download(tl.HF_REPO_DEFAULT, rel, repo_type="dataset",
                                    token=None, local_dir=str(pull_root)))
    except Exception as first_error:
        print("REBUILD missing predictions:", rid, type(first_error).__name__)

    ck_rel = f"runs/{rid}/checkpoints/ckpt_best.pt"
    ck = Path(hf_hub_download(tl.HF_REPO_DEFAULT, ck_rel, repo_type="dataset",
                              token=None, local_dir=str(pull_root)))
    st = torch.load(ck, map_location="cpu", weights_only=False); cfg = st["config"]
    model = tl.build_model(cfg["arch"], 3, pretrained=False, head=cfg["head_type"],
                           img_size=cfg["input_resolution"])
    model.load_state_dict(st["model"]); model = model.to(dev).eval()
    _, va = tl.load_split(DATA_ROOT, int(cfg["fold"])); va = va.sort_values("image_id")
    tf = tl.build_transforms(cfg["input_resolution"], False, cfg.get("preprocessing", "raw"))
    rows = []
    for r in va.itertuples():
        x = tf(Image.open(DATA_ROOT/r.relative_path).convert("RGB")).unsqueeze(0).to(dev)
        with torch.no_grad(), tl._autocast(dev): logits = model(x)
        pr = (tl.CoralHead.probs(logits.float()) if cfg["head_type"] == "coral"
              else logits.float().softmax(1))[0].cpu().numpy()
        rows.append({"image_id": r.image_id, "session_group": r.session_group,
                     "true": tl.C2I[r.proxy_label], "pred": int(pr.argmax()),
                     "prob_low": pr[0], "prob_mid": pr[1], "prob_high": pr[2]})
    out = Path(sess.stage_dir)/"recovered_predictions"/rid/"predictions.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    sess.uploader.enqueue(out, rel, force=True)
    sess.push_now(f"reconstructed predictions {rid}")
    del model; torch.cuda.empty_cache(); gc.collect()
    try: ck.unlink()
    except Exception: pass
    return out

frames = []
selected_ids = [f"a-{a}-base-f{f}-s{s}" for a in TOP3 for f in (0,1,2) for s in (1,2,3)]
for rid in selected_ids:
    p = get_or_rebuild_predictions(rid)
    d = pd.read_parquet(p); d["run_id"] = rid
    d["arch"] = rid.split("-")[1]; d["fold"] = int(rid.split("-f")[1].split("-")[0])
    d["seed"] = int(rid.rsplit("-s",1)[1]); frames.append(d)
P = pd.concat(frames, ignore_index=True)
PROB = ["prob_low", "prob_mid", "prob_high"]
print(f"{len(P)} rows from {P.run_id.nunique()} runs")
'''),
         md("## 3 — Single, seed-ensemble, and architecture-ensemble metrics"),
         code(r'''def score_groups(df, group, label):
    rows = []
    for keys, g in df.groupby(group):
        avg = g.groupby("image_id")[PROB].mean()
        truth = g.groupby("image_id")["true"].first().loc[avg.index].astype(int)
        met, _ = tl.classification_report_dict(truth.values, avg.values.argmax(1), avg.values, "")
        rows.append({"kind": label, "group": str(keys), "n": len(avg),
                     **{k: met[k] for k in ("f1_macro","qwk","ece","nll","brier")}})
    return pd.DataFrame(rows)

M = pd.concat([score_groups(P, ["arch","fold","seed"], "single"),
               score_groups(P, ["arch","fold"], "seed_ensemble"),
               score_groups(P, ["fold"], "architecture_ensemble")], ignore_index=True)
print(M.groupby("kind")[["f1_macro","qwk","ece","nll"]].mean().round(4).to_string())
'''),
         md("## 4 — Horizontal-flip TTA on one fixed checkpoint per architecture"),
         code(r'''def tta_one(arch):
    rid = f"a-{arch}-base-f1-s1"; rel = f"runs/{rid}/checkpoints/ckpt_best.pt"
    ck = Path(hf_hub_download(tl.HF_REPO_DEFAULT, rel, repo_type="dataset",
                              token=None, local_dir=str(pull_root)))
    st = torch.load(ck, map_location="cpu", weights_only=False); cfg = st["config"]
    model = tl.build_model(arch, 3, pretrained=False, head=cfg["head_type"],
                           img_size=cfg["input_resolution"])
    model.load_state_dict(st["model"]); model = model.to(dev).eval()
    _, va = tl.load_split(DATA_ROOT, 1); va = va.sort_values("image_id")
    tf = tl.build_transforms(cfg["input_resolution"], False, cfg.get("preprocessing", "raw"))
    raw, tta, y = [], [], []
    for r in va.itertuples():
        img = Image.open(DATA_ROOT/r.relative_path).convert("RGB")
        x0 = tf(img).unsqueeze(0).to(dev); x1 = torch.flip(x0, dims=[3])
        with torch.no_grad(), tl._autocast(dev): l0, l1 = model(x0), model(x1)
        p0 = tl.CoralHead.probs(l0.float()) if cfg["head_type"]=="coral" else l0.float().softmax(1)
        p1 = tl.CoralHead.probs(l1.float()) if cfg["head_type"]=="coral" else l1.float().softmax(1)
        raw.append(p0[0].cpu().numpy()); tta.append(((p0+p1)/2)[0].cpu().numpy())
        y.append(tl.C2I[r.proxy_label])
    out=[]
    for kind, pr in (("single_view",np.asarray(raw)),("hflip_tta",np.asarray(tta))):
        met,_=tl.classification_report_dict(np.asarray(y),pr.argmax(1),pr,"")
        out.append({"arch":arch,"kind":kind,"n":len(y),
                    **{k:met[k] for k in ("f1_macro","qwk","ece","nll")}})
    del model; torch.cuda.empty_cache(); gc.collect()
    try: ck.unlink()
    except Exception: pass
    return out

TTA = pd.DataFrame(sum((tta_one(a) for a in TOP3), []))
print(TTA.round(4).to_string(index=False))
'''),
         md("## 5 — Disjoint temperature/conformal/test splits"),
         code(r'''def stratified_three_way(y, seed=0):
    rng=np.random.default_rng(seed); parts=[[],[],[]]
    for cls in sorted(np.unique(y)):
        idx=np.where(y==cls)[0]; rng.shuffle(idx)
        for i,v in enumerate(idx): parts[i%3].append(int(v))
    return [np.array(sorted(x),int) for x in parts]

def fit_temperature(probs, y):
    lp=torch.tensor(np.log(np.clip(probs,1e-8,1)),dtype=torch.float32)
    yy=torch.tensor(y,dtype=torch.long); log_t=torch.zeros(1,requires_grad=True)
    opt=torch.optim.LBFGS([log_t],lr=.2,max_iter=80,line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss=torch.nn.functional.cross_entropy(lp/log_t.exp(),yy); loss.backward(); return loss
    opt.step(closure); return float(log_t.exp().clamp(.05,20))

def apply_temperature(probs,T):
    z=np.log(np.clip(probs,1e-8,1))/T; z-=z.max(1,keepdims=True)
    e=np.exp(z); return e/e.sum(1,keepdims=True)

def conformal_sets(cal_probs, cal_y, test_probs, alpha=.10):
    scores=1-cal_probs[np.arange(len(cal_y)),cal_y]
    q=np.quantile(scores,min(1,np.ceil((len(scores)+1)*(1-alpha))/len(scores)),method="higher")
    return (1-test_probs)<=q,float(q)

rows_cal=[]; rows_conf=[]
for fold,g in P.groupby("fold"):
    avg=g.groupby("image_id")[PROB].mean(); y=g.groupby("image_id")["true"].first().loc[avg.index].astype(int).values
    probs=avg.values; temp_i,conf_i,test_i=stratified_three_way(y,seed=100+int(fold))
    T=fit_temperature(probs[temp_i],y[temp_i]); pc=apply_temperature(probs,T)
    for kind,pr in (("raw",probs[test_i]),("temperature",pc[test_i])):
        met,_=tl.classification_report_dict(y[test_i],pr.argmax(1),pr,"")
        rows_cal.append({"fold":fold,"kind":kind,"temperature":T,"n":len(test_i),
                         **{k:met[k] for k in ("f1_macro","ece","nll","brier")}})
    sets,q=conformal_sets(pc[conf_i],y[conf_i],pc[test_i])
    rows_conf.append({"fold":fold,"n_cal":len(conf_i),"n_test":len(test_i),"q":q,
                      "coverage_90":float(sets[np.arange(len(test_i)),y[test_i]].mean()),
                      "mean_set_size":float(sets.sum(1).mean()),
                      "abstain_rate":float((sets.sum(1)>1).mean())})
CAL=pd.DataFrame(rows_cal); CONF=pd.DataFrame(rows_conf)
print("temperature scaling\n",CAL.round(4).to_string(index=False))
print("\nconformal sets\n",CONF.round(4).to_string(index=False))
'''),
         md("## 6 — Save, push, and finish"),
         code(r'''out=Path(sess.stage_dir)/"tables"; out.mkdir(parents=True,exist_ok=True)
for name,df in (("ensemble_metrics.csv",M),("tta.csv",TTA),
                ("calibration.csv",CAL),("conformal.csv",CONF)):
    df.to_csv(out/name,index=False); sess.uploader.enqueue(out/name,f"tables/{name}",force=True)
sess.push_now("ensemble and uncertainty tables complete"); sess.finish()
''')]
    write_nb("NB09_Ensembles_Calibration.ipynb", c)

    # ------------------------------------------------------------------ NB10
    c = [md(r"""# NB10 — Analysis and paper figures

This notebook is read-only with respect to experiment inputs: it pulls public
HF artifacts, builds the master tables, renders Figures 1–10, reports all three
preregistered outcomes whether supported or not, then pushes only the derived
analysis files. A figure whose upstream notebook is incomplete is explicitly
marked skipped; it is never fabricated from a fallback.
"""), code(bootstrap_cell()),
         md("## 1 — Session and public artifact pull"), code(session("analysis")),
         code(r'''from huggingface_hub import snapshot_download
from pathlib import Path
import pandas as pd, numpy as np

LOCAL = Path("/kaggle/temp/tyre_analysis_pull")
snapshot_download(tl.HF_REPO_DEFAULT, repo_type="dataset", token=None,
                  local_dir=str(LOCAL), allow_patterns=[
                    "runs/*/metrics/final.csv", "runs/*/metrics/epochs.csv",
                    "tables/*.csv", "analysis/hypotheses.json",
                    "analysis/xai_examples/*.png"])

frames=[]
for f in sorted(LOCAL.glob("runs/*/metrics/final.csv")):
    try: frames.append(pd.read_csv(f))
    except Exception as e: print("skip",f,e)
R=pd.concat(frames,ignore_index=True,sort=False) if frames else pd.DataFrame()
assert len(R), "no final.csv files were pulled"

# Two historic VGG jobs reached epoch 60 and wrote every scientific artifact,
# then the old telemetry observer raised while serialising. Preserve that fact
# without pretending their operational status field says completed.
R["scientific_complete"]=(R.status.eq("completed") |
    (R.epochs_trained.fillna(-1)>=R.epochs_planned.fillna(10**9)))
INVALID_STAGE_A_ARCHS = {a for a, spec in tl.ZOO.items()
                         if spec.get("stage_a_valid") is False}
stage_a = R[(R.stage.eq("a")) & (R.technique.eq("base")) &
            R.scientific_complete].copy()
Q = stage_a[stage_a.arch.isin(INVALID_STAGE_A_ARCHS)].copy()
A = stage_a[~stage_a.arch.isin(INVALID_STAGE_A_ARCHS)].copy()
print(f"public final rows: {len(R)}; Stage A valid: {len(A)}/153; "
      f"quarantined: {len(Q)}")
if len(Q): print("quarantined architecture labels:", sorted(Q.arch.unique()))
print(R.status.value_counts().to_string())
'''),
         md("## 2 — Master Stage-A table"),
         code(r'''OUT=Path(sess.stage_dir)/"analysis"; TAB=Path(sess.stage_dir)/"tables"
OUT.mkdir(parents=True,exist_ok=True); TAB.mkdir(parents=True,exist_ok=True)
T=(A.groupby("arch").agg(n=("run_id","size"),f1_mean=("best_val_f1_macro","mean"),
     f1_min=("best_val_f1_macro","min"),f1_max=("best_val_f1_macro","max"),
     qwk_mean=("best_val_qwk","mean"),final_f1=("final_val_f1_macro","mean"),
     best_epoch=("best_epoch","mean"),energy_wh=("total_energy_wh","mean"),
     wall_h=("total_wall_seconds",lambda x:x.mean()/3600))
   .assign(spread=lambda x:x.f1_max-x.f1_min).sort_values("f1_mean",ascending=False))
T.to_csv(TAB/"master_architectures.csv")
if len(Q): Q.to_csv(TAB/"stage_a_quarantined.csv", index=False)
print(T.round(4).to_string())
'''),
         md("## 3 — Figures 1 and 2: evidence/accuracy and fold instability"),
         code(r'''import matplotlib.pyplot as plt

def read_table(name):
    p=LOCAL/"tables"/name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

EV=read_table("xai_evidence_all.csv"); SEL=read_table("stage_b_selection.csv")
if len(EV):
    xe=EV.groupby("arch").ter_norm.mean(); xa=A.groupby("arch").best_val_f1_macro.mean()
    j=pd.DataFrame({"ter_norm":xe,"macro_f1":xa}).dropna()
    fig,ax=plt.subplots(figsize=(8,5.5)); ax.scatter(j.ter_norm,j.macro_f1,s=65)
    for k,r in j.iterrows(): ax.annotate(k,(r.ter_norm,r.macro_f1),fontsize=8,xytext=(4,3),textcoords="offset points")
    ax.axvline(1,ls=":",c="grey"); ax.axhline(tl.FLOOR,ls="--",c="crimson")
    ax.set(xlabel="TER_norm (area-normalised)",ylabel="mean best macro-F1",
           title="Figure 1 — accuracy versus evidence quality")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT/"fig01_accuracy_vs_ter.png",dpi=170); plt.show()
else: print("Figure 1 skipped: run NB07")

piv=A.pivot_table(index="arch",columns="fold",values="best_val_f1_macro").sort_index()
fig,ax=plt.subplots(figsize=(12,6)); x=np.arange(len(piv)); w=.25
for i,f in enumerate(piv.columns): ax.bar(x+(i-1)*w,piv[f],w,label=f"fold {f}")
for k,c in (("frame_occupancy","crimson"),("colour_probe","darkorange"),
            ("structure_probe","seagreen"),("annotation_sidechannel","purple")):
    ax.axhline(tl.BASELINES[k]["mean"],ls="--",lw=1,c=c,label=k)
ax.set_xticks(x); ax.set_xticklabels(piv.index,rotation=45,ha="right")
ax.set(ylabel="macro-F1",title="Figure 2 — per-fold Stage-A results and trivial baselines")
ax.legend(fontsize=7,ncol=2); ax.grid(axis="y",alpha=.25); fig.tight_layout()
fig.savefig(OUT/"fig02_per_fold.png",dpi=170); plt.show()
'''),
         md("## 4 — Figures 3 and 4: saliency examples and causal interventions"),
         code(r'''from PIL import Image
imgs=sorted((LOCAL/"analysis"/"xai_examples").glob("*.png"))
if imgs:
    n=len(imgs); cols=4; rows=int(np.ceil(n/cols)); fig,axes=plt.subplots(rows,cols,figsize=(14,3.5*rows))
    axes=np.asarray(axes).reshape(-1)
    for ax,p in zip(axes,imgs): ax.imshow(Image.open(p)); ax.set_title(p.stem); ax.axis("off")
    for ax in axes[len(imgs):]: ax.axis("off")
    fig.suptitle("Figure 3 — same preregistered evidence procedure across architectures")
    fig.tight_layout(); fig.savefig(OUT/"fig03_saliency_panels.png",dpi=160); plt.show()
else: print("Figure 3 skipped: NB07 examples missing")

ST=read_table("stress_tests.csv")
if len(ST):
    p=ST.pivot_table(index="arch",columns="intervention",values="f1_macro")
    d=p.sub(p["none"],axis=0).drop(columns="none")
    fig,ax=plt.subplots(figsize=(9,max(3,.7*len(d)))); im=ax.imshow(d.values,aspect="auto",cmap="RdBu",vmin=-.5,vmax=.5)
    ax.set_xticks(range(len(d.columns))); ax.set_xticklabels(d.columns,rotation=35,ha="right")
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index); plt.colorbar(im,ax=ax,label="Δ macro-F1 vs original")
    ax.set_title("Figure 4 — causal stress-test matrix"); fig.tight_layout()
    fig.savefig(OUT/"fig04_stress_matrix.png",dpi=170); plt.show()
else: print("Figure 4 skipped: run NB08")
'''),
         md("## 5 — Figure 5 and preregistered H1–H3 outcomes"),
         code(r'''outcomes=[]
if len(EV):
    ter=EV.groupby("arch").ter_norm.mean(); per=A.groupby("arch").best_val_f1_macro
    J=pd.DataFrame({"ter_norm":ter,"accuracy":per.mean(),
                    "stability":-(per.max()-per.min())}).dropna()
    rter=float(J.ter_norm.corr(J.stability)); racc=float(J.accuracy.corr(J.stability))
    outcomes.append({"hypothesis":"H1","n":len(J),"stat_primary":rter,"stat_reference":racc,
                     "supported":abs(rter)>abs(racc),
                     "reading":"corr(TER, stability) vs corr(accuracy, stability)"})
    fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(J.ter_norm,J.stability,s=65)
    for k,r in J.iterrows(): ax.annotate(k,(r.ter_norm,r.stability),fontsize=8)
    ax.set(xlabel="TER_norm",ylabel="negative cross-fold F1 spread (higher is stable)",
           title=f"Figure 5 — H1: r={rter:+.2f}; accuracy reference r={racc:+.2f}")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT/"fig05_h1_stability.png",dpi=170); plt.show()

    fine={"bilinear_cnn","hbp","csab"}
    n_fine=int(pd.Index(ter.index).isin(fine).sum())
    outcomes.append({"hypothesis":"H3","n":n_fine,"stat_primary":np.nan,"stat_reference":np.nan,
                     "supported":None,
                     "reading":"NOT TESTABLE YET: Stage A contains no preregistered fine-grained architectures"})

if len(EV) and len(ST):
    e=EV.groupby("arch").agg(sar=("sar","mean"),dmgar=("dmgar","mean"))
    s=ST.pivot_table(index="arch",columns="intervention",values=["recall_low","recall_high"])
    mark=s[("recall_low","mask_marking")]-s[("recall_low","none")]
    dmg=s[("recall_high","mask_damage")]-s[("recall_high","none")]
    h2=e.join(pd.DataFrame({"mark_dependence":-mark,"damage_dependence":-dmg})).dropna()
    rs=float(h2.sar.corr(h2.mark_dependence)); rd=float(h2.dmgar.corr(h2.damage_dependence))
    outcomes.append({"hypothesis":"H2","n":len(h2),"stat_primary":rs,"stat_reference":rd,
                     "supported":bool(rs>0 and rd>0),
                     "reading":"corr(SAR, marking dependence); reference=corr(DmgAR, damage dependence)"})

H=pd.DataFrame(outcomes); H.to_csv(TAB/"hypothesis_outcomes.csv",index=False)
print(H.round(4).to_string(index=False) if len(H) else "H1-H3 unavailable until NB07/NB08 complete")
'''),
         md("## 6 — Figures 6 and 7: OFAT effects and attribution faithfulness"),
         code(r'''EFF=read_table("stage_b_effects.csv")
if len(EFF):
    g=EFF.groupby("factor").delta_vs_stage_a.agg(["mean","std","count"]).sort_values("mean")
    g["ci95"]=1.96*g["std"]/np.sqrt(g["count"].clip(lower=1))
    fig,ax=plt.subplots(figsize=(9,max(4,.42*len(g)))); ax.barh(g.index,g["mean"],xerr=g.ci95)
    ax.axvline(0,c="black",lw=1); ax.set(xlabel="paired Δ macro-F1 vs Stage A",
        title="Figure 6 — Stage-B one-factor effects (95% normal CIs)")
    ax.grid(axis="x",alpha=.25); fig.tight_layout(); fig.savefig(OUT/"fig06_ofat_effects.png",dpi=170); plt.show()
else: print("Figure 6 skipped: Stage B effects not complete")

FA=read_table("xai_faithfulness.csv")
if len(FA):
    FA["faithfulness"]=FA.insertion_auc-FA.deletion_auc
    q=FA.pivot_table(index="arch",columns="method",values="faithfulness")
    ax=q.plot.bar(figsize=(12,5)); ax.axhline(0,c="black",lw=1)
    ax.set(ylabel="insertion AUC − deletion AUC",title="Figure 7 — attribution faithfulness")
    ax.grid(axis="y",alpha=.25); plt.tight_layout(); plt.savefig(OUT/"fig07_faithfulness.png",dpi=170); plt.show()
else: print("Figure 7 skipped: run NB07")
'''),
         md("## 7 — Figures 8 and 9: convergence and energy"),
         code(r'''for num,col,title,unit,name in [
    (8,"best_epoch","Figure 8 — mean best epoch by architecture","epoch","fig08_best_epoch.png"),
    (9,"total_energy_wh","Figure 9 — mean energy per Stage-A run","Wh","fig09_energy.png")]:
    g=A.groupby("arch")[col].mean().sort_values()
    fig,ax=plt.subplots(figsize=(8,max(4,.35*len(g)))); ax.barh(g.index,g.values)
    ax.set(xlabel=unit,title=title); ax.grid(axis="x",alpha=.25); fig.tight_layout()
    fig.savefig(OUT/name,dpi=170); plt.show()
'''),
         md("## 8 — Figure 10: validation-session heat map"),
         code(r'''rows=[]
for f in sorted(LOCAL.glob("runs/a-*/metrics/epochs.csv")):
    try:
        e=pd.read_csv(f)
        if not len(e): continue
        best=e.loc[e.val_qwk.idxmax()]
        arch=str(best.get("arch",f.parts[-3].split("-")[1]))
        for col in e.columns:
            if col.startswith("val_acc_session_"):
                rows.append({"arch":arch,"session":col.replace("val_acc_session_",""),"acc":best[col]})
    except Exception: pass
if rows:
    Q=pd.DataFrame(rows).pivot_table(index="arch",columns="session",values="acc")
    fig,ax=plt.subplots(figsize=(13,max(4,.45*len(Q)))); im=ax.imshow(Q.values,aspect="auto",vmin=0,vmax=1,cmap="RdYlGn")
    ax.set_xticks(range(len(Q.columns))); ax.set_xticklabels([x[:20] for x in Q.columns],rotation=60,ha="right",fontsize=7)
    ax.set_yticks(range(len(Q))); ax.set_yticklabels(Q.index); plt.colorbar(im,ax=ax,label="accuracy")
    ax.set_title("Figure 10 — best-epoch accuracy per validation session")
    fig.tight_layout(); fig.savefig(OUT/"fig10_per_session.png",dpi=170); plt.show()
else: print("Figure 10 skipped: per-session epoch columns unavailable")
'''),
         md("## 9 — Push derived tables and figures"),
         code(r'''sess.uploader.enqueue_dir(OUT,"analysis",force=True)
sess.uploader.enqueue_dir(TAB,"tables",force=True)
sess.push_now("NB10 analysis complete"); sess.finish()
print("analysis complete")
''')]
    write_nb("NB10_Analysis_Figures.ipynb", c)


if __name__ == "__main__":
    import build_notebooks as _base
    build_later(vars(_base))
