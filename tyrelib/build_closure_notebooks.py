"""Targeted recovery notebooks. Never overwrite executed NB00--NB10."""
import ast
import copy
import json
from pathlib import Path

import build_notebooks as b
from build_later_notebooks import build_later

REV = "closure_2026-09-09"
md, code = b.md, b.code

PULL = r'''from pathlib import Path
import json, numpy as np, pandas as pd
import re, time
from email.utils import parsedate_to_datetime
from huggingface_hub import HfApi, hf_hub_download
REPO = tl.HF_REPO_DEFAULT
READ_TOKEN = getattr(sess.uploader, "token", None)
if not READ_TOKEN:
    raise RuntimeError("HF_TOKEN is required: enable the Kaggle secret and rerun the Session cell.")

def hf_read(label, operation, attempts=8):
    # API metadata calls need retries too; uploads retain their existing schedule.
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            if status not in (429, 500, 502, 503, 504):
                raise
            if attempt == attempts - 1:
                raise RuntimeError(f"HF read {label} still unavailable after {attempts} attempts; rerun this cell later.") from None
            headers = {k.lower(): v for k, v in getattr(response, "headers", {}).items()}
            delays = [min(300, 5 * 2**attempt)]
            hint = headers.get("retry-after", "")
            if hint:
                try:
                    delays.append(float(hint))
                except ValueError:
                    try:
                        delays.append(parsedate_to_datetime(hint).timestamp() - time.time())
                    except (ValueError, TypeError, OverflowError):
                        pass
            for match in re.finditer(r'\bt\s*=\s*(\d+)', headers.get("ratelimit", "")):
                delays.append(float(match.group(1)))
            body_delay = tl.parse_retry_after(str(exc))
            if body_delay is not None:
                delays.append(body_delay)
            remaining = max(delays) + 2
            print(f"[HF read] {label}: HTTP {status}; waiting {remaining:.0f}s, retry {attempt+1}/{attempts-1}.", flush=True)
            # Short sleeps allow Kaggle Stop to interrupt the wait normally.
            while remaining > 0:
                step = min(10, remaining)
                time.sleep(step)
                remaining -= step

REVISION = hf_read("pin input revision", lambda: HfApi().repo_info(
    REPO, repo_type="dataset", token=READ_TOKEN)).sha
PULL_ROOT = Path(sess.stage_dir)/"closure_pull"
def pull(rel):
    return Path(hf_read(rel, lambda: hf_hub_download(
        REPO, rel, repo_type="dataset", revision=REVISION,
        token=READ_TOKEN, local_dir=str(PULL_ROOT))))
def public_csv(rel):
    return pd.read_csv(pull(rel))
TAB = Path(sess.stage_dir)/"tables"/"closure_2026-09-09"
TAB.mkdir(parents=True, exist_ok=True)
print("Pinned HF input revision:", REVISION)
'''

def start(title, stage, data=False):
    cells = [md(title), code(b.bootstrap_cell()),
             md("## Session — Internet ON, HF_TOKEN secret; one notebook by default"),
             code(b.SESSION_CELL.replace("stage='a'", f"stage='{stage}'"))]
    if data:
        cells.append(code(b.DATA_CELL))
    return cells

def save(name, cells):
    for i, c in enumerate(cells):
        c["id"] = f"closure-{i:03d}"
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]), filename=f"{name}:cell{i}")
    b.write_nb(name, cells)

def baseline_report():
    captured = []
    writer = b.write_nb
    try:
        b.write_nb = lambda name, cells: captured.extend(cells)
        b.nb01()
    finally:
        b.write_nb = writer
    features = next("".join(c["source"]) for c in captured
                    if c["cell_type"] == "code" and "def colour_feats" in "".join(c["source"]))
    cells = start("""# NB01A — Baseline reporting recovery (no neural training)

Run once, not on four workers. Attach Tire Dataset Prepared; CPU is sufficient.
All five legacy baseline rows already exist on HF. This notebook preserves them,
reproduces colour/structure/HOG and a **training-selected** majority baseline,
and reads the legacy random-init **final epoch**, not its validation-selected best.
HOG is full-frame, not the published segmented-tread method: do not claim exact replication.
The matched ResNet-50 training comparison is in NB01B, not hidden in this cell.
Only revisioned derived tables are uploaded; no old table/run is overwritten.
**Read fix r2:** HF_TOKEN-authenticated reads, bounded rate-limit retries, no repository inventory.
If HF asks for a wait, leave the cell running; Stop still interrupts it.
""", "s1report", True)
    cells += [code(PULL), code('''legacy = public_csv("tables/baselines.csv")
print("Already published (legacy):")
print(legacy.to_string(index=False))
legacy.to_csv(TAB/"baselines_legacy.csv", index=False)
'''), code(features), code('''from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from skimage.feature import hog
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("error", category=ConvergenceWarning)
COLOUR = ['R','G','B','bright','contrast','sat','sat_p99','blueish','bright_frac','dark_frac']
STRUCT = ['d10','d15','d20','gmean','gp95','lapvar','colstd','colmin','rowstd']
# Reproduce the legacy logistic solver, not a retuned sklearn classifier.
def fit_probe(X, y):
    W=np.zeros((X.shape[1],3)); bias=np.zeros(3); Y=np.eye(3)[y]
    for _ in range(4000):
        z=X@W+bias; z-=z.max(1,keepdims=True); p=np.exp(z); p/=p.sum(1,keepdims=True)
        G=(p-Y)/len(X); W-=.5*(X.T@G+1e-2*W); bias-=.5*G.sum(0)
    return W,bias
H=[]
for r in tqdm(clean.itertuples(), total=len(clean), desc="HOG"):
    with Image.open(DATA_ROOT/r.relative_path) as im:
        gray=np.asarray(im.convert("L").resize((128,128)))
    H.append(hog(gray,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2)))
H=np.asarray(H); rows=[]; prediction_rows=[]
for fold in (0,1,2):
    tr,va=tl.load_split(DATA_ROOT,fold)
    tr_ids=set(tr.loc[tr.image_kind.eq("clean_original"),"image_id"])
    va_ids=set(va.image_id)
    train=F.image_id.isin(tr_ids).to_numpy(); valid=F.image_id.isin(va_ids).to_numpy()
    assert not (train & valid).any() and train.sum()+valid.sum()==418
    assert set(F.loc[train,"sess"]).isdisjoint(F.loc[valid,"sess"])
    y=F.y.to_numpy(); predictions={}
    for label,cols in (("colour_probe",COLOUR),("structure_probe",STRUCT)):
        X=F[cols].to_numpy(); mu=X[train].mean(0); sd=X[train].std(0)+1e-8
        W,bias=fit_probe((X[train]-mu)/sd,y[train])
        predictions[label]=(((X[valid]-mu)/sd)@W+bias).argmax(1)
    # C and feature geometry are unchanged from legacy NB01. No validation tuning.
    svm=LinearSVC(C=.01,max_iter=3000,random_state=0).fit(H[train],y[train])
    predictions["hog_svm"]=svm.predict(H[valid])
    majority=int(np.bincount(y[train],minlength=3).argmax())
    predictions["majority_train_selected"]=np.full(valid.sum(),majority)
    for label,pred in predictions.items():
        metrics,cm=tl.classification_report_dict(y[valid],pred,None,"")
        rows.append(dict(baseline=label,fold=fold,**metrics,source_revision=REVISION))
        for r,p in zip(F.loc[valid].itertuples(),pred):
            prediction_rows.append(dict(baseline=label,fold=fold,image_id=r.image_id,
                                        session_group=r.sess,true=int(r.y),pred=int(p)))
    rid=f"s1-resnet18-randinit-f{fold}-s1"
    status=json.loads(pull(f"runs/{rid}/STATUS.json").read_text())
    history=public_csv(f"runs/{rid}/metrics/epochs.csv")
    assert status["status"]=="completed" and list(history.epoch)==list(range(1,16))
    last=history.iloc[-1]
    rows.append(dict(baseline="legacy_resnet18_random_15ep_seed1",fold=fold,
                     f1_macro=last.val_f1_macro,acc=last.val_acc,qwk=last.val_qwk,
                     source_revision=REVISION))
result=pd.DataFrame(rows)
assert len(result)==15 and np.isfinite(result.f1_macro).all()
result.to_csv(TAB/"baselines_by_fold.csv",index=False)
pd.DataFrame(prediction_rows).to_csv(TAB/"baseline_predictions.csv",index=False)
F.to_csv(TAB/"baseline_features.csv",index=False)
summary=result.groupby("baseline").f1_macro.agg(["mean","min","max","count"])
summary.to_csv(TAB/"baseline_summary.csv")
print(summary.to_string())
print("These are mileage-proxy pilot results; folds 0/2 remain leak-flagged.")
sess.uploader.enqueue_dir(TAB,"tables/closure_2026-09-09",force=True)
sess.push_now("baseline report recovery complete"); sess.finish()
''')]
    save("NB01A_Baseline_Recovery.ipynb", cells)

def matched_random():
    cells=start("""# NB01B — Matched random-initialised ResNet-50

**New planned comparison, not a rerun of the completed legacy ResNet-18.**
Nine jobs: 3 folds × 3 seeds, same ResNet-50/384px/CORAL/60 epochs as Stage A,
changing only pretraining to False. New `s1-resnet50-randmatched_r1-*` IDs.
The old three 15-epoch ResNet-18 jobs stay intact. No smaller substitute.

Attach Tire Dataset Prepared, Internet ON, HF_TOKEN, **T4 ×2**. One session by
default; four copies may use distinct ACCOUNT labels in the shared active tuple.
All other training notebooks must be stopped before changing worker ownership.
Each model runs in a fresh process. Re-run after a timeout: completed jobs skip,
partial jobs fetch optimizer/scaler/RNG/checkpoints from HF. Resume is from the
last **completed epoch**, not the interrupted batch. A hard OS kill cannot run
an emergency handler; only already-published progress survives that case.
Normal HF commits batch every 30 minutes; completion/clean Stop flush immediately.
Budget is nine full runs; use the printed measured plan, not legacy NB01's minutes estimate.
""", "s1",True)
    cells += [code('''import torch
assert torch.cuda.is_available(), "Select Kaggle GPU T4 x2 before training"
tl.assert_zoo_ok(["resnet50"])
cfgs=sess.configs(["resnet50"],folds=(0,1,2),seeds=(1,2,3),
                  technique="randmatched_r1",pretrained=False)
assert len(cfgs)==9 and all(c["max_epochs"]==60 and c["input_resolution"]==384 for c in cfgs)
run_ids=[c["run_id"] for c in cfgs]
print(sess.reconcile(run_ids).to_string(index=False))
sess.run_all(cfgs,title="Matched random-init ResNet-50",isolate_runs=True,
             steal_stale=False,takeover_when_idle=True)
'''), code('''# Only one analysis writer publishes the aggregate; rerun NB01A separately for the CPU baselines.
from pathlib import Path
df=sess.aggregate_remote(run_ids)
if len(df):
    out=Path(sess.stage_dir)/"tables"/"closure_2026-09-09"; out.mkdir(parents=True,exist_ok=True)
    # Account-specific partial summaries cannot overwrite each other.
    dest=out/f"matched_random_{ACCOUNT}.csv"; df.to_csv(dest,index=False)
    sess.uploader.enqueue(dest,f"tables/closure_2026-09-09/{dest.name}",force=True)
    print(df[["run_id","epochs_trained","final_val_f1_macro"]].to_string(index=False))
sess.finish()
sess.confirm_on_hf(run_ids)
''')]
    save("NB01B_Matched_RandomInit.ipynb",cells)

def architecture_audit():
    cells=start("""# NB03A — Stage-A coverage and quarantine audit (no training)

The original ConvNeXt-V2-S pretrained identifier is unavailable. Rerunning it
cannot repair nine ResNet-18 checkpoints. This notebook verifies all 153 valid
results and the nine excluded records, publishes a revisioned coverage table,
and shows what timm actually offers. It does **not** silently replace the model,
train from scratch under a pretrained label, overwrite old runs, or reselect NB06.
CPU sufficient, one session, Internet ON and HF_TOKEN. A replacement training
arm requires an explicit model/pretraining choice and fresh run IDs.
""", "aaudit")
    cells += [code(PULL + '''FILES = set(hf_read("architecture inventory", lambda: HfApi().list_repo_files(
    REPO, repo_type="dataset", revision=REVISION, token=READ_TOKEN)))
'''),code('''import timm
valid=[a for a,spec in tl.ZOO.items() if a!="resnet18" and spec.get("stage_a_valid") is not False]
rows=[]
for arch in valid+["convnextv2_s"]:
    for fold in (0,1,2):
        for seed in (1,2,3):
            rid=f"a-{arch}-base-f{fold}-s{seed}"
            status=json.loads(pull(f"runs/{rid}/STATUS.json").read_text())
            final=public_csv(f"runs/{rid}/metrics/final.csv").iloc[0]
            complete=int(final.epochs_trained)==60 and int(final.epochs_planned)==60
            artifacts=all(f"runs/{rid}/{p}" in FILES for p in
                          ("checkpoints/ckpt_best.pt","checkpoints/ckpt_last.pt","metrics/epochs.csv"))
            rows.append(dict(run_id=rid,arch=arch,fold=fold,seed=seed,
                             operational_status=status["status"],budget_complete=complete,
                             artifacts_present=artifacts,n_params=status.get("n_params_total"),
                             quarantined=arch=="convnextv2_s",source_revision=REVISION))
audit=pd.DataFrame(rows)
assert len(audit)==162 and audit.budget_complete.all() and audit.artifacts_present.all()
assert (~audit.quarantined).sum()==153 and audit.quarantined.sum()==9
assert (audit.loc[audit.quarantined,"n_params"]==11177538).all()
offers=timm.list_models("convnextv2_small*",pretrained=True)
print("Installed timm:",timm.__version__,"pretrained Small offerings:",offers)
print("153 valid executions; 9 quarantined. Full planned Tier 5/6 scope is not complete.")
print(audit.groupby(["arch","quarantined"]).size().to_string())
audit.to_csv(TAB/"stage_a_coverage.csv",index=False)
(TAB/"small_availability.json").write_text(json.dumps(dict(timm=timm.__version__,
    pretrained_small=offers,decision="No substitution authorized",source_revision=REVISION),indent=2))
sess.uploader.enqueue_dir(TAB,"tables/closure_2026-09-09",force=True)
sess.push_now("Stage-A coverage audit complete"); sess.finish()
''')]
    save("NB03A_Architecture_Audit.ipynb",cells)

FIGURE3 = '''# Same deterministic three images for every architecture with a surviving r3 method.
# Publish saliency + mask contours only; no source RGB or vehicle detail is uploaded.
import gc, torch
from PIL import Image
from huggingface_hub import hf_hub_download
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","-q","grad-cam"],check=True)
DATA_ROOT=sess.prepare_data(); ANN_ROOT=tl.find_annotations_root(DATA_ROOT)
assert ANN_ROOT is not None, "Attach the prepared dataset with clean annotation masks"
_,va=tl.load_split(DATA_ROOT,1)
examples=va.sort_values("image_id").groupby("proxy_label",group_keys=False).head(1)
assert len(examples)==3
FA=read_table("xai_faithfulness.csv")
chosen=FA[FA.selected.astype(str).str.lower().eq("true")].drop_duplicates("arch").sort_values("arch")
chosen=chosen[chosen.arch.isin(A.arch.unique())]
assert len(chosen), "No faithful methods available; do not fabricate a panel"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
fig,axes=plt.subplots(len(chosen),3,figsize=(12,3*len(chosen)),squeeze=False)
panel_records=[]
for row_index,method_row in enumerate(chosen.itertuples()):
    rid=f"a-{method_row.arch}-base-f1-s1"
    print(f"[PANEL] {row_index+1}/{len(chosen)} {rid} ({method_row.method})",flush=True)
    ck=Path(hf_hub_download(tl.HF_REPO_DEFAULT,f"runs/{rid}/checkpoints/ckpt_best.pt",
        repo_type="dataset",revision=SOURCE_REVISION,token=False,
        local_dir=str(LOCAL/"panel_weights")))
    state=torch.load(ck,map_location="cpu",weights_only=False); cfg=state["config"]
    actual=tl.infer_checkpoint_architecture(state["model"])
    assert cfg["arch"]==method_row.arch and actual in ("unknown",method_row.arch)
    model=tl.build_model(cfg["arch"],3,pretrained=False,head=cfg["head_type"],img_size=cfg["input_resolution"])
    model.load_state_dict(state["model"],strict=True); del state
    model=model.to(device).eval(); tf=tl.build_transforms(cfg["input_resolution"],False,cfg.get("preprocessing","raw"))
    cam,tag=tl.make_cam(model,cfg["arch"],method_row.method)
    assert cam is not None, f"Unavailable selected method: {tag}"
    try:
        for col,r in enumerate(examples.itertuples()):
            with Image.open(DATA_ROOT/r.relative_path) as im: x=tf(im.convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                logits=model(x)
                prob=tl.CoralHead.probs(logits) if cfg["head_type"]=="coral" else logits.softmax(1)
                pred=int(prob.argmax(1).item())
            sal=cam(input_tensor=x,targets=[tl.ClassProbabilityTarget(pred,cfg["head_type"])])[0]
            assert np.isfinite(sal).all()
            with Image.open(tl.mask_path(ANN_ROOT,r.image_id)) as mask:
                region=np.asarray(mask.resize((sal.shape[1],sal.shape[0]),Image.Resampling.NEAREST))>0
            ax=axes[row_index,col]; ax.imshow(sal,cmap="inferno",vmin=0,vmax=1)
            if region.any() and not region.all(): ax.contour(region,levels=[.5],colors=["cyan"],linewidths=.7)
            empty=float(sal.max())<=1e-12
            ax.set_title(f"{method_row.arch}: {tag}\\n{r.proxy_label}; pred={pred}"+("; ZERO MAP" if empty else ""),fontsize=8)
            ax.axis("off")
            panel_records.append(dict(run_id=rid,image_id=r.image_id,method=method_row.method,
                target_layer=tag,pred=pred,zero_map=empty,source_revision=SOURCE_REVISION))
            del x,logits,prob,sal
    finally:
        cam.activations_and_grads.release(); del cam,model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # Remove only this explicitly downloaded scratch checkpoint, never HF.
        ck.unlink(missing_ok=True)
fig.suptitle("Figure 3 — faithful-method survivors, same three fold-1 images; cyan = tyre boundary",fontsize=12)
fig.tight_layout(); fig.savefig(OUT/"fig03_saliency_panels.png",dpi=130); plt.show(); plt.close(fig)
pd.DataFrame(panel_records).to_csv(TAB/"saliency_panel_manifest.csv",index=False)
# Record intervention availability: an empty region is not an applied stress test.
coverage=[]
clean=tl.read_manifest(DATA_ROOT/"manifests/clean_manifest.csv")
for r in clean.itertuples():
    with Image.open(tl.mask_path(ANN_ROOT,r.image_id)) as im: mask=np.asarray(im)
    coverage.append(dict(image_id=r.image_id,fold=r.fold_id,marking_pixels=int((mask==3).sum()),
                         damage_pixels=int((mask==4).sum())))
coverage=pd.DataFrame(coverage); coverage.to_csv(TAB/"intervention_region_coverage.csv",index=False)
print("Images with marking/damage pixels by fold:")
print(coverage.assign(marking=coverage.marking_pixels.gt(0),damage=coverage.damage_pixels.gt(0)).groupby("fold")[["marking","damage"]].sum())
sess.uploader.enqueue(OUT/"fig03_saliency_panels.png","analysis/closure_2026-09-09/fig03_saliency_panels.png",force=True)
sess.uploader.enqueue_dir(TAB,"tables/closure_2026-09-09",force=True)
sess.push_now("saliency panel and region audit complete")
'''

def analysis_repair():
    collected={}
    ctx=vars(b).copy(); ctx["write_nb"]=lambda n,c: collected.update({n:copy.deepcopy(c)})
    build_later(ctx)
    cells=collected["NB10_Analysis_Figures.ipynb"]
    cells[0]=md("""# NB10R — Analysis recovery, no retraining

Use **one Kaggle T4 session**, prepared dataset, Internet and HF_TOKEN. This
reuses the public checkpoints and metrics; it does not run NB06 or NB08 again.
Inputs are revision-pinned. Outputs go to `analysis/closure_2026-09-09/` and
`tables/closure_2026-09-09/`; the original figures, hypotheses and run files stay intact.

Repairs: generates Figure 3 (same fixed images, saliency + mask contours only),
reports undefined H2 as inconclusive, excludes quarantine from Figure 10, uses
final-epoch scores for descriptive performance figures. H1's original selected-
epoch calculation is preserved and explicitly labelled; it is not re-registered.
The panel includes only faithful-method survivors, not excluded methods presented
as valid explanations. Zero maps are labelled. It does not complete H3 or S5/S9.
Figures 8–10 remain implementation-specific convergence/energy/session figures,
not the original plan's mask-audit/video/resolution-interaction experiments.
""")
    for cell in cells:
        s="".join(cell["source"])
        if cell["cell_type"]!="code": continue
        if "snapshot_download(tl.HF_REPO_DEFAULT" in s:
            s=s.replace('from huggingface_hub import snapshot_download','from huggingface_hub import snapshot_download, HfApi')
            s=s.replace('LOCAL = Path("/kaggle/temp/tyre_analysis_pull")','SOURCE_REVISION=HfApi().repo_info(tl.HF_REPO_DEFAULT,repo_type="dataset",token=False).sha\nprint("Pinned input revision:",SOURCE_REVISION)\nLOCAL = Path("/kaggle/temp/tyre_analysis_pull")/SOURCE_REVISION')
            s=s.replace('repo_type="dataset", token=None,','repo_type="dataset", token=False, revision=SOURCE_REVISION,')
            s+='\nassert len(A)==153 and A.run_id.nunique()==153 and len(Q)==9, "Stage-A coverage changed: inspect before reporting"\n'
        if 'OUT=Path(sess.stage_dir)/"analysis"' in s:
            s=s.replace('OUT=Path(sess.stage_dir)/"analysis"; TAB=Path(sess.stage_dir)/"tables"',
                        'OUT=Path(sess.stage_dir)/"analysis"/"closure_2026-09-09"; TAB=Path(sess.stage_dir)/"tables"/"closure_2026-09-09"')
        if 'xe=EV.groupby' in s:
            s=s.replace('best_val_f1_macro','final_val_f1_macro').replace('mean best macro-F1','mean final-epoch macro-F1')
            s=s.replace('Figure 2 — per-fold Stage-A results','Figure 2 — final-epoch per-fold Stage-A results')
        if 'imgs=sorted(' in s:
            s=FIGURE3+'\n'+s[s.index('ST=read_table'):]
        if 'outcomes=[]' in s:
            s=s.replace('"supported":bool(rs>0 and rd>0),',
                '"supported":bool(rs>0 and rd>0) if np.isfinite([rs,rd]).all() else None,\n                     "outcome":"supported" if np.isfinite([rs,rd]).all() and rs>0 and rd>0 else ("unsupported" if np.isfinite([rs,rd]).all() else "inconclusive_undefined"),')
            s=s.replace('"corr(TER, stability) vs corr(accuracy, stability)"','"Legacy selected-epoch comparison; not a fixed-budget performance claim"')
            s=s.replace('title=f"Figure 5 — H1:', 'title=f"Figure 5 — legacy selected-epoch H1:')
        if 'for f in sorted(LOCAL.glob("runs/a-*/metrics/epochs.csv"))' in s:
            if 'if f.parts[-3] not in set(A.run_id)' not in s:
                s=s.replace('    try:\n        e=pd.read_csv(f)','    if f.parts[-3] not in set(A.run_id): continue\n    try:\n        e=pd.read_csv(f)')
            s=s.replace('best=e.loc[e.val_qwk.idxmax()]','best=e.sort_values("epoch").iloc[-1]')
            s=s.replace('except Exception: pass','except Exception as exc: raise RuntimeError(f"Unreadable eligible history {f}") from exc')
            s=s.replace('best-epoch accuracy','final-epoch accuracy')
        if 'sess.uploader.enqueue_dir(OUT' in s:
            s=s.replace('enqueue_dir(OUT,"analysis"','enqueue_dir(OUT,"analysis/closure_2026-09-09"')
            s=s.replace('enqueue_dir(TAB,"tables"','enqueue_dir(TAB,"tables/closure_2026-09-09"')
            s=s.replace('print("analysis complete")','print("Recovery outputs published. Inspect the figure manifest; broader study still incomplete.")')
            s='''import json
figures=sorted(p.name for p in OUT.glob("fig*.png"))
(TAB/"analysis_manifest.json").write_text(json.dumps(dict(source_revision=SOURCE_REVISION,
    figures=figures,figure_count=len(figures),expected=10,
    remaining=["H3 fine-grained models", "S5 dense tasks", "S9 integration", "original-plan video/interaction figures"]),indent=2))
print("Generated figures:", len(figures), "/10", figures)
assert len(figures)==10, "Missing figure: do not call the reporting package complete"
'''+s
        cell["source"]=s.splitlines(True)
    save("NB10R_Analysis_Recovery.ipynb",cells)

if __name__=="__main__":
    baseline_report(); matched_random(); architecture_audit(); analysis_repair()
