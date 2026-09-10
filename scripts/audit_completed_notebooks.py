"""Read-only, revision-pinned HF audit of NB08-NB10 artifacts."""
import concurrent.futures
import io
import json
import numpy as np
import pandas as pd
import requests
from huggingface_hub import HfApi, hf_hub_url

repo = "Shanmuk4622/tyre-wear-study"
api = HfApi()
revision = api.repo_info(repo, repo_type="dataset").sha
files = set(api.list_repo_files(repo, repo_type="dataset", revision=revision))

def raw(path):
    response = requests.get(hf_hub_url(repo, path, repo_type="dataset", revision=revision), timeout=60)
    response.raise_for_status()
    return response.content

def csv(path):
    return pd.read_csv(io.BytesIO(raw(path)))

archs = ["regnety016", "densenet121", "resnet50"]
stress_ids = [f"a-{a}-base-f1-s{s}" for a in archs for s in (1, 2, 3)]
pred_ids = [f"a-{a}-base-f{f}-s{s}" for a in archs for f in (0, 1, 2) for s in (1, 2, 3)]
tables = ["stress_tests", "shuffled_control_2026-09-08-final-epoch-r1", "ensemble_metrics",
          "tta", "calibration", "conformal", "master_architectures", "stage_a_quarantined",
          "hypothesis_outcomes", "stage_b_effects"]
paths = [f"tables/{t}.csv" for t in tables] + [f"runs/{r}/stress/results.csv" for r in stress_ids]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    frames = dict(zip(paths, pool.map(csv, paths)))
print("HF_REVISION", revision)
for t in tables:
    d = frames[f"tables/{t}.csv"]
    print("TABLE", t, "ROWS", len(d))
stress = frames["tables/stress_tests.csv"]
expected = {(r, k) for r in stress_ids for k in
            ("none", "bg_blank", "bg_swap", "tyre_crop", "mask_marking", "mask_damage", "grayscale")}
assert set(zip(stress.run_id, stress.intervention)) == expected and len(stress) == 63
parts = pd.concat([frames[f"runs/{r}/stress/results.csv"] for r in stress_ids], ignore_index=True)
sort = lambda d: d.sort_values(["run_id", "intervention"])[sorted(d.columns)].reset_index(drop=True)
pd.testing.assert_frame_equal(sort(stress), sort(parts), check_dtype=False)
assert np.isfinite(stress[["f1_macro", "acc", "qwk", "recall_low", "recall_high"]]).all().all()
print("NB08_PASS: all 63 keys, all nine source tables agree; finite metrics; n", stress.n.unique())
gate = frames["tables/shuffled_control_2026-09-08-final-epoch-r1.csv"]
print("CONTROL_AUDIT", gate.to_json(orient="records"))
for f in range(3):
    r = f"stress-resnet18-shufflectl_r2-f{f}-s1"
    status = json.loads(raw(f"runs/{r}/STATUS.json"))
    h = csv(f"runs/{r}/metrics/epochs.csv")
    assert list(h.epoch) == list(range(1, 13)) and status["status"] == "completed"
    assert np.isclose(h.iloc[-1].val_f1_macro, gate.loc[gate.fold.eq(f), "final_val_f1_macro"].iloc[0])
    print("CONTROL_STATUS", f, status.get("iso"), status.get("lib_version"), status.get("account"))
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    preds = list(pool.map(lambda r: pd.read_parquet(io.BytesIO(raw(f"runs/{r}/per_sample/predictions.parquet"))), pred_ids))
for rid, d in zip(pred_ids, preds):
    fold = int(rid.split("-f")[1].split("-")[0])
    assert len(d) == {0: 186, 1: 128, 2: 104}[fold] and d.image_id.is_unique
    prob = d[["prob_low", "prob_mid", "prob_high"]].to_numpy()
    assert np.isfinite(prob).all() and np.allclose(prob.sum(axis=1), 1, atol=1e-4)
print("NB09_PREDICTIONS", len(preds), sum(map(len, preds)), "all readable/unique/normalised")
for t in ("ensemble_metrics", "tta", "calibration", "conformal", "hypothesis_outcomes"):
    d = frames[f"tables/{t}.csv"]
    print(t.upper(), d.to_json(orient="records") if t != "ensemble_metrics" else d.groupby("kind").size().to_dict())
figures = sorted(p for p in files if p.startswith("analysis/fig") and p.endswith(".png"))
from PIL import Image
for path in figures:
    with Image.open(io.BytesIO(raw(path))) as im:
        im.verify()
print("READABLE_FIGURES", figures)
print("XAI_EXAMPLES", sum(p.startswith("analysis/xai_examples/") for p in files))
print("MASTER_ARCHS", list(frames["tables/master_architectures.csv"].iloc[:, 0]))
print("QUARANTINED", frames["tables/stage_a_quarantined.csv"].arch.unique())
print("AUDIT FINISHED")
