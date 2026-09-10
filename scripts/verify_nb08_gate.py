"""Exercise the generated NB08 gate without GPU training or HF writes."""
import ast
import base64
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
nb = json.loads((ROOT / "notebooks/NB08_StressTests.ipynb").read_text(encoding="utf-8"))
codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
for code in codes:
    ast.parse(code)
tree = ast.parse(codes[0])
payload = next(ast.literal_eval(n.value) for n in tree.body
               if isinstance(n, ast.Assign) and any(
                   isinstance(t, ast.Name) and t.id == "_LIB" for t in n.targets))
assert base64.b64decode("".join(payload)) == (ROOT / "tyrelib/tyrelib.py").read_bytes()
gate = next(c for c in codes if "CONTROL_GATE_REVISION =" in c)
final_scores = [0.2634482758620689, 0.2853424819590677, 0.5104378189632017]
best_scores = [0.15, 0.6513734080707858, 0.6928651189667424]


def exercise(scores, expected_pass, missing=False):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = []
        for fold, score in enumerate(scores):
            rid = f"stress-resnet18-shufflectl_r2-f{fold}-s1"
            p = root / "runs" / rid / "metrics/epochs.csv"
            p.parent.mkdir(parents=True)
            count = 11 if missing and fold == 0 else 12
            pd.DataFrame({"epoch": range(1, count + 1),
                          "val_f1_macro": [score] * count}).to_csv(p, index=False)
            rows.append(dict(run_id=rid, fold=fold, status="completed",
                             final_val_f1_macro=score, best_val_f1_macro=best_scores[fold]))
        original = lambda root, fold: (pd.DataFrame({"proxy_label": ["a", "b", "c"]}), pd.DataFrame())
        tl = SimpleNamespace(load_split=original, HF_REPO_DEFAULT="test/repo",
                             read_epoch_history=lambda p: pd.read_csv(p))
        uploads = []
        def run_all(configs, **kwargs):
            assert kwargs["isolate_runs"] is False
            assert tl.load_split is not original
        sess = SimpleNamespace(
            stage_dir=root,
            config=lambda arch, fold, seed, **kw: {"run_id": rows[fold]["run_id"]},
            run_all=run_all,
            aggregate_remote=lambda ids, **kw: pd.DataFrame(rows),
            uploader=SimpleNamespace(enqueue=lambda *args, **kw: uploads.append(args)),
            push_now=lambda reason: True)
        ns = dict(np=np, pd=pd, Path=Path, tl=tl, sess=sess,
                  hf_hub_download=lambda repo, rel, **kw: str(root / rel))
        failed = False
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                exec(compile(gate, "NB08_control", "exec"), ns)
            except (RuntimeError, AssertionError):
                failed = True
        assert tl.load_split is original, "split loader was not restored"
        assert failed != expected_pass
        if expected_pass:
            assert ns["CONTROL_GATE_PASSED"] and uploads


exercise(final_scores, True)
exercise([0.6, 0.6, 0.6], False)
exercise([float("nan"), 0.3, 0.3], False)
exercise(final_scores, False, missing=True)
print("PASS: NB08 syntax/payload; public-score fixture passes; high, NaN, and incomplete histories block")
