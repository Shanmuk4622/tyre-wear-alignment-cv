"""CPU regression checks for generated recovery notebooks; never writes to HF."""
import ast
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tyrelib"))
import tyrelib as tl

def sources(name):
    nb=json.loads((ROOT/"notebooks"/name).read_text())
    for i,c in enumerate(nb["cells"]):
        if c["cell_type"]=="code":
            s="".join(c["source"]); ast.parse(s,filename=f"{name}:{i}")
            yield s

class NoUpload:
    def enqueue(self,*a,**k): pass
    def enqueue_dir(self,*a,**k): pass

class Session:
    def __init__(self,path): self.stage_dir=path; self.uploader=NoUpload()
    def push_now(self,*a): pass
    def finish(self): pass

def main():
    for name in ("NB01A_Baseline_Recovery.ipynb","NB01B_Matched_RandomInit.ipynb",
                 "NB03A_Architecture_Audit.ipynb","NB10R_Analysis_Recovery.ipynb"):
        list(sources(name))
    print("PASS: all generated cells parse")
    analysis=list(sources("NB10R_Analysis_Recovery.ipynb"))
    h=next(s for s in analysis if s.startswith("outcomes=[]"))
    with tempfile.TemporaryDirectory(prefix="tyre_closure_test_") as temp:
        tmp=Path(temp)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ev=pd.DataFrame([dict(arch=a,ter_norm=i+1,sar=.1*i,dmgar=.2*i) for i,a in enumerate("abc")])
        a=pd.DataFrame([dict(arch=arch,best_val_f1_macro=.1*i+.01*f) for i,arch in enumerate("abc") for f in range(3)])
        st=pd.DataFrame([dict(arch=arch,intervention=k,recall_low=.7,recall_high=.8)
                         for arch in "abc" for k in ("none","mask_marking","mask_damage")])
        ns=dict(pd=pd,np=np,EV=ev,A=a,ST=st,TAB=tmp,OUT=tmp,plt=plt)
        exec(h,ns)
        row=ns["H"].set_index("hypothesis").loc["H2"]
        assert pd.isna(row.supported) and row.outcome=="inconclusive_undefined"
        print("PASS: constant H2 intervention effects remain inconclusive")
        fig10=next(s for s in analysis if 'for f in sorted(LOCAL.glob("runs/a-*/metrics/epochs.csv"))' in s)
        for rid,arch in (("a-resnet50-base-f0-s1","resnet50"),("a-convnextv2_s-base-f0-s1","convnextv2_s")):
            path=tmp/"runs"/rid/"metrics"; path.mkdir(parents=True)
            pd.DataFrame([dict(epoch=e,arch=arch,val_qwk=1/e,val_acc_session_demo=e/10) for e in (1,2)]).to_csv(path/"epochs.csv",index=False)
        ns.update(LOCAL=tmp,A=pd.DataFrame({"run_id":["a-resnet50-base-f0-s1"]}))
        exec(fig10,ns)
        assert list(ns["Q"].index)==["resnet50"] and ns["Q"].iloc[0,0]==.2
        print("PASS: Figure 10 excludes quarantine and reads final epoch")
        # Verify new training IDs and isolation; no training occurs in this test.
        train=next(s for s in sources("NB01B_Matched_RandomInit.ipynb") if 'cfgs=sess.configs' in s)
        assert 'isolate_runs=True' in train and 'technique="randmatched_r1",pretrained=False' in train
        assert 'tl.Trainer(' not in train
        from types import SimpleNamespace
        cfgs=[tl.Session.config(SimpleNamespace(stage="s1"),"resnet50",fold,seed,
                                technique="randmatched_r1",pretrained=False)
              for fold in range(3) for seed in (1,2,3)]
        assert len({c["run_id"] for c in cfgs})==9
        assert all(c["max_epochs"]==60 and c["input_resolution"]==384 and
                   c["pretrained"] is False and c["head_type"]=="coral" for c in cfgs)
        print("PASS: matched random arm uses fresh IDs and HF-aware isolated scheduler")
        if "--baseline" in sys.argv:
            ns=dict(tl=tl,sess=Session(tmp),DATA_ROOT=Path(r"D:/Dataset Download/Tire Dataset Prepared/FINAL"))
            src=list(sources("NB01A_Baseline_Recovery.ipynb"))
            for s in src[3:]: exec(s,ns)
            assert len(ns["result"])==15
            print("PASS: full baseline recovery executed on real dataset + pinned public HF, uploads mocked")

if __name__=="__main__": main()
