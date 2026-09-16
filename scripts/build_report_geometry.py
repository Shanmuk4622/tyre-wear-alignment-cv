"""Extend report from audited local JSON and saved prototype evidence; no network."""
import hashlib,json,math,shutil,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'docs/report'

def build_geometry(table):
    evidence=REPORT/'evidence/geometry';evidence.mkdir(exist_ok=True)
    manifest=[]
    def preserve(source,dest):
        source=ROOT/source;dest=REPORT/dest;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
        manifest.append(dict(source=source.relative_to(ROOT).as_posix(),local=dest.relative_to(REPORT).as_posix(),
            bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    audit=json.loads((ROOT/'outputs/segformer_completion_audit/AUDIT.json').read_text())
    assert audit['status']=='verified' and audit['completed_seeds']==3 and audit['paired_points_recomputed']==432
    for name in ['AUDIT.json','comparison_REPORT.json','comparison_CONTRACT.json','comparison_PAIRED_POINTS.json','smoke_STATUS.json']:
        preserve('outputs/segformer_completion_audit/'+name,'evidence/geometry/'+name)
    for seed in (1,2,3):
        for name in [f'runs_seed{seed}_TEST_FINAL.json',f'hrnet_runs_seed{seed}_TEST_FINAL.json']:
            preserve('outputs/segformer_completion_audit/'+name,'evidence/geometry/'+name)
        sg=json.loads((evidence/f'runs_seed{seed}_TEST_FINAL.json').read_text())['records']
        hr=json.loads((evidence/f'hrnet_runs_seed{seed}_TEST_FINAL.json').read_text())['records']
        row=next(r for r in audit['results'] if r['seed']==seed)
        for name,records in [('segformer',sg),('hrnet',hr)]:
            assert len(records)==144
            assert math.isclose(statistics.mean(abs(r['prediction']-r['label']) for r in records),row[name],abs_tol=1e-12)
        assert all(not r['fallback_used'] for r in sg)
    for name in ['learned-integration-check.json','learned-ui-check.json','smoke-check.json','ui-check.json','portrait-controls-check.json']:
        preserve('prototype/results/'+name,'evidence/geometry/prototype_'+name)
    for name in ['learned-image-check.png','video1-learned-diagram.png','video1-learned-workstation.png','alignment-bench-check.png']:
        preserve('prototype/results/'+name,'assets/'+name)
    for name in ['LEARNED_GEOMETRY_LOG.md','CALIBRATED_ALIGNMENT.md']:
        # Descriptions are documented by source hashes without rewriting the user's files.
        path=ROOT/'prototype'/name
        manifest.append(dict(source=path.relative_to(ROOT).as_posix(),kind='local_source_document',
            bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    data=audit['results'];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    colors=['#126d78','#d18c26']
    fig,ax=plt.subplots(figsize=(9,4.8))
    for i,r in enumerate(data):
        ax.bar(i-.19,100*r['hrnet'],.38,color=colors[0],label='HRNet' if i==0 else None)
        ax.bar(i+.19,100*r['segformer'],.38,color=colors[1],label='Matched SegFormer' if i==0 else None)
    ax.set(xticks=range(3),xticklabels=['Seed 1','Seed 2','Seed 3'],ylabel='Mean horizontal error (% image width)',ylim=(0,2.15))
    ax.legend(frameon=False,ncol=2);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Matched geometry: HRNet reduces mean point error',fontweight='bold',fontsize=15)
    fig.text(.08,.015,'24 images · 2 test tyres · 100% SegFormer coverage · no independent-sample uncertainty claim',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.95));fig.savefig(REPORT/'assets/geometry_seed_comparison.png',dpi=170);plt.close(fig)
    tyres=list(data[0]['per_tyre']);fig,ax=plt.subplots(figsize=(9,4.8))
    for i,t in enumerate(tyres):
        for j,(name,label) in enumerate([('hrnet','HRNet'),('segformer_with_fallback','Matched SegFormer')]):
            values=[r['per_tyre'][t][name]*100 for r in data];x=i+(j-.5)*.28
            ax.scatter([x-.035,x,x+.035],values,s=65,color=colors[j],label=label if i==0 else None)
            ax.plot([x-.09,x+.09],[statistics.mean(values)]*2,color=colors[j],lw=2)
    ax.set(xticks=range(2),xticklabels=['High-mileage tyre\nsession 006','New tyre\nsession 001'],ylabel='Mean horizontal error (% image width)',ylim=(0,3.3))
    ax.legend(frameon=False,ncol=2);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Both test tyres favour HRNet; error still varies by tyre',fontweight='bold',fontsize=14)
    fig.text(.08,.015,'Dots: three seeds. Horizontal ticks: seed means. Not population confidence intervals.',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.95));fig.savefig(REPORT/'assets/geometry_per_tyre.png',dpi=170);plt.close(fig)
    (REPORT/'evidence/geometry_manifest.json').write_text(json.dumps(dict(hf_revision=audit['revision'],
        scope='Matched geometry HF audit plus locally saved software-test evidence; no new execution',files=manifest),indent=2),encoding='utf-8')
    local=json.loads((evidence/'prototype_learned-integration-check.json').read_text());assert local['status']=='passed'
    timing=table(['Component','Five-run median (ms)','Observed range (ms)'],[
        [label,f"{statistics.median(r[k] for r in local['timings_ms']):.2f}",
            f"{min(r[k] for r in local['timings_ms']):.2f}–{max(r[k] for r in local['timings_ms']):.2f}"]
        for k,label in [('hrnet','HRNet-W18'),('matched','Matched SegFormer-B0')]])
    metrics=table(['Seed','HRNet error (% width)','SegFormer error (% width)','SegFormer coverage'],[
        [r['seed'],f"{r['hrnet']*100:.3f}",f"{r['segformer']*100:.3f}",'100%'] for r in data]+
        [['Mean',f"{audit['hrnet_mean']*100:.3f}",f"{audit['segformer_mean']*100:.3f}",'100%']])
    return {'GEOMETRY_RESULTS':metrics,'GEOMETRY_TIMING':timing}
