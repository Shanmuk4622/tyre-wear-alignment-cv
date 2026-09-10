"""Locked, descriptive S4b protocol. Does not train or access the network."""
import hashlib
import json
import numpy as np
import pandas as pd

SOURCE_REVISION = 'bf62f9e9cbedacc580aa42542da14a068b8f9215'
REVISION = 's4b-2026-09-09-r1'
FACTORS = {
    'sampler_classweighted': {'sampler_name':'class_weighted'},
    'transfer_random': {'pretrained':False},
    'sampler_uniform': {'sampler_name':'uniform'},
}

def make_plan(selection, effects):
    assert selection.arch.is_unique
    assert set(selection.selection_revision) == {'2026-08-30-r3'}
    selected = selection.loc[selection.selected_top3.eq(True), 'arch'].tolist()
    assert set(selected)=={'regnety016','densenet121','resnet50'}
    eligible = selection[selection.eligible.eq(True) & selection.seeds.eq(3) & selection.xai_status.eq('ok')]
    targets = eligible[~eligible.arch.isin(selected)].sort_values(['ter_norm','bar','arch'],ascending=[False,True,True]).arch.tolist()
    assert targets==['convnextv2_t','mobilenetv4']
    assert len(effects)==108 and not effects.duplicated(['arch','factor','fold','seed']).any()
    assert set(effects.arch)==set(selected) and set(effects.fold)=={1}
    assert effects.groupby('factor').size().eq(9).all()
    assert set(effects.seed)=={1,2,3} and np.isfinite(effects.delta_vs_stage_a).all()
    ranking = effects.groupby('factor',as_index=False).delta_vs_stage_a.mean().sort_values(['delta_vs_stage_a','factor'],ascending=[False,True])
    factors = ranking.head(3).factor.tolist()
    assert factors==list(FACTORS), 'Frozen discovery ranking does not match this revision'
    return dict(revision=REVISION,source_revision=SOURCE_REVISION,architectures=targets,
        discovery_architectures=selected,factors=factors,overrides=FACTORS,folds=[1],seeds=[1,2,3],
        epochs=60,primary='best_val_f1_macro selected by best QWK, paired by architecture/fold/seed',
        secondary='final_val_f1_macro at epoch 60',
        ranking='descending mean signed paired NB06 selected-epoch F1 delta across all 9 discovery runs; alphabetical tie break',
        discovery_mean_delta=dict(zip(ranking.factor,ranking.delta_vs_stage_a)),
        claim='cross-architecture directional replication on the same fold; not an independent dataset or significance test')

def plan_hash(plan):
    return hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def configs(sess, tl, plan):
    out=[]
    for factor in plan['factors']:
        for arch in plan['architectures']:
            for seed in plan['seeds']:
                cfg=sess.config(arch,1,seed,stage='c',technique='s4b_r1_'+factor,
                                _strict_resume=True,_single_gpu=True,
                                _max_epoch_seconds=600,**FACTORS[factor])
                tl.validate_config(cfg)
                assert cfg['max_epochs']==60 and cfg['input_resolution']==384
                out.append(cfg)
    assert len(out)==18 and len({c['run_id'] for c in out})==18
    return out

def analyse(plan, trained, baseline):
    base=baseline.set_index(['arch','fold','seed'])
    assert base.index.is_unique and len(base)==6
    rows=[]
    for r in trained.to_dict('records'):
        key=(r['arch'],int(r['fold']),int(r['seed']))
        factor=r['technique'].removeprefix('s4b_r1_')
        assert factor in plan['factors']
        a=base.loc[key]
        rows.append(dict(run_id=r['run_id'],arch=r['arch'],factor=factor,fold=1,seed=key[2],
            selected_delta=float(r['best_val_f1_macro']-a.best_val_f1_macro),
            final_delta=float(r['final_val_f1_macro']-a.final_val_f1_macro)))
    paired=pd.DataFrame(rows)
    assert len(paired)==18 and not paired.duplicated(['arch','factor','seed']).any()
    assert np.isfinite(paired[['selected_delta','final_delta']]).all().all()
    summary=paired.groupby(['arch','factor']).agg(n=('seed','size'),mean_selected_delta=('selected_delta','mean'),
        sd_selected_delta=('selected_delta','std'),mean_final_delta=('final_delta','mean')).reset_index()
    assert summary.n.eq(3).all()
    summary['discovery_delta']=summary.factor.map(plan['discovery_mean_delta'])
    summary['same_direction']=summary.mean_selected_delta * summary.discovery_delta > 0
    summary['interpretation']=np.where(summary.same_direction,'same_direction_descriptive','not_replicated_direction')
    return paired,summary
