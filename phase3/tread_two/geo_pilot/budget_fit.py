"""Fixture-measured CSV widths plus exact native shapes; no reliance on 1 TB scratch."""
import csv,math,os,shutil
from collections import defaultdict
from common import atomic,read
from schema_fit import SCHEMA

def project(engine):
    samples=23520+4860+1140+386+81;probes=2060;all_samples=samples+probes;views=all_samples*2
    planned=dict(HF_publications=200,HF_requests=50000,annotation_targets=3420,atlas_bands=all_samples*8,atlas_rows=1140*512,
                 batch_objectives=2940,dense_pixels=1140*2*128*96,endpoint_neighbours=5700,environment=50,epoch_metrics=3000,
                 epoch_schedule=23520,gate_checks=500,geometry_drift=all_samples,geometry_losses=all_samples,
                 global_embeddings=views*96,group_gradient_summary=2940*3,interventions=probes,latent_diagnostics=3100,
                 masked_patch_metrics=views*8,masked_patch_plan=views*8,native_chunks_index=10000,numeric_events=1000,
                 pair_losses=all_samples,parameter_gradients=len(list(engine.model.parameters()))*2940,patch_support=all_samples*32,
                 pipeline_timings=12000,point_values=all_samples*6,projected_embeddings=views*64,quality_bins=62*30,
                 quality_predictions=all_samples*3,region_metrics=all_samples*2,resource_samples=12000,
                 sample_observations=all_samples,selection=60,source_exposure=386*60,source_images=570,
                 spatial_maps_index=samples*15+probes*5,train_updates=2940,unit_ledger=12500,view_plans=views)
    measured={};bounds={};estimate=bound=0
    # Largest actual string widths seen in the complete disposable lifecycle.
    observed=defaultdict(lambda:defaultdict(int));sizes=defaultdict(int);counts=defaultdict(int)
    for family in SCHEMA:
        paths=list((engine.store.root/'csv'/family).glob('*.csv'))
        if family in engine.store.csv:paths.append(engine.store.root/engine.store.csv[family]['path'])
        for path in paths:
            sizes[family]+=path.stat().st_size
            with path.open(encoding='utf-8',newline='') as stream:
                for row in csv.DictReader(stream):
                    counts[family]+=1
                    for k,v in row.items():observed[family][k]=max(observed[family][k],len(v))
        per_row=0
        for col in SCHEMA[family]['columns']:
            name,dtype=col['name'],col['dtype'];width=observed[family][name]
            if dtype=='utf8':
                width=max(width,12 if name=='run' else 64 if 'sha256' in name else 100 if name in ('image_id','reference_id','confirmed_tyre_id','source_group','leakage_component','reference_component','group') else 128 if 'native_' in name or name in ('file','evidence_path') else 64)
            elif dtype=='bool':width=1
            elif dtype in ('int64','uint64'):
                width=max(width,20 if 'seed' in name else 6 if name in ('sample','first_sample','last_sample','unit','event_id','committed_unit','parameter_update','completed_update','update','base_sample') else 3 if name in ('row','column','dimension','atlas_row','source_row_start','source_row_end') else 12 if name in ('bytes','valid_pixels','target_pixels','predicted_pixels','tp','fp','fn') else 6)
            else:width=max(width,15 if dtype=='float32' else 25)
            per_row+=width+1
        # Dense/vectors have especially strict fixed coordinate/flag bounds.
        if family=='dense_pixels':per_row=6+1+3+2+15+1+2+2+8
        if family in ('global_embeddings','projected_embeddings'):per_row=12+6+1+2+15+5
        bound_family=per_row*planned[family]
        observed_width=sizes[family]/max(counts[family],1) if counts[family] else per_row
        # Account for larger later sample/update/epoch decimal IDs than tiny fixtures.
        adjustment=sum(3 for col in SCHEMA[family]['columns'] if col['name'] in ('sample','unit','update','first_sample','last_sample','parameter_update','completed_update'))
        average=min(per_row,observed_width+adjustment)
        projected=math.ceil(average*planned[family]);estimate+=projected;bound+=bound_family
        measured[family]=dict(fixture_rows=counts[family],fixture_bytes=sizes[family],planned_max_rows=planned[family],projected_bytes=projected)
        bounds[family]=dict(maximum_declared_row_bytes=per_row,projected_upper_bytes=bound_family)
    body_per_sample=2*128*96*4+512*3*4+512+9*4+2*2*32*96*4+2*8*2*4*16*16*4+2*8*16*16*4
    probe_per_sample=2*128*96*4+512*3*4+512+9*4
    native=samples*body_per_sample+probes*probe_per_sample
    allowances=600000000+150000000+100000000+100000000  # checkpoints/exports; gallery; sources/SDK; headers/index/retired prefixes
    adjusted=math.ceil(1.2*(native+estimate+allowances));worst=native+bound+allowances
    peak=max(adjusted,worst)+128*1024**2+100000000
    hardlink=engine.root/'hardlink_probe';hardlink.write_bytes(b'GEO')
    linked=engine.root/'hardlink_probe_link'
    try:
        os.link(hardlink,linked);hardlinks=os.stat(hardlink).st_ino==os.stat(linked).st_ino
    except OSError:hardlinks=False
    finally:
        hardlink.unlink(missing_ok=True);linked.unlink(missing_ok=True)
    report=dict(status='passed' if adjusted<=15000000000 and worst<=15000000000 and peak<=18000000000 and hardlinks else 'failed',
                samples_before_probes=samples,probe_samples=probes,measured_CSV=measured,typed_width_bounds=bounds,
                exact_native_body_bytes=native,measured_CSV_projection_bytes=estimate,typed_CSV_upper_bytes=bound,
                other_allowance_bytes=allowances,adjusted_measured_projection_bytes=adjusted,typed_upper_logical_bytes=worst,
                projected_peak_bytes=peak,logical_limit_bytes=15000000000,peak_limit_bytes=18000000000,
                free_reserve_bytes=2000000000,hardlinks_supported=hardlinks,
                note='Measured forecast uses1.2 factor; independent typed bound is already pessimistic. Runtime guards stop safely rather than drop telemetry.')
    atomic(engine.root/'STORAGE_PROJECTION.json',report)
    return report
