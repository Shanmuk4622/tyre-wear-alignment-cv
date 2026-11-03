"""Build the TRACE design specification and actual-data interactive report.

No measured training notebook, model fit, network request or credential access.
"""
import hashlib
import json
import sys
from pathlib import Path
assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tread_two/trace_design'
REPORT = ROOT / 'tread_two/trace_report'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Compact field declarations are expanded into typed, explicitly nullable columns.
# Role/image/view fields are nullable for update/run-level families, never fake 0.
common = [('run', 'utf8', False), ('stage', 'utf8', False), ('epoch', 'int64', True),
          ('update', 'int64', True), ('image_id', 'utf8', True), ('view', 'int64', True)]
spec = [
 ('source_images','source image','once','source_sha256:s role:s component:s domain:s label_proxy:i geometry_kind:s original_path:s'),
 ('geometry_provenance','frozen input artifact','once','artifact:s revision:s sha256:s selected:b inherited_status:s source_sha256:s'),
 ('target_cells','image tile cell','once per role','target_revision:s tile:i cell_y:i cell_x:i valid_pixels:i contrast:f gradient_energy:f orientation_axis:f orientation_cross:f relative_curvature:f contrast_occupancy:f'),
 ('target_normalization','target channel','train construction','channel:s weighted_sum:d weighted_square_sum:d count:d mean:f std:f floor_applied:b fit_role:s'),
 ('target_position_control','tile channel cell','train construction','tile:i channel:s cell_y:i cell_x:i weighted_sum:d count:d mean:f available:b'),
 ('target_input_inventory','input artifact','construction','logical_file:s sha256:s bytes:i source_revision:s target_revision:s'),
 ('view_plans','source draw view','every view','observation:i source_sha256:s appearance_seed:i brightness:f contrast:f gamma:f target_id:s geometry_id:s'),
 ('mask_plans','view tile','every view','observation:i tile:i band:i column:i mask_seed:i supported:b hidden:b valid_fraction:f'),
 ('sample_schedule','source draw','every draw','draw:i source_sha256:s component:s primary:b padding:b ordinal_in_epoch:i batch_position:i'),
 ('sample_observations','source draw','every draw','observation:i source_sha256:s primary:b eligible:b skip_reason:s supported_tiles:i supported_bands:i visible_tiles:i hidden_cells:i'),
 ('reconstruction_cells','view tile cell','every view','observation:i tile:i cell_y:i cell_x:i target_id:s valid_pixels:i hidden:b local_0:f local_1:f local_2:f local_3:f local_4:f local_5:f masked_0:f masked_1:f masked_2:f masked_3:f masked_4:f masked_5:f'),
 ('image_losses','observation view objective','every view','observation:i objective:s huber_sum:d weighted_channel_count:d available:b eligible:b image_mean:d'),
 ('objective_terms','optimizer update','every update','local:d masked:d pair:d variance:d covariance:d total:d local_weight:d masked_weight:d pair_weight:d variance_weight:d covariance_weight:d eligible_rows:i diversity_available:b'),
 ('update_gradients','parameter group','every update','group:s pre_clip_norm:d post_clip_norm:d clip_limit:d finite:b learning_rate:d optimizer_step:i'),
 ('parameter_statistics','parameter tensor','epoch and bounded step sample','parameter:s elements:i l2:d max_absolute:d mean:d std:d finite:b observation_frequency:s'),
 ('epoch_metrics','role task endpoint','every epoch','role:s task:s metric:s value:d numerator:d denominator:d images:i available:b'),
 ('component_metrics','role component task','every epoch','role:s component:s task:s metric:s value:d images:i target_cells:i available:b'),
 ('geometry_sentinels','source image','cache and endpoint','source_sha256:s frozen_export_sha256:s logits_sha256:s points_sha256:s masks_equal:b max_logit_difference:d human_points_available:b'),
 ('representation_spectra','role space eigenvalue','every epoch','role:s space:s component_centered:b dimension:i eigenvalue:d fraction:d image_count:i std_median:d effective_rank:d'),
 ('local_token_statistics','role layer tile','every epoch','role:s layer:s tile:i image_count:i feature_count:i effective_rank:d std_median:d leading_fraction:d'),
 ('scatter_decomposition','role feature space','every epoch','role:s space:s components:i total_scatter:d within_scatter:d between_scatter:d between_fraction:d decomposition_error:d'),
 ('control_losses','source image control','baseline validation endpoints','control:s role:s model_sum:d control_sum:d common_count:d model_mean:d control_mean:d available:b'),
 ('control_coverage','source image control','baseline validation endpoints','control:s role:s model_native_count:d control_native_count:d common_count:d coverage:d missing_reason:s'),
 ('intervention_metrics','source image variant','fixed endpoint probes','variant:s plan_sha256:s space:s cosine_change:d l2_change:d local_loss:d masked_loss:d support_changed:b'),
 ('retrieval_neighbors','query neighbor rank','endpoints','query_sha256:s neighbor_id:s neighbor_sha256:s neighbor_component:s rank:i distance:d excluded_self:b permitted:b'),
 ('native_chunks_index','native shard','each sealed shard','file:s kind:s first_observation:i last_observation:i shape:s dtype:s bytes:i sha256:s committed_unit:i'),
 ('state_ledger','atomic completed unit','every complete update and phase','unit:i previous_unit_sha256:s state_sha256:s csv_prefix_sha256:s native_prefix_sha256:s next_cursor:s chain_sha256:s'),
 ('resume_events','restore event','every session/restore','session:s local_unit:i remote_unit:i chosen_unit:i revision:s compatibility:b action:s replayed_incomplete_update:b'),
 ('numerical_events','numerical event','event and finite checks','event:s tensor:s finite:b minimum:d maximum:d action:s available:b'),
 ('resource_measurements','device measurement','update peaks and bounded heartbeat','session:s device:s allocated_bytes:i reserved_bytes:i max_allocated_bytes:i max_reserved_bytes:i peak_reset_unit:i host_RSS_bytes:i free_bytes:i measurement_scope:s'),
 ('staging_inventory','publication stage object','every publication','snapshot:s file:s bytes:i sha256:s hardlink:b staging_bytes:i logical_total_bytes:i actual_free_bytes:i'),
 ('transport_events','HTTP event','every parent request','session:s event_id:s category:s status:i attempt:i bytes:i retry_after_seconds:d quota_remaining:i reset_seconds:d duration_seconds:d'),
 ('transport_receipts','verified selecting commit','every publication','snapshot:s revision:s owner_released:b verified:b last_update:i pending_updates:i event_coverage_through:s unsent_events:i'),
 ('errors','error/stop event','every event','session:s phase:s exception_type:s sanitized_message:s traceback_file:s last_committed_unit:i last_remote_revision:s flush_attempted:b'),
 ('endpoint_records','checkpoint endpoint','selected and final','endpoint:s available:b epoch_selected:i parameter_update:i file:s sha256:s provenance:s scientific_status:s'),
 ('eligibility','validation checkpoint','every epoch','geometry:b exposure:b control_coverage:b masked_utility:b local_utility:b projector_diversity:b encoder_diversity:b within_component_diversity:b finite:b eligible:b objective:d rejection_reasons:s'),
 ('source_exposure','source epoch','every epoch','source_sha256:s primary_draws:i padding_draws:i eligible_draws:i skipped_draws:i views:i'),
 ('environment','environment key','every session','session:s key:s value:s available:b identity_relevant:b'),
 ('run_configuration','configuration key','once and explicit repair','key:s value:s type:s protocol_revision:s source_sha256:s'),
 ('artifact_gaps','evidence gap','once and every discovered gap','artifact:s requested_grain:s available:b reason:s reconstruction_inputs:s interpretation_limit:s'),
 ('timings','phase duration','every update/stage/transport boundary','session:s phase:s start_monotonic:d duration_seconds:d images:i complete:b'),
 ('source_hashes','source artifact','once and every session verification','logical_file:s bytes:i sha256:s origin:s frozen:b'),
 ('preflight_gates','engineering gate','pre-fit and recovery','gate:s passed:b evidence_file:s evidence_sha256:s actual_Kaggle:b actual_HF:b detail:s'),
 ('target_channel_diagnostics','role channel','construction','role:s channel:s count:i mean:d std:d minimum:d maximum:d zero_fraction:d unavailable_cells:i'),
 ('spatial_moment_vectors','observation view coordinate','every view','observation:i moment:s channel:i value:f space:s native_file:s'),
 ('parameter_deltas','parameter group','every epoch','group:s reference:s l2_delta:d max_absolute_delta:d unchanged:b elements:i'),
 ('reconstruction_verification','independent reduction','review','artifact:s grain:s compared_values:i max_absolute_difference:d max_relative_difference:d tolerance:s passed:b'),
 ('target_space_diagnostics','role target space','construction and final review','role:s space:s dimensions:i images:i within_component_centered:b effective_rank:d leading_fraction:d scope:s'),
]
assert len(spec) == 48
types = {'s':'utf8', 'i':'int64', 'f':'float32', 'd':'float64', 'b':'bool'}
families = {}
for name, grain, frequency, fields in spec:
    columns = [{'name': n, 'dtype': t, 'nullable': nullable} for n,t,nullable in common]
    for field in fields.split():
        key, kind = field.split(':')
        assert key not in {c['name'] for c in columns}
        columns.append({'name': key, 'dtype': types[kind], 'nullable': True,
                        'missingness': 'Empty CSV cell only when explicitly unavailable/not applicable; record reason/count.'})
    families[name] = {'grain': grain, 'frequency': frequency, 'writer_status': 'planned_not_implemented', 'columns': columns}
write(BASE / 'SCHEMAS.json', {'status':'draft_specification_not_measured_writer_coverage', 'version':'trace-schema-draft-v1',
       'families':families, 'families_count':48, 'field_declarations':sum(len(f['columns']) for f in families.values()),
       'CSV_null':'empty; never reinterpret as measured zero', 'bool_values':[0,1],
       'units':{'points':'image-width fractions', 'resource_memory':'bytes', 'durations':'seconds',
                'loss':'standardised descriptor Smooth L1 beta0.1', 'eigenvalues':'sample covariance ddof1'},
       'precision':{'float32':'round-trip binary32 with nine significant digits; verify native equality', 'float64':'17 significant digits or equivalent round-trip'},
       'fit_delivery_gate':'Every declared family/field must map to an exercised writer before measured notebook delivery.'})

budget = {'status':'draft_math_requires_actual_writer_and_dual_T4_measurement', 'assumed_output_bytes':20000000000,
          'logical_ceiling_bytes':15000000000, 'projected_peak_ceiling_bytes':18000000000, 'reserve_bytes':2000000000,
          'optional_1TB_required':False, 'training_views':47040, 'shape_per_view':[32,6,4,4],
          'one_prediction_body_train_bytes':47040*32*6*4*4*4, 'all_three_token_bodies_train_bytes':3*47040*32*48*4,
          'spatial_moment_train_bytes':47040*240*4, 'full_target_all570_bytes':570*32*6*4*4*4,
          'full_weight_all570_bytes':570*32*1*4*4*4,
          'forbidden_all_training_view_input_lc_bytes':47040*32*4*64*64*4,
          'missing_before_freeze':['All formatted CSV family sizes', 'Validation/baseline/endpoint/probe bodies', 'Staging and actual allocator peaks',
                                   'Both optimizer/recovery slots', 'Every epoch weight and transport index', 'Conservative uncertainty multiplier'],
          'on_over_budget':'stop before long fit; never silently drop promised evidence'}
write(BASE / 'BUDGET.json', budget)
write(BASE / 'STATUS.json', {'date':'2026-11-02','name':'TREAD-TWO / TRACE','status':'design_and_local_prototype_checked',
       'design_complete':True,'core_prototype_implemented':True,'actual_386_train_target_audit':True,
       'trainable_parameters':71022,'frozen_geometry_parameters':1399227,
       'full_training_runtime_implemented':False,'CSV_runtime_writers_48_implemented':False,'Kaggle_notebooks_delivered':False,
       'dual_T4_proof':False,'live_HF_continuation_proof':False,'measured_training_started':False,
       'new_photos':0,'workstation_changes':False,'local_credentials_read':False,'new_HF_publication':False,
       'next':'new data/transaction/store/writers then resource/HF gates and complete automatic notebook batch'})
pins = json.loads((ROOT / 'tread_two/geo_pilot/PLAN.json').read_text(encoding='utf-8'))
write(BASE / 'SOURCES.json', {'status':'design_input_pins_not_new_measured_protocol','data':pins['data'], 'original_anchor':pins['anchor'],
       'GEO_fit':json.loads((ROOT / 'tread_two/geo_postrun/FIT_INPUT.json').read_text(encoding='utf-8')),
       'geometry_export_sha256':'8358e0595d46caee1160818ad975042f2031e14f7d0fa6f2ac105b9312325450',
       'geometry_selection_status':'GEO unselected final epoch60; original selected=None preserved',
       'GEO_NB20_verification_sha256':sha(Path(json.loads((ROOT / 'tread_two/nb20_completion/LATEST.json').read_text())['path']) / 'VERIFICATION.json'),
       'prototype_files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (BASE/'runtime').glob('*.py')}})

data = json.loads((REPORT / 'EXAMPLES.json').read_text(encoding='utf-8'))
payload = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<','\\u003c')
page = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TREAD-TWO / TRACE design explorer</title>
<style>
:root{font-family:system-ui,sans-serif;color:#182e3c;background:#eef3f5}body{margin:0}main{max-width:1160px;margin:auto;padding:32px 22px}h1{font-size:clamp(28px,4vw,44px);margin:10px 0}h2{font-size:20px}p{line-height:1.6}.eyebrow{color:#22766f;font-weight:750;letter-spacing:.09em;font-size:12px}.badge{display:inline-block;background:#fff0d9;color:#7b4e0c;border-radius:20px;padding:6px 12px;font-size:13px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.card,.panel{background:white;padding:20px;border-radius:16px;border:1px solid #d9e4e8}.card strong{display:block;font-size:27px}.card span{font-size:13px;color:#567}.flow{display:flex;align-items:stretch;gap:8px;flex-wrap:wrap;margin:20px 0}.flow div{padding:12px;background:#dbece9;border-radius:10px;flex:1;min-width:130px}.flow b{display:block;font-size:13px}.flow small{font-size:12px}label{font-weight:650;font-size:13px}select,button{padding:10px 12px;border-radius:9px;border:1px solid #bed0d7;background:#fff;color:#183d4c;font:inherit}button{cursor:pointer}button.primary{background:#176e67;color:white}button:focus-visible,select:focus-visible{outline:3px solid #b07321}.controls{display:flex;gap:12px;align-items:end;flex-wrap:wrap}.controls label{display:grid;gap:6px}.views{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px}.visual{display:flex;justify-content:center;background:#e5ecef;border-radius:12px;padding:12px}.visual img,.visual canvas{width:192px;height:384px;object-fit:fill}.visual canvas{image-rendering:pixelated}#caseid{font-size:12px;overflow-wrap:anywhere;color:#476273}.note{padding:14px;background:#f1f6f8;border-left:3px solid #26877d;font-size:13px}.readout{font-variant-numeric:tabular-nums;min-height:24px;font-size:13px}footer{font-size:12px;color:#567;margin-top:24px}@media(max-width:650px){.cards{grid-template-columns:1fr 1fr}.views{gap:8px}.visual img,.visual canvas{width:128px;height:256px}.panel{padding:12px}}
</style><main>
<div class="eyebrow">PHASE 3 · EXISTING DATA · 2 NOVEMBER 2026</div>
<h1>TREAD-TWO / TRACE</h1><span class="badge">Design prototype · no measured training yet</span>
<p>Learn fixed local tread structure while keeping geometry frozen. Explore observed descriptor maps from four existing training images below.</p>
<div class="cards"><div class="card"><strong>71,022</strong><span>new texture parameters</span></div><div class="card"><strong>386</strong><span>training photos audited</span></div><div class="card"><strong>197,632</strong><span>full target cells retained</span></div><div class="card"><strong>0</strong><span>new photos or measured fits</span></div></div>
<div class="flow"><div><b>Frozen geometry</b><small>Original anchor + unselected GEO final adapters</small></div><div><b>Fixed target maps</b><small>Contrast, edges, orientation, curvature and occupancy</small></div><div><b>Local + masked learning</b><small>Shared encoder and decoder; hidden raw pixels removed</small></div><div><b>Spatial evidence</b><small>240D moments + guarded 6D auxiliary projector</small></div></div>
<section class="panel"><h2>Observed targets, before learning</h2>
<div class="controls"><label>Training example<select id="example"></select></label><label>Descriptor channel<select id="channel"></select></label><label>Display<select id="display"><option value="all">All supported cells</option><option value="masked">Highlight one hidden tile per band</option></select></label><button id="previous">Previous</button><button id="next">Next</button><button id="download" class="primary">Export this map as CSV</button></div>
<p id="caseid"></p><div class="views"><div><h3>Existing canonical image</h3><div class="visual"><img id="photo" alt="Existing training tyre in the saved GEO canonical coordinates"></div></div><div><h3>Fixed observed descriptor</h3><div class="visual"><canvas id="map" width="256" height="512" aria-label="Descriptor cell heatmap"></canvas></div></div></div>
<p id="readout" class="readout">Move over a cell to inspect its value.</p><p id="scale" class="readout"></p>
<p class="note">These are image-derived targets, not model predictions, measured tread depth, damage labels or confidence scores. White cells are unavailable. The optional hidden-tile overlay illustrates the mask layout; it is not a learned explanation.</p></section>
<section class="panel" style="margin-top:18px"><h2>What the audit changed</h2><p>Six image averages retain little variation (effective rank 1.458). Full common spatial maps retain much more (rank 15.657; within-component centred rank 19.854). TRACE therefore keeps local maps and explicit row/column moments. These are train-only descriptor diagnostics, not validation performance.</p><p>The current CPU checks cover target leakage, objective arithmetic, gradients, frozen-geometry equivalence and complete-state restart. A full transactional trainer, dual-T4 proof and live HF continuation are still required before one measured run.</p></section>
<footer>Four training examples selected by fixed source order. All 386 train photos were audited. New design records are local; the future Kaggle preparation stage must publish complete CSV/native evidence to HF. Completed experiments and both workstations are preserved.</footer>
<script type="application/json" id="data">__DATA__</script>
<script>
const cases=JSON.parse(document.getElementById('data').textContent);
const el=id=>document.getElementById(id);let caseIndex=0,channelIndex=0;
const labels=['Local contrast','Gradient energy','Orientation axis','Orientation cross','Relative curvature','Contrast occupancy'];
cases.forEach((c,i)=>{const o=document.createElement('option');o.value=String(i);o.textContent='Train example '+(i+1);el('example').append(o)});
labels.forEach((name,i)=>{const o=document.createElement('option');o.value=String(i);o.textContent=name;el('channel').append(o)});
el('example').value='0';el('channel').value='0';
function current(){return cases[caseIndex].channels[channelIndex]}
function hidden(y,x){return Math.floor(x/4)===(Math.floor(y/4)+caseIndex)%4}
function render(){const c=cases[caseIndex],a=current();el('photo').src=c.image;el('caseid').textContent=(caseIndex+1)+' / '+cases.length+' · train only · '+c.image_id;
 const values=a.values.flat().filter((v,i)=>a.available.flat()[i]);const lo=Math.min(...values),hi=Math.max(...values),ctx=el('map').getContext('2d');ctx.clearRect(0,0,256,512);
 for(let y=0;y<32;y++)for(let x=0;x<16;x++){let color='#fff';if(a.available[y][x]){const t=hi===lo?.5:(a.values[y][x]-lo)/(hi-lo);color='rgb('+Math.round(20+220*t)+','+Math.round(45+140*(1-Math.abs(2*t-1)))+','+Math.round(130-90*t)+')';}ctx.fillStyle=color;ctx.fillRect(x*16,y*16,16,16);if(el('display').value==='masked'&&hidden(y,x)){ctx.fillStyle='rgba(255,255,255,.55)';ctx.fillRect(x*16,y*16,16,16)}}
 ctx.strokeStyle='rgba(255,255,255,.65)';ctx.lineWidth=1;for(let x=0;x<=16;x+=4){ctx.beginPath();ctx.moveTo(x*16,0);ctx.lineTo(x*16,512);ctx.stroke()}for(let y=0;y<=32;y+=4){ctx.beginPath();ctx.moveTo(0,y*16);ctx.lineTo(256,y*16);ctx.stroke()}
 el('scale').textContent=labels[channelIndex]+' · per-example scale '+lo.toPrecision(4)+' to '+hi.toPrecision(4)+' · 16 × 32 cells';return {lo,hi}}
function inspect(x,y){const a=current();if(x<0||x>=16||y<0||y>=32)return null;const value=a.available[y][x]?a.values[y][x]:null;el('readout').textContent='Band '+Math.floor(y/4)+', column '+Math.floor(x/4)+', cell '+(y%4)+','+(x%4)+' · '+(value===null?'unavailable':value.toPrecision(7));return value}
function csv(){const a=current(),rows=['image_id,role,channel,cell_y,cell_x,available,value'];for(let y=0;y<32;y++)for(let x=0;x<16;x++)rows.push([cases[caseIndex].image_id,'train',a.name,y,x,a.available[y][x]?1:0,a.available[y][x]?a.values[y][x]:''].join(','));return rows.join('\n')+'\n'}
el('example').addEventListener('change',()=>{caseIndex=Number(el('example').value);render()});el('channel').addEventListener('change',()=>{channelIndex=Number(el('channel').value);render()});el('display').addEventListener('change',render);
el('next').addEventListener('click',()=>{caseIndex=(caseIndex+1)%cases.length;el('example').value=String(caseIndex);render()});el('previous').addEventListener('click',()=>{caseIndex=(caseIndex+cases.length-1)%cases.length;el('example').value=String(caseIndex);render()});
el('map').addEventListener('mousemove',e=>{const r=el('map').getBoundingClientRect();inspect(Math.floor((e.clientX-r.left)/r.width*16),Math.floor((e.clientY-r.top)/r.height*32))});
el('download').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([csv()],{type:'text/csv'})),a=document.createElement('a');a.href=url;a.download='TRACE_observed_descriptor.csv';a.click();URL.revokeObjectURL(url)});
globalThis.traceDesign={render,inspect,csv,cases,get state(){return {caseIndex,channelIndex}}};render();
</script></main></html>'''
(REPORT / 'TRACE_Design_Explorer.html').write_text(page.replace('__DATA__',payload), encoding='utf-8')
print(json.dumps({'status':'built','schema_families':len(families),'field_declarations':sum(len(f['columns']) for f in families.values()),
                  'report_examples':len(data),'report_bytes':(REPORT/'TRACE_Design_Explorer.html').stat().st_size,
                  'training_notebook_created':False}), flush=True)
