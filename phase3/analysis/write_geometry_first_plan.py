"""Author the complete design-only GEO plan and machine-readable contracts."""
import hashlib,json,re,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tread_two/geometry_first/plan';OUT.mkdir(parents=True,exist_ok=True)
def write(name,text):(OUT/name).write_text(text.strip()+'\n',encoding='utf-8')
def js(name,value):(OUT/name).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
plan=dict(status='design_complete_implementation_pending',family='TREAD-TWO/GEO',revision='geo-r1-plan-v1',date_Asia_Calcutta='2026-11-01',
    scope=dict(existing_photos_only=True,new_independent_tyres=0,workstation_changes=False,new_optimizer_updates_performed=0,new_model_forwards_performed=0,actual_Kaggle_gate_passed=False,measured_training_started=False),
    data=dict(kaggle_input='shanmuk4622/tire-dataset-prepared-phase2',checksum_file_sha256='06c6c9d003aff808babf7ebb5786f441e95064eb000f5e5ed5b155d0ee9e7c6a',overlay_sha256='f81a80d20b60ba505d94e31c26615e7e7171a91a8506948be6c392b68107a039',images=570,train=386,validation=81,historical_test=103,known_train_tyres=[1,3,4],conservative_train_components=[1,1,4],train_human_geometry_images=70,train_weak_geometry_images=152,train_no_point_target_images=164,validation_human_geometry_images=25,test_human_geometry_images=25),
    anchor=dict(model='TREAD-ONE r1',seed=3,epoch=20,parameters=1372159,weights_sha256='a5b872ed2e4cbfb3b9782b0df563e2606631744cd0274fee2da42a832f985f65',source_sha256='d16cd4eef75779044dd95900c0531fee474704cf88f523326ee069115347ba09',url='https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/a0296f18e02b12d28f7e51377783d68538032209/phase3/tread-one-r1/371c40c0c6aaedaf5c5970f7721a284b042494db7bb330065f6cbde8fc3a0b8c/report/weights.pt',frozen=True),
    architecture=dict(input_hw=[512,384],geometry_feature_shape=[48,128,96],guides=[384/1535,768/1535,1151/1535],mask_threshold=.5,mask_adapter_channels=[48,24,2],mask_logit_delta_bound=1.0,point_adapter_channels=[256,64,2],point_raw_delta_bound=.25,quality_channels=[64,32,3],atlas_hw=[512,256],atlas_align_corners=True,atlas_min_source_row_width=16,patch_hw=[64,64],patch_grid=[8,4],masked_tile_min_coverage=.25,guide_mask_gap_flag=.05,texture_input_channels=4,texture_encoder_channels=[24,32,64,96],axial_blocks=2,token_dimensions=96,projector_channels=[96,128,64],masked_decoder_channels=[96,128,1024],masked_target_shape=[4,16,16],transformer=False,EMA_teacher=False,shared_trainable_geometry_texture_parameters=False,classification_loss=False,target_max_trainable_parameters=1000000,target_max_total_parameters=3000000,actual_parameter_count_pending=True),
    fit=dict(candidate_fits=1,seed=1,epochs=60,batch=8,batches_per_epoch=49,updates=2940,draws=23520,texture_views=47040,schedule='complete386-image coverage plus6 documented padding draws per epoch; at least one largest-component and one other-component image per batch',precision='FP32',optimizer='AdamW',foreach=False,fused=False,betas=[.9,.999],eps=1e-8,weight_decay=.0001,geometry_lr=.0001,texture_lr=.001,quality_lr=.0003,lr_schedule='cosine by completed global update',separate_group_clip_norm=1.0,texture_ramp_epochs=5,minimum_SSL_images_per_batch=4,minimum_SSL_components_per_batch=2),
    losses=dict(segmentation='masked BCE + soft Dice, mean of2 regions',point_weight=2.0,weak_point_weight=.25,point_smooth_l1_beta=.02,containment_weight=.1,row_agreement_weight=.1,anchor_drift_weight=.1,quality_weight=.25,quality_smooth_l1_beta=.02,VIC_invariance_weight=25.0,VIC_variance_weight=25.0,VIC_covariance_weight=1.0,VIC_variance_floor=1.0,VIC_variance_epsilon=.0001,masked_prediction_weight=.5,masked_huber_beta=.1),
    selection=dict(mask_dice_floor_delta=-.002,human_point_MAE_ceiling_delta=.001,invalid_atlas_fraction_ceiling_delta=.01,projector_std_median_floor=.1,projector_effective_rank_floor=4,objective='masked_Huber / training_mean_predictor_Huber +0.1*paired_cosine_distance; lower is better; earliest tie',no_eligible_checkpoint='reject candidate; retain final and all evidence; do not substitute historical-test winner',quality_head_selection_metric=False,mileage_classification_selection_metric=False,success_masked_loss_relative_gain=.05,success_paired_cosine_distance=.05,success_minimum_SSL_draw_coverage=.90),
    publication=dict(repository='Shanmuk4622/tyre-wear-study',namespace='phase3/tread-two/geo-r1',secret='HF_TOKEN',normal_push_seconds=1800,major_cell_push=True,catchable_stop_attempt_immediate=True,native_epoch_checkpoints=60,zip=False,npz=False,private_HF_SDK='0.36.2 until separately tested transport revision',local_commit_attempts_per_hour=96,server_headers_authoritative=True),
    resources=dict(assumed_output_bytes=20000000000,logical_artifact_budget_bytes=15000000000,maximum_projected_peak_bytes=18000000000,free_reserve_bytes=2000000000,geometry_microbatch=1,training_gpu='cuda:0',frozen_feature_and_evaluation_gpu='cuda:1',DDP=False,scratch_1TB_required=False,live_measurement_required=True,wall_time_estimate='pending actual fixture; no speed claim',observation_native_format='uncompressed NPY shards, at most8complete updates; active raw append streams with committed byte prefixes and indexed shapes',storage_uncertainty_factor=1.2),
    notebooks=dict(NB16='planned GEO preparation, math/recovery/storage and actual dual-T4 gate; disposable engineering updates only',NB17='planned single resumable GEO measured pilot',NB18='planned completed-run review/export if NB17 completes; no automatic extra model fit'))
js('PLAN.json',plan)
plan['architecture']['minimum_SSL_supported_tiles']=4;plan['architecture']['minimum_SSL_supported_bands']=2
js('PLAN.json',plan)
# Exact draft field names. The first runtime freeze must add executable dtypes and tests.
families={
'source_images':('once', 'image_id', 'run,image_id,image_sha256,role,domain,confirmed_tyre_id,source_group,leakage_component,label_proxy,point_kind,human_target_available,weak_target_available,mask_sha256,ignore_sha256'),
'annotation_targets':('once', 'image_id,guide,side', 'run,image_id,guide,side,x_width_fraction,y_height_fraction,target_kind,optimization_weight,human_eligible,metric_available'),
'epoch_schedule':('every epoch before updates', 'epoch,draw', 'run,epoch,draw,batch,batch_slot,image_id,source_sha256,leakage_component,is_padding,base_augmentation_seed,view_a_seed,view_b_seed,mask_a_seed,mask_b_seed,schedule_sha256'),
'sample_observations':('every train/evaluation source pass', 'sample', 'run,sample,stage,endpoint,epoch,completed_update,parameter_update,image_id,source_sha256,geometry_valid,SSL_valid,SSL_skip_reason,guide_count,mask_valid_pixels,point_target_kind'),
'view_plans':('two texture views per eligible sample', 'sample,view', 'run,sample,view,brightness,contrast,gamma,geometry_augmentation_seed,texture_seed,mask_seed,masked_tiles,visible_tiles,masked_target_pixels'),
'region_metrics':('every sample and2regions', 'sample,region', 'run,sample,region,tp,fp,fn,valid_pixels,target_pixels,predicted_pixels,dice,iou,box_iou,target_box_available,predicted_box_available,predicted_box_left,predicted_box_top,predicted_box_right,predicted_box_bottom,target_box_left,target_box_top,target_box_right,target_box_bottom,bce_sum,bce_count'),
'point_values':('every sample and6points', 'sample,guide,side', 'run,sample,guide,side,prediction,target,absolute_error_width_fraction,smooth_l1_sum,target_kind,optimization_weight,human_metric_available'),
'atlas_bands':('every sample and8bands', 'sample,band', 'run,sample,band,valid_rows,valid_fraction,mean_width_width_fraction,clipped_fraction,point_mask_gap_width_fraction,unsupported'),
'atlas_rows':('selected/final source cohorts; all training row parameters additionally native', 'sample,atlas_row', 'run,sample,atlas_row,source_y,left_x,right_x,row_valid,source_clipped,grid_parameter_native_key'),
'patch_support':('every sample and32tiles', 'sample,patch', 'run,sample,patch,band,column,valid_fraction,pooling_mass,valid_target_fraction,source_row_start,source_row_end'),
'masked_patch_plan':('every eligible texture view and masked tile', 'sample,view,patch', 'run,sample,view,patch,masked,valid_loss_pixels,mask_weight_sum,target_mean,target_std,native_prediction_key,native_target_key,native_weight_key'),
'masked_patch_metrics':('every masked tile', 'sample,view,patch', 'run,sample,view,patch,huber_sum,huber_weight_count,absolute_error_sum,absolute_error_max,mean_predictor_huber_sum,mean_predictor_weight_count'),
'global_embeddings':('every eligible view, all96coordinates', 'sample,view,dimension', 'run,sample,view,dimension,value'),
'projected_embeddings':('every eligible view, all64coordinates', 'sample,view,dimension', 'run,sample,view,dimension,value'),
'pair_losses':('every eligible paired draw', 'sample', 'run,sample,invariance,masked_huber_a,masked_huber_b,pair_cosine_distance,valid_images_in_batch,valid_components_in_batch'),
'geometry_losses':('every sample; unreduced sufficient statistics', 'sample', 'run,sample,bce_sum,bce_count,dice_loss_sum,dice_region_count,point_loss_sum,point_weight_count,containment_sum,containment_count,row_agreement_sum,row_agreement_count,anchor_drift_sum,anchor_drift_count'),
'quality_predictions':('every sample and3error outputs', 'sample,quantity', 'run,sample,quantity,predicted_error,observed_error,target_available,human_only_target,loss_sum,loss_count,validated_for_deployment'),
'quality_bins':('each validation epoch and endpoints', 'stage,epoch,quantity,bin', 'run,stage,endpoint,epoch,quantity,bin,count,predicted_error_sum,observed_error_sum,absolute_error_sum,median_baseline_error_sum'),
'batch_objectives':('every completed update', 'update', 'run,update,epoch,geometry_objective,quality_objective,SSL_objective,variance_a,variance_b,covariance_a,covariance_b,texture_ramp,total_loss,geometry_samples,quality_mask_targets,quality_human_point_targets,SSL_pairs,texture_step_skipped'),
'train_updates':('every completed update', 'update', 'run,update,epoch,next_draw,geometry_optimizer_steps,texture_optimizer_steps,quality_optimizer_steps,geometry_lr,texture_lr,quality_lr,gradient_finite,update_committed,unit_sha256'),
'parameter_gradients':('every trainable tensor/update', 'update,parameter', 'run,update,group,parameter,gradient_available,numel,l2_norm,max_absolute,mean_absolute,zero_fraction,finite'),
'group_gradient_summary':('every group/update', 'update,group', 'run,update,group,pre_clip_norm,post_clip_norm,clip_coefficient,gradient_count,optimizer_step_applied,frozen_anchor_gradient_count'),
'geometry_drift':('every sample against same-view anchor', 'sample', 'run,sample,mask_logit_delta_max,mask_probability_delta_mean,point_delta_max_width_fraction,tyre_box_delta,tread_box_delta,anchor_weights_unchanged'),
'latent_diagnostics':('every batch and validation epoch', 'update,stage', 'run,update,stage,eligible_images,components,projector_std_min,projector_std_median,dimensions_below_floor,covariance_offdiag_energy,effective_rank,embedding_norm_min,embedding_norm_max,finite'),
'epoch_metrics':('every epoch and endpoint task/group', 'stage,epoch,task,group', 'run,stage,endpoint,epoch,task,group,numerator,denominator,value,available,unit,aggregation_definition'),
'source_exposure':('each epoch per original training image', 'epoch,image_id', 'run,epoch,image_id,geometry_draws,texture_views,padding_draws,SSL_valid_draws,SSL_skipped_draws,first_update,last_update'),
'pipeline_timings':('every update/evaluation unit', 'unit', 'run,unit,stage,feature_seconds,transfer_seconds,geometry_seconds,texture_seconds,record_seconds,publish_wait_seconds,evaluation_seconds'),
'resource_samples':('every update and publication boundary', 'unit', 'run,unit,gpu0_allocated_bytes,gpu0_reserved_bytes,gpu1_allocated_bytes,gpu1_reserved_bytes,host_rss_bytes,working_free_bytes,logical_artifact_bytes,staging_bytes'),
'numeric_events':('only when event occurs; empty table valid', 'event_id', 'run,event_id,unit,event_kind,stage,parameter_or_tensor,finite,action,replay_from_update,reason'),
'endpoint_neighbours':('selected/final diagnostic retrieval', 'sample,reference_id', 'run,sample,reference_id,reference_sha256,reference_component,reference_stage,cosine_similarity,rank,same_source_blocked,same_component_blocked,proxy_label_context_only'),
'spatial_maps_index':('every retained native map chunk', 'sample,quantity', 'run,sample,quantity,native_file,native_key,shape,dtype,coordinate_system,sha256,precision_note'),
'native_chunks_index':('each closed epoch or snapshot', 'file', 'run,file,kind,epoch,first_sample,last_sample,bytes,sha256,immutable,committed_unit'),
'selection':('each validation epoch', 'epoch', 'run,epoch,validation_mean_dice,baseline_mean_dice,validation_human_point_MAE,baseline_human_point_MAE,invalid_atlas_fraction,baseline_invalid_atlas_fraction,projector_std_median,effective_rank,geometry_eligible,collapse_eligible,selection_objective,selected_so_far,rejection_reasons'),
'interventions':('selected/final predeclared probes', 'sample,variant', 'run,sample,variant,base_sample,mask_dice_change,point_MAE_change,embedding_cosine_change,quality_error_change,valid_fraction_change,scope'),
'dense_pixels':('selected/final identity cohorts only; run identity bound by file manifest; each row is one4x4report block; all logits additionally native', 'sample,region,row,column', 'sample,region,row,column,probability,predicted,target_foreground_subpixels,valid_subpixels'),
'unit_ledger':('every complete atomic unit', 'unit', 'run,unit,kind,update,csv_prefix_sha256,native_state_sha256,previous_unit_sha256,current_unit_sha256'),
'HF_requests':('every request category/event; no secrets', 'event_id', 'run,event_id,category,method,http_status,retry_delay_seconds,rate_remaining,reset_seconds,bytes_transferred,outcome'),
'HF_publications':('every publication attempt', 'attempt', 'run,attempt,reason,local_update,remote_update,revision,manifest_sha256,new_objects,verified_objects,elapsed_seconds,owner_released,pending'),
'environment':('once and recovery attempts', 'record_id', 'run,record_id,python,torch,cuda,cudnn,numpy,pillow,HF_SDK,devices,free_output_bytes,source_inventory_sha256,dataset_checksum,anchor_sha256'),
'gate_checks':('preflight and final audit', 'check', 'run,check,stage,passed,observed,required,evidence_path,fixture_or_measured')}
schemas={}
string_fields=set('run image_id image_sha256 source_sha256 confirmed_tyre_id source_group leakage_component label_proxy point_kind mask_sha256 ignore_sha256 target_kind schedule_sha256 stage endpoint SSL_skip_reason point_target_kind native_prediction_key native_target_key native_weight_key quantity parameter group unit_sha256 event_kind parameter_or_tensor action reason previous_unit_sha256 current_unit_sha256 csv_prefix_sha256 native_state_sha256 reference_id reference_sha256 reference_component reference_stage native_file native_key shape dtype coordinate_system sha256 precision_note file kind category method outcome revision manifest_sha256 python torch cuda cudnn numpy pillow HF_SDK devices source_inventory_sha256 dataset_checksum anchor_sha256 check observed required evidence_path fixture_or_measured aggregation_definition variant scope unit'.split())
bool_fields=set('human_target_available weak_target_available human_eligible metric_available is_padding geometry_valid SSL_valid human_metric_available source_clipped unsupported masked target_available human_only_target validated_for_deployment texture_step_skipped gradient_finite update_committed gradient_available finite optimizer_step_applied anchor_weights_unchanged available same_source_blocked same_component_blocked proxy_label_context_only immutable geometry_eligible collapse_eligible selected_so_far owner_released pending passed target_box_available predicted_box_available'.split())
integer_fields=set('guide sample region side row column atlas_row patch band view dimension bin record_id event_id attempt update parameter_update completed_update epoch draw batch batch_slot base_augmentation_seed geometry_augmentation_seed texture_seed mask_seed view_a_seed view_b_seed mask_a_seed mask_b_seed guide_count mask_valid_pixels masked_tiles visible_tiles masked_target_pixels tp fp fn valid_pixels target_pixels predicted_pixels valid_rows target_foreground_subpixels valid_subpixels source_row_start source_row_end valid_loss_pixels valid_images_in_batch valid_components_in_batch dice_region_count geometry_samples quality_mask_targets quality_human_point_targets SSL_pairs next_draw geometry_optimizer_steps texture_optimizer_steps quality_optimizer_steps numel gradient_count frozen_anchor_gradient_count eligible_images components dimensions_below_floor first_update last_update geometry_draws texture_views padding_draws SSL_valid_draws SSL_skipped_draws first_sample last_sample bytes committed_unit verified_objects new_objects local_update remote_update rate_remaining http_status'.split())
for name,(cadence,key,columns) in families.items():
    entries=[]
    for column in columns.split(','):
        unit='bytes' if column.endswith('_bytes') or column=='bytes' else 'seconds' if 'seconds' in column else 'width-normalized coordinate/error' if 'width_fraction' in column else 'dimensionless fraction/statistic' if any(w in column for w in ['dice','iou','fraction','cosine','variance','covariance','probability','effective_rank','invariance','huber','loss','norm']) else 'index/count/text/boolean as defined by field name'
        dtype='utf8' if column in string_fields else 'bool' if column in bool_fields else ('uint64' if column.endswith('_seed') else 'int64') if column in integer_fields else 'float64'
        if column.startswith(('predicted_box_','target_box_')) and not column.endswith('_available'):dtype='int64'
        if (name in ('global_embeddings','projected_embeddings') and column=='value') or (name=='dense_pixels' and column=='probability'):dtype='float32'
        if column=='unit':dtype='utf8' if name=='epoch_metrics' else 'int64'
        entries.append(dict(name=column,dtype=dtype,definition=column.replace('_',' ')+' for '+name.replace('_',' ')+'; precise joins/denominators in RETENTION.md',units=unit,missing='Blank only when unavailable/not applicable; availability/count fields must explain why'))
    schemas[name]=dict(cadence=cadence,key=key.split(','),columns=entries,status='draft_required_family')
js('SCHEMAS.json',dict(status='design_only_not_runtime_validated',CSV_families=len(schemas),declared_columns=sum(len(v['columns']) for v in schemas.values()),families=schemas))
sources=['workstation/models/MODEL.json','workstation_validation/DELIVERY.json','tread_two/context_pilot/geometry.py','tread_two/context_pilot/frozen_anchor.py','tread_two/context_pilot/frozen_lc.py','tread_two/context_pilot/RECIPE.json','tread_two/context_pilot/data_atlas.py','tread_two/nb15_completion/98837b907abc/VERIFICATION.json','tread_two/readiness_report/SUMMARY.json','revision2_validation/nb05_publication/SCIENTIFIC_VERIFICATION.json','data_gate_validation/nb07_publication/remote/csv/historical_class_support.csv']
js('SOURCES.json',dict(local_sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},primary_references=[dict(title='VICReg original paper',url='https://arxiv.org/abs/2105.04906',role='Variance/invariance/covariance loss motivation; no project performance guarantee'),dict(title='PyTorch2.11 grid_sample',url='https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.grid_sample.html',role='CUDA backward caveat motivates a detached sampling boundary'),dict(title='PyTorch2.11 reproducibility',url='https://docs.pytorch.org/docs/2.11/notes/randomness.html',role='Same-environment RNG/state seam tests; no cross-platform bit-exact promise'),dict(title='HF Hub rate limits',url='https://huggingface.co/docs/hub/en/rate-limits',role='Separate HTTP buckets and server headers; commit action limits also apply')],checked_date='2026-11-01'))
write('README.md','''# TREAD-TWO / GEO — complete design plan

**Status: design complete; implementation, numerical/resource gates and measured results pending.** User requested the full geometry-first plan on 1 November 2026. This is a new objective under TREAD-TWO; the completed ATLAS classifier series and both workstations remain preserved.

| Read in order | Purpose |
|---|---|
| [Project plan](../../../phase3TreadTwoGeometryPlan.md) | Goal, evidence, deliverables, stages and acceptance |
| [Architecture](ARCHITECTURE.md) | Exact branch shapes, gradients, geometry/texture/error heads and loss math |
| [Experiment](EXPERIMENT.md) | Data, deterministic schedule, augmentations, selection and evaluation |
| [Retention](RETENTION.md) | Complete draft CSV fields, native raw tensors, budgets and explicit gaps |
| [Operations](OPERATIONS.md) | Kaggle cells, two-GPU ownership, recovery/HF/stop/resource behavior |
| [Implementation](IMPLEMENTATION.md) | File ownership, dependency order, actual test cases and next notebooks |
| [Risks/decisions](RISKS_AND_DECISIONS.md) | Failure controls, comparison limits, unresolved measurements |
| [PLAN.json](PLAN.json) | Chosen design values, status and named engineering thresholds |
| [SCHEMAS.json](SCHEMAS.json) | Every planned CSV family/field/key/cadence; executable dtypes pending freeze |
| [BUDGET.json](BUDGET.json) | Shape-based raw byte totals and provisional CSV/staging allowances; actual fixture still required |
| [SOURCES.json](SOURCES.json) | Pinned local evidence hashes and consulted primary sources |

No model implementation, training notebook, optimizer update, new inference, HF write or workstation change is claimed by these design documents. The next implementation deliverable is the planned NB16 engineering/contract gate; it will be a Kaggle .ipynb, not a pasted training script. NB17 is the single measured fit only after its gates pass. A successful new model is a hypothesis to test, not a result.

[Planning checks](../validation/PLAN_VALIDATION.json) cover source preservation, metadata/shape arithmetic and schedule feasibility only; model math and actual T4/resource checks are pending.
''')
write('ARCHITECTURE.md',r'''
# GEO architecture and mathematical contract

**Design only.** Names below are planned modules, not implemented files. Configuration values are in [PLAN.json](PLAN.json).

## One packaged model, separated learning paths

```mermaid
flowchart TD
    I[RGB512x384 + fixed3guide rows] --> A[Frozen TREAD-ONE seed3 epoch20\n1372159 parameters]
    A --> F[Frozen decoder features48x128x96]
    A --> B[Anchor mask logits + raw point parameters]
    F --> G[Zero-init bounded geometry adapters]
    B --> G
    G --> O[Refined2masks /6points / mask-derived boxes]
    O --> D[Detached visible tread atlas512x256]
    I --> D
    D --> V[Two nuisance views /32LCtiles each]
    V --> E[Custom96D patch encoder]
    E --> C[Two axial context blocks]
    C --> P[Global96D embedding /64D projector]
    C --> M[Masked tile prediction4x16x16]
    F --> Q[Detached geometry features +16scalars]
    O --> Q
    Q --> R[Experimental3error estimates]
    O --> L[Geometry supervised loss]
    P --> S[Invariance / variance / covariance loss]
    M --> S
    R --> T[Observed-error regression loss]
```

The frozen anchor's entire state stays bit-identical, in eval mode. New geometry adapters, texture modules and the error head use disjoint trainable parameters. Atlas construction detaches masks, row coordinates and RGB. Error-head inputs and targets are detached. Texture/error losses therefore have zero gradient to geometry adapters and the anchor. Geometry gradients cannot reach texture parameters. Clip each parameter group separately; a single global clip would couple updates despite disjoint branches, as the earlier shared-gradient study demonstrated.

## Geometry initialization and refinement

Anchor: exact project-selected TREAD-ONE r1, seed3/epoch20, SHA `a5b872ed2e4cbfb3b9782b0df563e2606631744cd0274fee2da42a832f985f65`. Source SHA `d16cd4eef75779044dd95900c0531fee474704cf88f523326ee069115347ba09`. This is not the failed projected r2 condition or a newly test-selected checkpoint. The existing full model file is loaded strictly; its old class head is frozen and unused by GEO losses/selection.

| Module | Chosen specification |
|---|---|
| Input / normalization |512 high ×384 wide RGB; original ImageNet mean/std and overlay guide definitions |
| Frozen features |48channels ×128×96, before old segmentation head |
| Mask residual |3×3 Conv48→24, GN8, SiLU, 1×1 Conv24→2; final weights/bias zero |
| Refined mask logits |`anchor_logits +1.0*tanh(mask_residual)`; independent overlapping tyre/tread sigmoid channels |
| Point residual |Existing 256D guide-row features → Linear256→64 → SiLU → Linear64→2; final zero; shared across3guides |
| Refined point raw parameters |`anchor_raw +0.25*tanh(point_residual)`; use original raw parameters directly, never inverse-sigmoid approximations |
| Point coordinates |`left=.99*sigmoid(a)`; `right=left+(1-left)*sigmoid(b)`;6normalized x coordinates |
| Localization |Bounding boxes enclosing masks at0.5; empty masks yield unavailable boxes |

At zero initialization, refined mask logits/points/boxes must match the anchor under the identical device/math path. The new forward adapter must expose old frozen features/raw point parameters without changing their computation or source checkpoint keys. Exact guide fractions remain384/1535,768/1535,1151/1535. Bounded residuals constrain change, not correctness. Weak points retain weight0.25, human points1, absent point targets0. Raw masks and explicit ignore regions remain intact.

Training mask targets use fixed4×4mean aggregation with any-ignore blocks excluded; soft Dice smoothing is1.0 and BCE uses logits. Reporting repeats native128×96predictions by4 to the512×384input grid and applies the original per-pixel ignore rule. Dense CSV stores quarter-grid probability/prediction plus the number of valid native subpixels and foreground subpixels in each4×4block. These integer counts reproduce full-grid TP/FP/FN exactly; fractional coarse targets are not mislabeled as binary human truth. Store every predicted/target box as explicit left/top/right/bottom half-open integer coordinates in the512×384report frame, with availability flags. Ground-truth boxes remain derived from the original valid target mask, not from a coarse averaged surrogate.

## Atlas and support

Use predicted tyre∩tread support and the existing visible-row algorithm: input512×384 → atlas512×256 → eight bands × four64×64tiles. Rows need at least16source pixels. Keep row-valid/clipped flags, left/right/source-y parameters and actual pooling mass. Bilinear RGB and nearest support sampling use the existing `align_corners=True` convention. No source truth mask/point gates ordinary inference. Ignore annotations can exclude training loss pixels after projection; they do not provide the forward support.

Guide predictions are an independent consistency check, not an automatically trusted replacement warp. Log point-versus-mask-row gaps, with the existing0.05width engineering flag; it is not a calibrated rejection probability. Invalid geometry remains an observable outcome, not a silently repaired atlas. Fully unsupported samples keep geometry outputs and skip texture loss with a reason.

The sampling boundary is `no_grad`: input/grid and result are detached. This avoids asking CUDA to differentiate `grid_sample`; [PyTorch2.11 documents a backward nondeterminism caveat](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.grid_sample.html). New texture gradients begin at its encoder. That boundary also prevents the network learning a convenient distorted warp to reduce its texture objective.

## Texture representation and prediction

| Module | Chosen specification |
|---|---|
| Tile inputs |4channels: patch-local9×9 normalized luminance, Sobel x/y/magnitude; std floor0.02, clip±3 then divide3 |
| Patch encoder |Four3×3 stride2 convolution/GN8/SiLU stages, channels4→24→32→64→96;4×4 output mean-pools to96 |
| Layout metadata |Linear projection of row/column normalized coordinates, coverage and pooling mass:4→96 |
| Context |Two residual axial blocks over8×4×96; depthwise3×3,5×1,1×3 spatial paths summed; GN8;1×1 96→192→96 with SiLU; learned residual scale initially0.1 |
| Global embedding |Geometric-support-weighted mean of96D full-view contextual tokens; raw and L2-normalized forms exposed |
| Projector |Linear96→128, SiLU, Linear128→64; unnormalized outputs for variance/covariance loss; no BatchNorm/dropout |
| Prediction decoder |Linear96→128, SiLU, Linear128→1024; reshape4×16×16 per masked tile |
| Masking |One supported tile per band where at least2eligible tiles exist;≤8masked tiles/view; at least2visible valid tiles overall |
| Target |Detached4×16×16 local-contrast target, from fixed4×4 mean aggregation of the corresponding64×64input tile |

Two photometric views share geometry and coordinates. Compute local contrast within each tile, so hidden tile pixels cannot leak through a cross-tile normalization kernel. The masked pass replaces selected encoder tokens with a learned mask token before context; hidden tile features are absent from that path. Full and masked context passes share weights and may reuse the patch-encoder outputs. There is no direct visible-to-target skip. Full-view embeddings feed the invariance loss; masked outputs feed prediction loss.

A source image is SSL-eligible only with at least4supported tiles of coverage≥.25 spanning at least2bands, at least2visible supported tiles and at least1masked tile with nonzero target-loss weight in each view. Otherwise retain its geometry outputs and explicit texture skip. Batch-level variance/covariance still require4eligible distinct images and2components. These checks prevent empty masked reductions being misreported as perfect predictions.

First run uses no Transformer, EMA teacher, negative-image bank or new external backbone. The custom axial branch is small and appropriate to32tiles. Parameter targets: trainable≤1M, total≤3M; exact counts and tensor shapes are mandatory implementation measurements before freeze, not claimed measurements today.

## Experimental error estimation

Quality head64→32→3 with SiLU and three sigmoid outputs predicts annotation-agreement errors, not probabilities of physical safety. Input64 =48pooled frozen features +16detached scalars: two mask areas, two mask entropies, two empty flags, two clipped fractions, three guide/mask gaps, three guide-gap availability flags, atlas coverage and valid-band fraction. Undefined gaps use a zero sentinel plus their associated availability flags; empty boxes are never given fabricated aspect ratios.

Targets: `(1-Dice_tyre,1-Dice_tread,human_point_MAE_width)`, all detached, with explicit target availability. Point-error targets use only the70human-labelled training images; weak point annotations do not become human calibration truth. Compare MAE/RMSE and fixed-bin predicted/observed error with a training-only constant-median predictor. If the head is uninformative, export its raw values as experimental/unvalidated; never market them as calibrated uncertainty.

## Objective and gradient rules

```text
L_geo = mean_region(masked_BCE +soft_Dice)
        +2*weighted_point_SmoothL1(beta=.02)
        +.1*containment +.1*eligible_guide_row_agreement
        +.1*anchor_drift

L_quality = masked_SmoothL1(predicted_error,detached_observed_error,beta=.02)

L_texture =25*invariance +25*(variance_A+variance_B)
           +(covariance_A+covariance_B) +.5*masked_prediction_Huber(beta=.1)

L_total = L_geo +.25*L_quality +min(1,epoch/5)*L_texture
```

Geometry drift is mean squared mask-probability change plus mean squared point-coordinate change against the same-view anchor on applicable pixels/points. Point reduction divides weighted loss sums by available weight, not by the full batch. Missing targets give zero contribution and zero denominators, logged as unavailable. Soft containment/row terms retain original eligibility rules; contradictory/ignored annotation cells do not enter reference supervision.

The [VICReg paper](https://arxiv.org/abs/2105.04906) motivates matching views while preserving variance and reducing covariance. Here invariance is mean squared difference of64D projections. Each variance term is mean `relu(1-sqrt(sample_variance+1e-4))`; sample variance uses N−1 across eligible distinct images in the current batch. Each covariance term sums squared off-diagonal covariance entries and divides by64. These25/25/1starting coefficients follow the source method, but this small tyre-cohort adaptation is unvalidated. Patches/paired views are never counted as independent tyres.

Variance/covariance terms are batch-level and appear once in batch_objectives, not duplicated as independently summable pair losses. Masked Huber divides the summed weighted element losses by the summed valid weights across all masked tiles/views and four channels; retain both sums. Any4×4target block touching a projected ignore pixel is excluded from that target loss. Compute variance/covariance only with≥4eligible distinct images from≥2conservative components. Otherwise set every texture gradient to None, skip that optimizer group, record the reason and still commit the geometry/error update. Log global and per-group update counters. Full FP32 training, no AMP/dropout, separate clip1.0 per group; frozen anchor gradient count must always be0.

## Explainability and classification boundary

Outputs explain geometry, support, observed masked-prediction residuals and reference similarity. A reconstruction residual is not a crack/damage label. Retrieval must show exact source references, coordinates, similarities and source/component exclusions; it does not yield a calibrated wear class. No mileage CE, ordinal loss or trained class head participates in the first pilot. A later classification readout would require its own declared protocol and cannot fix missing independent tyre support by itself.
''')
write('EXPERIMENT.md',r'''
# Data, experiment, selection and evaluation protocol

**Chosen design values; executable freeze pending.** One candidate, seed1,60epochs,49updates/epoch, batch8:2,940updates and23,520source draws. Two texture views per eligible draw; at most47,040training views. No automatic seed2/ablation/sweep. Disposable engineering seam updates are a separate namespace, never reported as measured fit.

## Immutable inputs and identity

| Item | Contract |
|---|---|
| Kaggle source |`shanmuk4622/tire-dataset-prepared-phase2`, expanded read-only input, no duplicate output copy |
| SHA256SUMS identity |`06c6c9d003aff808babf7ebb5786f441e95064eb000f5e5ed5b155d0ee9e7c6a` |
| Historical overlay |`f81a80d20b60ba505d94e31c26615e7e7171a91a8506948be6c392b68107a039` |
| Historical roles |386train /81validation /103already-inspected test; no reassignment |
| Conservative train groups |Low/mid/high1/1/4; confirmed original tyres1/3/4 |
| Point target support |Training70human /152weak /164none; validation/test25human images each |
| Label meaning |Mileage proxy retained as registry context; not a GEO optimization target |
| Base checkpoint |Pinned TREAD-ONE seed3/epoch20 SHA and URL in PLAN.json; no test substitution |

Sources and point counts follow the existing independently audited registries. The inherited anchor was selected historically using validation, and old held-out outputs have informed development. Therefore all new validation/test results remain development evidence. A two-tyre low/high endpoint cannot confirm full three-class unseen-tyre transfer.

## Deterministic full-coverage epoch plan

The new objective does not need class-balanced replacement draws. Cover every original training image exactly once per epoch, with six explicitly padded draws to reach392samples. This yields23,160primary draws and360padding draws over60epochs. Save the complete plan before each epoch; never invent an independent draw probability for this structured schedule.

1. Build queues from the six conservative components; sort IDs, then seed-shuffle by `(seed,epoch,component)`.
2. Identify the largest component (249images) and its complement (137images). Reserve49distinct images from each, supplying one from each side to each of49batches.
3. Seed-shuffle the remaining288images, append six padding IDs from the epoch permutation, and fill six remaining slots/batch.
4. Deterministically repair any within-batch duplicate by swapping available remaining slots while preserving the epoch multiset. Fail preflight if the required invariants cannot be met; never silently drop an image.
5. Record all draws, padding flags, component counts, augmentation/mask seeds and plan SHA. Validate386unique primary IDs,392draws,8distinct source IDs/batch and≥2components/batch.

The support constraint makes this packing feasible; actual implementation must test many seeded epochs. It does not produce eight independent tyres per batch. SSL-valid support can shrink after predicted masks; separate eligibility checks still apply.

## Views and warm start

Geometry receives one shared source photometric perturbation: brightness, contrast and gamma independently uniform[.9,1.1], applied in a frozen documented order to RGB[0,1]. Geometry targets/guide rows do not move. Its detached atlas then supplies two independent texture photometric perturbations with factors[.8,1.2]. Local contrast is computed after those factors, inside each tile. No elastic transforms, arbitrary geometric crops or label-changing flips in pilot1.

Training masks/points supervise geometry and provide error targets only; predicted masks generate the normal atlas. Masked tile choice uses predicted valid coverage≥.25. Keep at least one visible valid tile in every maskable band. Invalid/empty support yields logged texture skips, not fabricated texture embeddings.

Before the measured fit, run zero-update controls: exact anchor-versus-zero-adapter agreement, seeded random texture initialization, fixed training-only mean tile-target predictor, and train-only gradient-isolation checks. The mean predictor is computed once from a deterministic identity pass over386train images using the anchor atlas, stored by tile/channel/target pixel and valid counts. It supplies a loss baseline, not a learned semantic label or an additional optimized model.

## Optimizer and precision

One AdamW instance with disjoint parameter groups: geometry lr1e-4, texture lr1e-3, quality lr3e-4; wd1e-4, betas(.9,.999), eps1e-8, foreach/fused disabled. Cosine LR by global completed update, no warmup; texture objective ramps over first5epochs. Separate norm1.0 clipping for each group. A texture-skipped batch leaves all texture grads None so AdamW does not apply a weight-decay-only texture step. Save global and per-group counters/moments and the explicit scheduler convention.

All FP32. Frozen anchor eval/GroupNorm; no running BatchNorm, dropout, flash attention, SDPA or CUDA grid-sampling backward. Deterministic mode fails on unsupported operations. Local CPU/GTX checks and actual T4 checks are separate; [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.11/notes/randomness.html) does not promise CPU/GPU or cross-release equality.

## Validation and original checkpoint selection

Every epoch: all81validation images with fixed identity geometry inputs and the same seed-addressed pair of texture views/masks for comparability. Compute baseline anchor geometry on these identical images once before fitting and retain its exact scores. No test-forward checkpoint selection.

| Eligibility condition | Predeclared rule |
|---|---|
| Mean two-region Dice |≥anchor validation mean−.002absolute |
| Human-point MAE |≤anchor validation MAE+.001image widths, on identical available human targets |
| Invalid atlas fraction |≤anchor baseline+.01absolute |
| Projector collapse check |Median per-dimension std≥.1 and covariance effective rank≥4 across validation images; finite values required |

These are engineering non-regression/collapse tolerances, not statistically proven population margins or semantic quality guarantees. Effective rank is `exp(entropy(normalized nonnegative covariance eigenvalues))`, with zero-total covariance reported as rank0. Stats use one fixed view per image for rank; paired views are not independent samples. Empty/undefined denominators reject eligibility with a reason.

Among eligible epochs minimize `masked_Huber / frozen_training_mean_predictor_Huber +.1*paired_cosine_distance`, earliest tie. The mean predictor is evaluated against the identical validation targets/masks; denominators below1e-6 make the ratio unavailable and the epoch ineligible. Quality error estimates and mileage labels do not enter selection. If no epoch qualifies, selected checkpoint is unavailable and the candidate fails; retain final/all checkpoints and evidence. Do not silently export the anchor or a test-winning epoch as the trained winner.

## Endpoint evaluation and success

Retain selected and final identity passes on train386,validation81,historical test103, clearly separated. Initial anchor controls stay separately tagged. All human/weak/missing targets remain distinct. Full dense/geometry per-image metrics, boxes/missing boxes, point errors, atlas validity, paired consistency, masked-target loss against the mean predictor, representation rank, retrieval exclusions and error-estimation quality are saved.

Predeclared diagnostics on selected/final historical endpoint: brightness.8/1.2, gamma.8/1.2, grayscale, known synthetic blur, background-neutralized source, reverse bands/columns and top predicted-residual tile masking. Compare geometry changes and texture features under each actual intervention; keep source-patch mapping. Blur/appearance perturbations are synthetic robustness probes, not newly annotated capture defects or independent photos. Background intervention uses masks only in the explicitly marked diagnostic branch, never normal inference.

| Question | Result needed for a useful research candidate |
|---|---|
| Dense preservation |Selected validation floors pass; historical endpoint dense changes are reported, not used to reselect |
| Nontrivial texture prediction |≥5%lower validation masked loss than the frozen mean predictor under identical mask weights |
| Paired consistency |Mean cosine distance≤.05; report variance/rank and nuisance sensitivity together |
| Training support |≥90%training draws receive valid SSL loss, every training source has recorded coverage; otherwise failure/coverage limit recorded |
| Error head |Compare with constant training median; unvalidated scores stay explicitly experimental |
| Classification |No improvement claim from this pilot; no new class-head fitting |

The practical success claim is preserved/possibly refined visible geometry plus a noncollapsed, useful local texture representation on this development cohort. A classifier, depth estimator, defect detector, safety assessor and production application are separate unestablished claims.
''')
write('RETENTION.md',r'''
# Evidence retention, native tensors and storage budget

**Design contract; exact fixture byte measurements pending.** All scalar/tabular records are direct CSV. JSON carries manifests/configuration. Bulk tensors use direct uncompressed native files, never a ZIP/NPZ/database envelope. Native bulk is additional evidence, not a substitute for browsable per-image/loss/gradient CSV.

## IDs, types and loss reconstruction

`run` is the12-character measured protocol prefix; full SHA stays in CONTRACT.json. `sample` identifies one geometry/atlas source pass; `(sample,view)` identifies its two texture views. Sample IDs are monotonically allocated by committed units, not assumed contiguous within one stage. `image_id/source_sha256` join immutable source bytes; `epoch/draw` join the complete schedule; `parameter_update` is the already-applied state used for a pre-update observation. Geometry/error/texture optimizer-step counters are separate. `stage/endpoint` distinguish initial control, online train, validation, selected/final and interventions. `region=0/1` means tyre/tread, `guide=0..2`, `side=0/1` means left/right, `view=0/1` identifies the paired texture views, and row/column/patch indices are zero-based.

Use a single CSV null representation: empty field with an availability flag/count/reason. Absent human targets or unsupported boxes are not numerical zeros. Counts/IDs stay integers; counter seeds preserve full64-bit identity, with decimal-string JSON/UI representation where needed. Serialize boolean CSV fields as0/1. Scientific scalars accumulate in FP64 where appropriate. Raw FP32 vector/probability columns use9significant decimal digits with a required FP32 roundtrip test; derived FP64 metrics use17digits. Exact raw FP32 logits/tensors also remain native. The dense-pixel CSV omits a repeated run column and repeated logits to control size; its file manifest supplies run identity and the indexed native tensor supplies every original logit. The design JSON proposes every field/type; it is not yet an executed runtime schema.

Retain unreduced numerator/count statistics for BCE, weighted point loss, masked Huber, row agreement, drift and quality losses. Retain every64D projector output so each batch's invariance, sample variance and covariance objective can be independently recomputed. Save raw96D global embeddings plus normalized retrieval values/normalization rules. Loss accounting includes epoch ramp, per-group clipping, LR, skip flags and exact sample/component denominators.

## Bulk native records — all training observations

Closed, immutable uncompressed .npy shards contain at most8completed updates per tensor type and close early at epoch/evaluation boundaries. Active raw FP32 append streams have fixed block-shape metadata, per-unit byte/hash cursors and explicit CSV indexes. Closing writes a self-describing NPY file, verifies it, commits the new index/pointer, then releases the redundant active stream only after that atomic unit is durable. On a crash, only uncommitted suffixes or unselected close outputs are discarded. Each shard/index maps sample/view/patch to slice, shape, dtype, byte length and SHA. Variable masked tiles pad to8 with an availability mask, never fake loss observations. Both active-stream and NPY roundtrip/cursor cases must pass before freeze.

| Native content | Planned maximum for60epochs; excludes validation/endpoints |
|---|---:|
| Every training geometry logit:23,520×2×128×96 FP32 |2,312,110,080bytes |
| Both pre/post-context patch tokens:47,040×2×32×96 FP32 |1,156,055,040bytes |
| Every masked prediction and target:47,040×8×2×4×16×16 FP32 |3,082,813,440bytes |
| Masked-pixel weights:47,040×8×16×16 FP32 |385,351,680bytes |
| Canonical row parameters:23,520×512×3 FP32 plus validity |Approximately145MB plus row flags |

These are shape-based upper bounds, not measured file sizes. Ineligible views reduce actual counts and are explicitly logged. Together with global/projector CSV, this is substantially fuller raw evidence than earlier context runs. Every valid training mask logit, masked target/prediction and patch token is retained; original annotated source files remain pinned on Kaggle/HF rather than duplicated per draw.

Validation native chunks retain every epoch's81mask-logit maps, all points/quality outputs, atlas row parameters and paired token/prediction data. Selected/final identity outputs retain all570images, complete dense logits/probabilities, points, predicted support/row parameters, texture tensors and overlays; source-cohort roles remain distinct. Dense CSV exports cover both regions at128×96, up to28,016,640pixel rows across two570-image endpoints. If no eligible selected checkpoint exists, selected data stays unavailable instead of fabricating another endpoint.

Source RGB plus exact photometric parameters, original preprocessing code/version, predicted row-grid parameters and validity are sufficient to reconstruct the deterministic forward atlas. The full512×256×2coordinate grid and repeated RGB atlas for every training observation are not separately duplicated. Selected/final examples include rendered atlas images for inspection. All source grid conventions, clipping and intermediate precision must have a direct reconstruction check.

## Recovery state versus permanent history

At every complete update persist model/trainable state, all optimizer groups/moments and counters, scheduler, scaler-disabled state, Python/NumPy/CPU/CUDA RNGs, sampler/mask plans, cursor, partial epoch accumulators, native chunk positions, CSV byte/row/hash cursors, source/environment identities and current selection. Two recent local native recovery states rotate atomically. Preserve all60epoch model checkpoints, selected/final full exports, and every confirmed30-minute/major/stop remote snapshot. All40CSV families and native observation chunks survive rotation.

Every individual update's complete optimizer state is not kept permanently; its contemporaneous outputs/metrics/gradients, unit hash and the rolling recoverable state are retained. There is no claim of2940permanent full model/optimizer checkpoint copies. All epoch weights and published snapshots remain available. Do not use a hidden local database as the sole record of any requested metric.

## Output capacity and publication staging

| Category | Planning allowance, to measure before training |
|---|---|
| Full training native observations |About7.1GB including weights/row flags; shape math above |
| Direct embeddings/scalars/gradients/points/support CSV |About0.7–1.5GB depending actual row width |
| Selected/final dense CSV |About1.5–3GB; exact ASCII size must be measured |
| Validation/endpoint native chunks |About0.8–1.5GB; fixture-derived projection required |
|60epoch weights, two rotating states, sources/anchor |Target≤0.7GB; exact counts pending |
| Figures/interactive examples/indexes |Target≤0.3GB; bounded gallery, complete numeric records |
| Logical artifact ceiling |15,000,000,000bytes |
| Projected working peak ceiling |18,000,000,000bytes, preserving2,000,000,000free bytes within a20GB assumption |

The ranges may exceed the logical ceiling when pessimistic ends are combined. This is an explicit preflight risk, not a claim that the design already fits. Measure real CSV/native writes from a representative batch and complete validation fixture; include a1.2uncertainty factor and active staging. If projected logical or peak exceeds the ceiling, fail before measured updates. Revise the retention contract/implementation before freeze or improve exact-format/chunk encoding. Never silently discard requested telemetry during the measured fit.

Closed epoch chunks are immutable and can be hard-linked into same-filesystem snapshots after an actual hard-link/immutability test. Active CSV/native prefixes must be copied to their committed boundary; never hard-link a file still being appended. Rotating recovery states use replace-by-new-inode, not in-place overwrite. If hard links are unsupported, the full-copy peak must fit the same limit or preflight fails. Do not depend on the reported1TBscratch allowance; measure it and use it only as optional cache. Confirmed remote staging can be removed after hash verification without deleting science or active state.

## CSV dictionary and cadence

[SCHEMAS.json](SCHEMAS.json) contains all40draft families, every field, key, type, units and missingness. The inventory below is generated from it, so the documentation and machine proposal agree. A typed executable schema and validators must be frozen with the eventual runtime FILES.json.
''')
with (OUT/'RETENTION.md').open('a',encoding='utf-8') as f:
    f.write('\n| Family | Key | Cadence | Fields |\n|---|---|---|---:|\n')
    for name,s in schemas.items():f.write(f'| `{name}` | `{", ".join(s["key"])}` | {s["cadence"]} | {len(s["columns"])} |\n')
    f.write('\n## Every declared column\n\n')
    for name,s in schemas.items():f.write('### '+name+'\n\n'+', '.join('`'+c['name']+'`' for c in s['columns'])+'.\n\n')
    f.write('## Explicit unavailable measurements\n\nNo new independent tyres, physical tread depth, safety truth, crack/damage annotations, causal wear explanations or full three-class independent confirmation are created. Full grids/RGB are reconstructible rather than duplicated per observation. No per-update permanent optimizer archive or new classification accuracy result is claimed. Exact wall time/storage/performance remain pending gates. Any further omission discovered during implementation must be recorded before the measured protocol is frozen.\n')
write('OPERATIONS.md',r'''
# Kaggle operation, recovery and HF transport

**Planned behavior, not a runnable notebook yet.** Preserve local `cv_conda`; use `phase3/run_cv.cmd` for development. Do not replace CUDA Torch or the existing workstation dependencies.

## Two T4s, one measured model

| Device / component | Actual intended ownership |
|---|---|
|GPU1 |Frozen TREAD-ONE feature/anchor inference with geometry microbatch1; checkpoint-matched validation/evaluation at barriers |
|GPU0 |Trainable geometry adapters, texture encoder/context/decoder/projector and error head; batch8texture losses |
|CPU coordinator |Seeded epoch plan, source verification, serialization, CSV/native writes, one HF writer and clear status |

The frozen48×128×96features, anchor logits/raw points and minimal metadata move explicitly fromGPU1toGPU0. They are detached. GPU1's anchor is immutable; it never performs optimization. At validation barriers, copy the exact completed trainable state to the evaluation modules onGPU1 and record its SHA/update; optimizer training pauses until the declared evaluation unit is committed. This is one candidate using two GPUs, not two seeds or DDP. Initial synchronous ownership is favored; pipeline overlap may be added only before freeze with a matched seam test. Never claim both GPUs continuously train.

## Planned notebooks and settings

| Notebook | Purpose | User settings |
|---|---|---|
|NB16 GEO preparation/contract gate |Source/input checks, disposable train-only optimizer/recovery seam, actual devices/math/memory/storage/HF gate; no measured candidate fit |T4×2, Internet ON, existing expanded Kaggle input, HF_TOKEN |
|NB17 GEO resumable pilot |One measured seed/60epochs under the passing frozen contract |Same exact settings/source/environment; Run All; automatic compatible resume |
|NB18 completed review/export |Independent retained-data audit, figures and model-card/capability export after NB17 completion |CPU or GPU only if an explicitly declared extra saved-state verification requires it; no automatic retraining |

These names/numbers are planned reservations. No files with these notebook names are delivered today. The measured notebook must embed exact sources/config/schema, not rely on the repository being cloned or previous session variables. Dataset auto-discovery must find exactly one expanded input with the pinned checksum. It must not download an image archive from HF when the Kaggle input is attached.

NB17 cell order:1bootstrap/private tested HF SDK/source extraction;2dataset/anchor/source/environment/space checks and verified recovery;3actual live gate if no compatible saved gate exists;4single training/validation loop;5selected/final evaluation and final publication;6report/interactive links/receipt. Every major completed cell publishes its relevant direct evidence. Run All is the normal instruction; do not ask the user to paste helper code or retain an invisible Python variable across fresh sessions.

## Complete-update persistence

```mermaid
stateDiagram-v2
    [*] --> VerifySources
    VerifySources --> RestoreVerifiedHF: compatible incomplete manifest
    VerifySources --> Initialize: no active fit
    RestoreVerifiedHF --> ValidateCursor
    Initialize --> LiveGate
    LiveGate --> Update: actual checks pass
    ValidateCursor --> Update
    Update --> Commit: all optimizer groups and logs complete
    Commit --> Update: next planned draw
    Commit --> Validate: epoch boundary
    Validate --> Commit
    Commit --> Publish:30min / major / catchable stop
    Publish --> Update: verified snapshot
    Publish --> Completed: endpoints and final receipt
    Publish --> Stopped: preserved state / explicit stop
    Stopped --> VerifySources: rerun unchanged
```

A unit includes all geometry/texture/error steps, scheduling, CSV rows/native offsets and the next cursor. Save an atomic pointer only after all associated files flush and hash. If interruption occurs inside the unit, restore the preceding native state, truncate only uncommitted append tails and replay the same seeded unit. Never keep a partially applied geometry update with an unrecorded texture update. All three group moment/step counters must be restored, including skipped texture updates.

Fresh session: verify manifest, source/runtime/environment/data/anchor hashes and owner lease; download required immutable files to unique partial paths; verify SHA/size; install one internally consistent state; compare native/CSV/chunk prefixes; resume the next complete update. Reuse local/saved Kaggle files only when their verified ledger extends the same history. Publish ahead local committed progress before doing further fitting. Completed compatible run → display receipt and exit without another optimizer step.

Strict replay binds actual numeric environment/devices. A runtime change is not silently ignored. [PyTorch2.11 reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html) motivates exact same-environment state tests; CPU/T4 equality is not promised. Original workstations and historical sources are outside all mutation paths.

## HF writes and quotas

Account/repo: `Shanmuk4622/tyre-wear-study`. Token: authorized Kaggle Secret `HF_TOKEN` only. Private tested HF SDK directory, publishing subprocess only; no global package downgrade. Worker tensors/training code do not receive the token. Namespace: `phase3/tread-two/geo-r1/{engineering,fit,review}/<contract>`, distinct from every ATLAS/TREAD-ONE namespace.

Reuse the tested immutable-object/CAS owner transport as a separately copied, hash-pinned module; do not edit completed copies. One owner, explicit heartbeat/expiry, at most80scientific objects/batch, final selecting manifest after object checks. Existing verified object receipts skip redundant work; lost successful commit responses reconcile their marker instead of creating duplicate writes. Do not erase a live lease or fork another fit to bypass recovery.

Normal push interval1,800seconds; additionally after each major stage and immediately attempted at a catchable stop/error once the current complete unit is sealed. Server backoff may delay transfer. The existing conservative96commit-attempt/hour guard is below the user's128/hour concern; actual API/resolver/action policy headers govern waiting. The [HF documentation](https://huggingface.co/docs/hub/en/rate-limits) distinguishes request buckets and granular actions. Do not assume128/hour is a universal resolver limit or reset the budget to evade a wait.

Catch SIGINT/SIGTERM/KeyboardInterrupt: request a safe boundary, flush native/CSV state, attempt verified publication, write an explicit pending receipt if transfer fails, and exit clearly. Forced kernel death/network loss cannot execute an upload; recover from surviving local output or the latest verified HF snapshot. Preserve all source files and the failure transcript. This is strong resumability with stated boundaries, not an impossible guarantee that every crash uploads unsent bytes.

## Status and troubleshooting built into delivery

Print update/epoch/next draw and current phase at least every30seconds while waiting. Distinguish feature inference, texture backward, native writes, prefix verification, quota wait, upload and validation. Enable periodic stack diagnostics for long blocking paths without printing tokens/headers containing credentials. No silent2GBobject stream: bounded verified reads with Retry-After/RateLimit handling.

| Trigger | Required behavior |
|---|---|
|Wrong source/input/anchor/environment |Fail before measured update; show expected identity and safe observed metadata |
|Missing local directory |Restore full compatible HF state; no old POINTER-only requirement |
|Disk projection exceeds limit |Fail preflight; revise format/schema before freeze; never silently shrink measured retention |
|In-run free reserve breached |Commit safe state, attempt publication, stop explicitly; keep evidence |
|Unsupported deterministic CUDA op |Fail fixture; repair/test new source before freeze; do not switch determinism off silently |
|NaN/Inf |Do not commit invalid update; retain diagnostic event/native last-good cursor; explicit bounded retry policy frozen before fit |
|Invalid atlas / too few SSL images |Log sample/batch skip, keep geometry loss, preserve group counters; monitor coverage gate |
|HF429/backoff |Visible waiting with retained state/owner heartbeat; no fake training failure or reset of quotas |
|No eligible checkpoint |Keep final/all epoch states and rejection reasons; no test-selected replacement |

Actual new throughput, GPU memory, transfer overhead, native chunk peak and upload duration remain measurements for NB16. Never give the user an untested hour estimate as a promise.
''')
write('IMPLEMENTATION.md',r'''
# Implementation sequence and acceptance matrix

**User asked for the complete plan. Model code/notebooks and measured training remain pending.** This file fixes what will be built and in which order. Engineering gates are automatic checks in implementation; they are not additional user permission requests.

## Planned isolated directory

```text
phase3/tread_two/geometry_first/
  plan/                  # delivered design documents and JSON proposals
  runtime/               # new model/data/objective/coordinator modules; pending
  validation/            # fixtures, receipts and acceptance; pending
  reports/               # actual future derived science; pending
phase3/notebooks/
  phase3_NB16_TREAD_TWO_GEO_Preparation_And_Contract_Gate.ipynb   # planned
  phase3_NB17_TREAD_TWO_GEO_Resumable_Pilot.ipynb                # planned
  phase3_NB18_TREAD_TWO_GEO_Completed_Review.ipynb               # conditional plan
```

| Order | Planned source owner | Deliverable / condition |
|---|---|---|
|1 |`data_geo.py`, `schema_geo.py` |Pinned split/annotation adapter, typed40-family schema, exact field reductions |
|2 |`anchor_geo.py` |Strict a5b8… checkpoint loader, original features/raw points exposed, zero-update equivalence |
|3 |`model_geo.py`, `atlas_geo.py` |Bounded adapters, quality head, detached atlas, encoder/context/masked head; shapes/parameter inventory |
|4 |`objective_geo.py`, `schedule_geo.py` |Loss sufficient statistics, separate gradients/clips, full-coverage deterministic392draw plan |
|5 |`store_geo.py`, `telemetry_geo.py` |Atomic units, native chunks/indexes, append-prefix and schema verification |
|6 |`worker_geo.py`, `evaluate_geo.py` |Two-GPU ownership, optimizer/evaluation barriers, selection and predeclared controls |
|7 |`hub_geo.py`, `coordinator_geo.py` |Versioned transport copy/private SDK, source/runtime binding, visible stop/resume/rate behavior |
|8 |Notebook builder +NB16 |Self-contained actual Kaggle preparation/contract gate and disposable seam |
|9 |NB17 builder |One measured .ipynb only after local and actual live checks pass |
|10 |Review/export +NB18 |Raw-file audit, every loss/geometry/latent calculation, interactive atlas and model-card status |

Keep new module names distinct from completed `pilot/`, `lc_pilot/`, `context_pilot/`, their reviews and both workstation trees. A future FILES.json freezes only the new runtime/config/schema and exact copied source identities. PLAN.json remains the design record, not the numeric measured contract.

## Required tests before measured fitting

| Area | Meaningful acceptance test |
|---|---|
|Data |386/81/103roles, hashes, original ignore policy, human70/weak152/none164train point support, component consistency |
|Schedule |Many seeded epochs: all386primary IDs once,6padding,49×8, within-batch uniqueness and≥2components, identical replay |
|Warm start |Zero adapters reproduce anchor masks/points/boxes/features under matched CPU and matchedT4math paths |
|Gradient isolation |Texture-only and error-only backward give zero geometry/anchor grads; geometry-only gives zero texture grads; group clipping independence |
|Point/mask math |Ordered bounds; correct no-target reduction; independent overlap labels; ignored cells excluded; reference eligibility preserved |
|Masking leakage |Change hidden tile pixels after encoding replacement: masked path's visible tokens/inputs unchanged; patch-local LC prevents boundary leakage |
|SSL |Constant-vector fixture triggers variance/collapse checks; covariance matches manual calculation; N<4/one-component skip preserves texture optimizer state |
|Loss accounting |Recompute every term from CSV/native arrays including ramp, covariance denominator, padding/mask availability and group counts |
|Error head |No weak-point target used as human error truth; targets detached; compare training-median baseline without new calibration promises |
|Atlas |Reconstruct exact grid from retained row parameters; empty/clipped/misaligned rows logged; no input/grid gradient path |
|Resume |Same-environment update1→4 direct versus update1save/restore→4: parameters/moments/counters/RNG/CSV/native outputs equal except declared timing fields |
|Crash points |Before/after optimizer groups, CSV append, native flush, pointer swap, upload success-with-lost-response and evaluation chunk boundary |
|Storage |Actual chunk sizes, float precision roundtrip, active prefix copies, closed-file hard-link immutability, fallback-copy peak,≥2GBreserve |
|HF |Live owned throwaway namespace, actual restore/continue, no oldnamespace writes, idempotent completed rerun and clear quota wait |
|Notebook |Fresh extracted setup in empty Kaggle-style directory, no hidden old variables,40-family source payload matches readable source |
|Preservation |All completed notebooks/catalogs, original103Markdown files and both workstation source hashes unchanged |

Disposable fixtures use declared training-only images and a separate namespace. Their updates/time/metrics are tagged `engineering_fixture`, excluded from the measured fit and never selected as the final model. Actual dual-T4 proof is required, not inferred from a localGTX1650test.

## Freeze checklist and measured-run start

Before NB17 is delivered: executable dtypes/null handling and numerical reductions complete; all sources/checkpoint/input/overlay hashes bound; exact parameter counts measured; schema/native indexes and raw capacity projection pass; actual deterministic resume seam and liveHF gate pass; complete training schedule/augmentation/mask seeding policy fixed; validation thresholds/selection/diagnostics fixed; Notebook instructions/stop/error recovery tested; all retention omissions explicit.

No source, objective, loss coefficient, sampler or logging family changes mid-fit. A necessary repair must preserve numeric intent/history or become a separately identified protocol; never update FILES.json just to bypass a source mismatch. Compatible completed run exits/display-only. New measured run creates initial checkpoint/source/environment/coverage/reference-mean records and publishes them before fitting.

## Notebook handoff and later review

User instructions must say acceleratorT4×2, Internet ON, exact expanded Kaggle input, authorized HF_TOKENSecret, measured free-space requirement, Run All order, expected phase messages, local/native/remote outputs, restart route and known failure actions. Return a compact self-contained .ipynb plus readable source/module docs. No training-script-only handoff.

After user execution, inspect the supplied executed notebook without regenerating it; preserve exact SHA/output; verify pinned public publication bytes/native/source/CSV prefixes; independently recompute geometry, quality, every SSL objective and collapse/selection rule; inspect figures/interactive images; update project trackers/model card and pass/fail capability flags. A failed research hypothesis is retained honestly and does not trigger another automatic fit.

## What happens immediately after this plan

Implement modules1–5 and local fixtures first, then build NB16. This is the concrete next engineering milestone. NB17 is not yet runnable and no GPU fit should be started from this design-only folder. Workstation/application changes remain deferred. The existing completed ATLAS research hub stays intact and is contextual evidence for the new objective.
''')
write('RISKS_AND_DECISIONS.md',r'''
# Evidence, design decisions and risks

## Observation → choice → test

| Existing project evidence | Chosen response | What must be measured |
|---|---|---|
|All v1/LC/context selected/final low recalls0/62 |Remove mileage CE from this first new pilot; target geometry/texture/error objectives |Masked prediction/noncollapse and dense preservation; no classification improvement claim |
|Frozen LC already correct on47,040sampled train views; context sharpened confidence |Use self-supervised target prediction and representation diagnostics instead of another confidence-maximizing classifier |Mean-predictor comparison, invariant features with retained variance/rank |
|Shared-gradient projection frequently acted and selected point error worsened |Frozen backbone, disjoint adapter/texture/error learning paths; separate clips |Gradient and optimizer-routing tests; actual geometry floors |
|Strong existing visible geometry and exact project checkpoint available |Warm-start the selected seed3/epoch20 and train bounded small refinements |Zero-start equivalence and matched baseline geometry on identical views |
|Source/clip parent uncertainty remains |Preserve NB07 co-assignment and historical roles |Source hashes/schedule/exclusion checks; no new independent support claim |
|Prior quiet stages looked frozen; one recovery required missing-file repair |Self-contained bootstrap, visible phase status, full verifiedHF restore |Actual fresh-directory restart and lost-response/stop tests |
|User requires much richer evidence in directCSV |All40families plus every train geometry logit/token/masked target/prediction native chunk |Exact storage projection and independent sufficient-statistic/native loss reconstruction |

Sources: [NB05 results](../../../phase3NB05Completion.md), [NB15 completion](../../../phase3NB15Completion.md), [research closure](../../../phase3TreadTwoResearchClosure.md), [source hashes/primary references](SOURCES.json). These observations motivate a new hypothesis; they do not prove it will succeed.

## Failure register

| Risk | Prevention / observation | Decision if observed |
|---|---|---|
|Texture collapse or coordinate-only shortcut |Global across-image variance/rank, masked-loss mean baseline, fixed input shuffle controls, no hidden token leakage |Fail eligibility or report insufficient texture representation; no appearance-stability-only success claim |
|Tiny correlated batch makes covariance noisy |N≥4distinct images/≥2components; no patch-as-subject counting; coverage/stats retained |Skip invalid texture steps; fail useful-coverage gate if excessive; no mid-fit batch-rule change |
|Geometry adapters overfit or atlas drifts |Small zero-start bounds, anchor drift, no texture gradients into geometry, validation floors |Reject checkpoint; retain exact failed evidence and final state |
|Mask refinement learns annotation noise |Preserve ignore map/native overlapping labels, raw human/weak point provenance |Show affected groups/targets and avoid hidden annotation repairs |
|Error head predicts a constant low error |Training-median baseline, error bins, MAE/RMSE, separate human support |Keep experimental/unvalidated label; no reliability or safety score claim |
|Same-image invariance learns tyre identity |No population/physical-wear claim; per-source/component probes and cross-component exclusions |Report limited representation utility; do not infer new-tyre recognition |
|Masked target is solved by local average |Frozen mean-target baseline plus≥5%gain requirement; hidden pixels absent from masked path |No useful masked-prediction claim if baseline not beaten |
|Raw retention exceeds20GBarea |Immutable chunking, typed CSV projection, measured staging/hard-link test,15GBlogical/18GBpeak caps |Fail before measured fit; revise explicit source/schema before freezing |
|Cross-device math or unsupported op breaks replay |FP32, explicit boundaries, identical-math local/T4seams, strictactualenvironment binding |Repair fixtures before fit; never silently relax determinism |
|HF throttling or forced kernel death |30min/major/stop publication, durable local units, budget/backoff/status, explicit pending receipt |Resume last surviving verified state; unsent destroyed files cannot be fabricated |
|Repeated historical validation shapes model design |Explicit development scope, freeze new selection before fitting, no test reselection |No blind/generalization or formal noninferiority claim |

## Fixed now versus implementation measurements

Fixed design: one candidate/seed/budget, dataset/anchor, branch structures and disjoint gradients, losses/coefficient starting values, structured epoch schedule, no class loss, proposed selection/failure rules,40requiredCSVfamilies, raw tensor scope, new namespace and notebook stages.

Pending measurements with explicit acceptance: exact instantiated parameter count/shapes, actual forward/backward memory/time, live output/native CSV byte projection and staging capacity, deterministicT4seam, actualHFrestore/quota behavior, reconstruction loss baseline numeric values, baseline validation geometry, and all model-quality results. These are tasks in the plan, not missing user preferences or a reason to invent performance numbers.

No new user permission checkpoint is introduced by these engineering tests. They automatically determine whether the already planned implementation is internally valid. This turn delivers planning; it does not claim implementation or initiate a measured fit. Future runtime refinements before first fit remain possible if all changed choices are documented and revalidated before the measured contract is frozen.
''')
master=r'''
# TREAD-TWO / GEO — complete geometry-first development plan

**1 November 2026. Full design plan prepared; implementation and measured training pending.** This is a new TREAD-TWO objective built around segmentation, guide geometry, aligned texture learning and traceable evidence. The completed ATLAS v1/LC/context classifier experiments remain preserved as research evidence. Only existing photos are used; workstation changes remain deferred.

## What we will build

One custom packaged model accepts a512×384tyre photograph and returns tyre/tread masks, mask-derived localization boxes, six guide points, a512×256visible-tread atlas, local/global texture embeddings, masked-texture prediction diagnostics and three experimental annotation-error estimates. The first pilot has **no mileage-classification loss**. Later classification probing is a separately declared question, so another confidence increase is not presented as useful tyre recognition.

The creative component is a geometry-guided masked texture learner: use the existing visible geometry to establish32comparable local tiles, predict hidden texture from its neighbors, and learn stable but noncollapsed paired representations. Small geometry-refinement heads train with the masks/points already available. Disjoint gradients prevent texture/error objectives altering the dense branch. This combines useful existing supervision with a genuinely different learning objective; improvement is not yet measured.

```mermaid
flowchart LR
    A[Existing tyre photo] --> B[Frozen project encoder + bounded geometry refinement]
    B --> C[Masks / boxes / guide points]
    C --> D[Detached aligned tread atlas]
    D --> E[Two texture views + hidden tiles]
    E --> F[Custom axial encoder\nself-supervised representation]
    C --> G[Experimental observed-error estimates]
    F --> H[Patch evidence / reconstruction / reference similarities]
    G --> H
```

## Why this follows our results

| Observed fact | Consequence in the new plan |
|---|---|
|All three ATLAS candidates retain low recall0/62 |Do not launch another mileage-CE capacity revision as the main experiment |
|Context mainly lowers loss on already-correct sampled training views |Test masked-target prediction/representation utility with explicit collapse and mean-baseline checks |
|Existing geometry is useful; shared-gradient revision worsened point error |Keep the exact selected anchor frozen, train small supervised residuals, isolate task gradients and clip groups separately |
|Only1/1/4conservative training components across low/mid/high |Keep claims at existing-cohort development scope; preserve identity constraints and require no new photography |
|User wants one practical, highly retained run |One seed/60epochs; same Kaggle/HF workflow with more raw outputs and directCSV |

Evidence and limitations: [NB05](phase3NB05Completion.md), [NB15](phase3NB15Completion.md), [completed research handoff](phase3TreadTwoResearchClosure.md). Geometry labels can support class-agnostic learning even though a complete independent three-class component holdout is unsupported. This does not eliminate identity confounding or turn the historical endpoint into a blind test.

## Chosen experiment

| Item | Decision |
|---|---|
|Data |Existing570images;386train /81validation /103historical test; exact checksums/grouping retained |
|Point supervision |Train70human /152weak /164none; weak point weight0.25; error targets human-only |
|Anchor |TREAD-ONE r1 seed3/epoch20, original validation selection, SHA`a5b872ed…`;1,372,159frozen parameters |
|New trainable parts |Bounded mask/point adapters,96Dtexture encoder and2axial blocks,64Dprojector,masked4×16×16decoder,3error outputs |
|Capacity target |≤1Mnew trainable /≤3Mtotal parameters; exact counts pending implementation |
|Fit budget |One seed1,60epochs,49×8draws/epoch;2,940updates; every386train source each epoch plus6loggedpaddingdraws |
|Precision / learning |FP32; AdamW separate groups/clips; masked prediction +invariance/variance/covariance; supervised geometry/error losses |
|Checkpoint selection |Geometry preservation/collapse eligibility, then lowest masked-loss ratio+.1paired distance; earliest tie; no test selection |
|Retention |40draftCSVfamilies plus full train logits/tokens/masked predictions/targets; all60epochweights and complete rolling recovery |
|Kaggle / HF |TwoT4s, existing expanded Kaggle input, HF_TOKEN, direct immutable files,30min/major/catchable-stop pushes |
|Space |15GBlogical /18GBprojected peak /2GBreserve within20GBassumption; actual fixture projection is mandatory |
|Application |Both workstations unchanged; this phase creates research code/evidence and a later export contract |

## Deliverables and dependency order

| Stage | Concrete deliverable | Completion evidence |
|---|---|---|
|A: data and source contract |Typed schema, full-coverage schedule, pinned anchor/data, exact annotation availability |Hash/identity/packing and warm-start tests |
|B: local model and persistence |Geometry/texture/error branches, losses, raw native chunks, atomic store, replay tests |Shapes/gradients/accounting/disk tests; no measured accuracy claim |
|C: plannedNB16 |Self-contained Kaggle preparation/contract gate |Actual dualT4math/memory/storage/liveHFresume; disposable engineering namespace only |
|D: plannedNB17 |One resumable measured GEO pilot |All60epochs or honest failed/incomplete state; direct telemetry/native checkpoints |
|E: plannedNB18 |Completed-run audit, interactive atlas, model card and capability export |Pinned public verification, independent scientific reconstruction and declared result |

The immediate engineering milestone is **implement stagesA/B and buildNB16**. These notebook files are planned; none is runnable yet. Existing NB08–NB15 are not regenerated or repeated. An engineering gate is an automatic validity check, not a new permission request. No arbitrary extra full seed/sweep starts if the hypothesis fails.

## What success will mean

Preserve validation dense geometry within declared limits; beat the frozen train-only mean masked-target predictor by at least5%on validation; retain finite noncollapsed features and mean paired cosine distance≤.05; obtain validSSLloss for≥90%trainingdraws; report the experimental error head against a simple fixed baseline. Evaluate selected/final historical outputs only after original selection and retain every failure/group/availability count.

These thresholds are frozen engineering criteria for this development study, not guarantees of deployment or physical tyre condition. If no checkpoint passes, the candidate fails and its complete evidence is retained. There is no automatic fallback to an attractive historical-test checkpoint. Classification, crack/damage detection, physical depth and safety remain unestablished.

## Full technical documents

| Document | Detail |
|---|---|
| [Architecture](tread_two/geometry_first/plan/ARCHITECTURE.md) |Exact tensor shapes, bounded refinement, masking, error outputs, loss formulas and gradient isolation |
| [Experiment](tread_two/geometry_first/plan/EXPERIMENT.md) |Data/schedule/augmentation, baseline controls, optimizer, selection, endpoints and diagnostics |
| [Retention](tread_two/geometry_first/plan/RETENTION.md) |Every requiredCSVfield/cadence, native arrays, raw capacity math, recovery versus permanent history and declared gaps |
| [Operations](tread_two/geometry_first/plan/OPERATIONS.md) |Two-GPU ownership, Kaggle cells, crash boundaries, HF cadence/budgets, fresh restore and troubleshooting |
| [Implementation](tread_two/geometry_first/plan/IMPLEMENTATION.md) |Module ownership, dependency order, local/live acceptance matrix, source freeze and next notebooks |
| [Risks and decisions](tread_two/geometry_first/plan/RISKS_AND_DECISIONS.md) |Observation-to-design mapping, failure actions and pending measured engineering facts |
| [Machine plan](tread_two/geometry_first/plan/PLAN.json), [schema](tread_two/geometry_first/plan/SCHEMAS.json), [sources](tread_two/geometry_first/plan/SOURCES.json) |Exact chosen values, draft field/type proposal and pinned evidence/primary references |

The self-supervised objective adapts [VICReg](https://arxiv.org/abs/2105.04906); this small-cohort design has no measured improvement yet. The new detached sampling boundary follows the project's previous deterministic-operation failures and [PyTorch's CUDA backward caveat](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.grid_sample.html). [HF bucket/action guidance](https://huggingface.co/docs/hub/en/rate-limits) informs visible server-aware backoff; all user cadence, stop and direct-record requirements remain in force.

[Planning validation](tread_two/geometry_first/validation/PLAN_VALIDATION.json) records480schedule feasibility cases, raw CSV precision examples, dense-count reconstruction, metadata consistency and preserved sources. This is design validation, not a tested model or passed Kaggle gate.
'''
(ROOT/'phase3TreadTwoGeometryPlan.md').write_text(master.strip()+'\n',encoding='utf-8')
# Arithmetic design evidence, not a measured resource benchmark.
ntrain=23520;nval=81*60;nend=570*2;nsample=ntrain+nval+nend;nview=2*nsample
native={
 'all_geometry_logits_FP32':nsample*2*128*96*4,
 'pre_and_post_context_tokens_FP32':nview*2*32*96*4,
 'masked_predictions_and_targets_FP32':nview*8*2*4*16*16*4,
 'masked_loss_weights_FP32':nview*8*16*16*4,
 'atlas_row_parameters_FP32':nsample*512*3*4}
provisional=dict(global_and_projected_CSV=nview*(96+64)*40,endpoint_dense_CSV=nend*2*128*96*42,other_CSV_and_native_flags=500000000,epoch_checkpoints_and_rolling_states=600000000,galleries=150000000,source_and_SDK=100000000,intervention_outputs=400000000)
logical=sum(native.values())+sum(provisional.values());safe=logical*1.2;staging=384*1024**2+2*32*1024**2+64*1024**2
js('BUDGET.json',dict(status='arithmetic_design_projection_not_live_measurement',sample_upper_bound=nsample,texture_view_upper_bound=nview,native_exact_shape_bytes=native,provisional_allowance_bytes=provisional,provisional_dense_row_bytes=42,provisional_vector_row_bytes=40,unadjusted_logical_bytes=logical,uncertainty_factor=1.2,projected_logical_bytes=safe,projected_same_filesystem_peak_bytes=safe+staging,logical_limit_bytes=15000000000,peak_limit_bytes=18000000000,free_reserve_bytes=2000000000,assumed_output_bytes=20000000000,requires_live_fixture=True,full_copy_staging_may_fail=True))
with (OUT/'RETENTION.md').open('a',encoding='utf-8') as f:
    f.write('\n## Arithmetic planning check\n\n[BUDGET.json](BUDGET.json) records all formulas and provisional row-width allowances. Across train/validation/selected+final identity stages, the maximum is29,520source passes and59,040texture views before optional controls/probes. Native arrays contribute '+f'{sum(native.values()):,}'+'bytes; provisional other outputs give '+f'{logical:,.0f}'+'bytes logical before a1.2factor. The adjusted estimate is '+f'{safe:,.0f}'+'bytes and same-filesystem staging projection '+f'{safe+staging:,.0f}'+'bytes. These pass the proposed15GB/18GB ceilings arithmetically, but the40/42byte CSV row-width allowances are unmeasured and must be replaced by actual fixture writes. A full-copy staging path can fail even when this same-filesystem plan fits.\n')
# Format prose while preserving code, formulas, hashes and link destinations.
protected=re.compile(r'(```[\s\S]*?```|`[^`]*`|https?://[^\s)]+|\]\([^)]+\))')
def prose(segment):
    segment=re.sub(r'(\d+(?:\.\d+)?)(GB|MB|KB|GiB|MiB|D|M)(?=[A-Za-z])',r'\1\2 ',segment)
    segment=re.sub(r'\b([A-Za-z][a-z]{1,})(\d)',r'\1 \2',segment)
    segment=re.sub(r'(?<=\d)(?=[A-Za-z]{2,})',' ',segment)
    segment=re.sub(r'\b(a|A|all|All|the|The)(?=\d)',r'\1 ',segment)
    segment=re.sub(r';(?=\S)','; ',segment)
    fixes={'directCSV':'direct CSV','draftCSVfamilies':'draft CSV families','epochweights':'epoch weights','loggedpaddingdraws':'logged padding draws','TwoT4s':'Two T4s','dualT4':'dual T4','liveHF':'live HF','HFresume':'HF resume','HFrestore':'HF restore','T4seam':'T4 seam','stagesA/B':'stages A/B','plannedNB':'planned NB','buildNB':'build NB','all40':'all 40','all60':'all 60','runtime FILES':'runtime FILES','privateSDK':'private SDK','oneclass':'one class'}
    for a,b in fixes.items():segment=segment.replace(a,b)
    return segment
for path in [*OUT.glob('*.md'),ROOT/'phase3TreadTwoGeometryPlan.md']:
    parts=protected.split(path.read_text(encoding='utf-8'))
    path.write_text(''.join(s if i%2 else prose(s) for i,s in enumerate(parts)),encoding='utf-8')
print(json.dumps(dict(status='design_written_not_implemented',documents=8,CSV_families=len(schemas),declared_fields=sum(len(v['columns']) for v in schemas.values()),next_deliverable='NB16 engineering and contract gate after isolated implementation'),indent=2))
