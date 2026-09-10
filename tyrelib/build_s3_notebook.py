"""Build only NB11, preserving every executed notebook."""
from pathlib import Path
import hashlib
import build_notebooks as b
import build_closure_notebooks as c

CONFIG = '''# First run PREPARE on CPU. RUN needs your independent inputs and one T4.
MODE = "PREPARE"  # change to "RUN" after completing the image-only input package
INPUT_ROOT = Path("/kaggle/input/s3-independent-inputs")
# Required only in RUN: affirm the actual procedure, not merely to bypass a check.
BOXES_DRAWN_FROM_IMAGES_ONLY = False
BLIND_PASS_WITHOUT_VIEWING_OLD_MASKS = False
# No four-worker inference: one session owns this mask arm.
assert NUM_WORKERS == 1
'''

PREPARE = '''import json, hashlib, zipfile, shutil
import numpy as np, pandas as pd
from PIL import Image
clean = pd.read_csv(Path(DATA_ROOT)/"manifests/clean_manifest.csv").sort_values("image_id")
assert len(clean)==418 and clean.image_id.is_unique
assert ANN_ROOT is not None, "Attach the existing annotations beside FINAL"
manual_root = Path(ANN_ROOT)/"clean/masks"
assert manual_root.is_dir()
blind_ids = selfcheck_ids(clean)
if MODE == "PREPARE":
    # Download this PRIVATE local output; never enqueue source images to public HF.
    output = Path('/kaggle/working/S3_inputs_to_annotate.zip')
    total = sum((Path(DATA_ROOT)/r.relative_path).stat().st_size for r in clean.itertuples())
    assert total < 2*1024**3 and shutil.disk_usage(output.parent).free > total*2 + 1024**3
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as z:
        for r in clean.itertuples():
            source = Path(DATA_ROOT)/r.relative_path
            z.write(source, f'boxes/{r.image_id}{source.suffix}')
            if r.image_id in blind_ids:
                z.write(source, f'blind_pass/{r.image_id}{source.suffix}')
        z.writestr('INSTRUCTIONS.txt', 'Use labelme locally. boxes/: draw exactly two RECTANGLES tyre and tread on each image, no old masks. blind_pass/: independently draw tyre/tread polygons, marking/damage when visible, WITHOUT opening old masks. Disable embedded imageData on export. Upload JSON only as a PRIVATE Kaggle dataset with boxes/ and blind_pass/ folders. Do blind_pass FIRST, before viewing any old annotations or model results.')
    print('PREPARE COMPLETE. Download', output, 'and finish human inputs; no GPU work started.')
elif MODE != "RUN":
    raise ValueError('MODE must be PREPARE or RUN')
'''

READS = c.PULL[:c.PULL.index('REVISION = hf_read')]

RUN = r'''if MODE == "RUN":
    import os, sys, subprocess, time, gc, platform
    import torch
    from packaging.version import Version
    from huggingface_hub import HfApi, hf_hub_download
    assert BOXES_DRAWN_FROM_IMAGES_ONLY and BLIND_PASS_WITHOUT_VIEWING_OLD_MASKS, "Complete and truthfully confirm independent inputs first"
    assert torch.cuda.is_available(), 'Select GPU T4 x2 (one GPU used, no training)'
    assert Version(torch.__version__.split('+')[0]) >= Version('2.5.1'), 'Use a current Kaggle image with torch >=2.5.1; do not hot-replace torch'
    # Validate every input BEFORE downloading a model or starting inference.
    boxes, reference_hashes, image_hashes, blind = {}, {}, {}, {}
    for r in clean.itertuples():
        image = Path(DATA_ROOT)/r.relative_path
        with Image.open(image) as im:
            w,h = im.size
        boxes[r.image_id] = read_boxes(INPUT_ROOT/'boxes'/f'{r.image_id}.json', w,h)
        image_hashes[r.image_id] = digest(image)
        mp = manual_root/f'{r.image_id}.png'
        reference_hashes[r.image_id] = digest(mp)
        with Image.open(mp) as im:
            assert np.asarray(im).shape == (h,w)
            regions(np.asarray(im))
        if r.image_id in blind_ids:
            bp = INPUT_ROOT/'blind_pass'/f'{r.image_id}.json'
            blind[r.image_id] = digest(bp)
            read_blind(bp,w,h)
    protocol = dict(revision='s3-box-sam21-small-r1', code=SAM_COMMIT, model=MODEL_ID,
        model_revision=MODEL_REVISION, prompts=boxes, images=image_hashes,
        references=reference_hashes, blind_hashes=blind, blind_ids=blind_ids,
        mask_selection='maximum SAM predicted IoU; never reference IoU', precision='fp32',
        human_box_prompted=True, manual_mask_correction=False, seed=4622,
        self_consistency_gate='mean tread IoU > 0.90 on all 30 fixed images',
        torch=torch.__version__, numpy=np.__version__, implementation=S3_IMPLEMENTATION_SHA)
    arm = signature(protocol)
    prefix = f's3/{protocol["revision"]}/{arm}'
    root = Path(sess.stage_dir)/'s3'/arm
    (root/'records').mkdir(parents=True,exist_ok=True)
    def publish(path):
        sess.uploader.enqueue(path, prefix+'/'+Path(path).relative_to(root).as_posix())
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    publish(root/'protocol.json')
    assert sess.push_now('S3 protocol frozen before inference'), 'HF protocol upload failed; rerun later'
    revision = hf_read('S3 resume revision', lambda: HfApi().repo_info(REPO,repo_type='dataset',token=READ_TOKEN)).sha
    files = set(hf_read('S3 resume inventory', lambda: HfApi().list_repo_files(REPO,repo_type='dataset',revision=revision,token=READ_TOKEN)))
    # Restore only this exact fingerprint. Different prompts/inputs cannot reuse old masks.
    pending = []
    for r in clean.itertuples():
        path = root/'records'/f'{r.image_id}.npz'
        remote = prefix+'/records/'+path.name
        if not path.exists() and remote in files:
            downloaded = hf_read('restore '+r.image_id, lambda: hf_hub_download(REPO,remote,repo_type='dataset',revision=revision,token=READ_TOKEN,local_dir=str(root/'restore')))
            shutil.copyfile(downloaded,path)
        if path.exists():
            load_record(path, signature([arm,r.image_id]), (r.height,r.width))
            if remote not in files:
                publish(path)  # retry unpublished local records after an interrupted flush
        else:
            pending.append(r)
    print('HF/local valid records:', 418-len(pending), 'remaining:',len(pending))
    started = time.monotonic()
    if pending:
        # Pin upstream source. Do not upgrade Kaggle torch/torchvision or compile CUDA extensions.
        os.environ['SAM2_BUILD_CUDA'] = '0'
        subprocess.run([sys.executable,'-m','pip','install','-q','hydra-core==1.3.2','iopath==0.1.10'],check=True)
        subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps','--no-build-isolation',
            f'git+https://github.com/facebookresearch/sam2.git@{SAM_COMMIT}'],check=True)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        ck = hf_read('pinned SAM2 checkpoint', lambda: hf_hub_download(MODEL_ID,'sam2.1_hiera_small.pt',revision=MODEL_REVISION,token=READ_TOKEN,local_dir=str(root/'model_cache')))
        torch.manual_seed(4622); np.random.seed(4622)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        predictor = SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',ck,device='cuda:0',apply_postprocessing=False))
    failure = None
    try:
        for index,r in enumerate(pending):
            if time.monotonic()-started > 7.5*3600:
                raise RuntimeError('Planned session pause; rerun RUN to restore completed image records')
            assert shutil.disk_usage(root).free > 2*1024**3, 'Low staging disk: pause safely'
            with Image.open(Path(DATA_ROOT)/r.relative_path) as im:
                rgb = np.asarray(im.convert('RGB'))
            masks, confidence = {}, {}
            with torch.inference_mode():
                predictor.set_image(rgb)
                for name in ('tyre','tread'):
                    m, q, _ = predictor.predict(box=np.array(boxes[r.image_id][name],dtype=np.float32),multimask_output=True)
                    assert np.isfinite(q).all()
                    best = int(np.argmax(q))
                    masks[name] = m[best].astype(bool)
                    confidence[name] = float(q[best])
            predictor.reset_predictor()
            meta = dict(key=signature([arm,r.image_id]),image_id=r.image_id,arm=arm,confidence=confidence,
                fold=int(r.fold_id),session=r.session_group)
            path = root/'records'/f'{r.image_id}.npz'
            save_record(path,masks,meta); publish(path)
            del rgb, masks, m
            gc.collect(); torch.cuda.empty_cache()
            print(f'{index+1}/{len(pending)} new masks saved: {r.image_id}',flush=True)
            sess.maybe_push('S3 30-minute checkpoint',min_gap_min=30)
    except BaseException as exc:
        failure = dict(type=type(exc).__name__, completed=len(list((root/'records').glob('*.npz'))))
        (root/'last_interruption.json').write_text(json.dumps(failure)); publish(root/'last_interruption.json')
        raise
    finally:
        sess.push_now('S3 inference complete or interrupted')
        if 'predictor' in globals():
            del predictor
        gc.collect(); torch.cuda.empty_cache()
    # Manual references are read ONLY for evaluation, not prediction or mask selection.
    rows, self_rows = [], []
    for r in clean.itertuples():
        pred, meta = load_record(root/'records'/f'{r.image_id}.npz',signature([arm,r.image_id]),(r.height,r.width))
        with Image.open(manual_root/f'{r.image_id}.png') as im:
            refs = regions(np.asarray(im))
        for name in ('tyre','tread'):
            rows.append(dict(image_id=r.image_id,fold=r.fold_id,session=r.session_group,region=name,
                **scores(refs[name],pred[name]),confidence=meta['confidence'][name],
                tread_outside_tyre=int((pred['tread'] & ~pred['tyre']).sum())))
        if r.image_id in blind_ids:
            second_path = INPUT_ROOT/'blind_pass'/f'{r.image_id}.json'
            second = regions(read_blind(second_path,r.width,r.height))
            # Save indexed masks, never embedded RGB from labelme JSON.
            dest = root/f'blind_{r.image_id}.png'
            Image.fromarray(read_blind(second_path,r.width,r.height)).save(dest); publish(dest)
            for name in ('tyre','tread'):
                self_rows.append(dict(image_id=r.image_id,region=name,**scores(refs[name],second[name])))
    metrics, consistency = pd.DataFrame(rows),pd.DataFrame(self_rows)
    metrics.to_csv(root/'agreement_by_image.csv',index=False)
    consistency.to_csv(root/'self_consistency_by_image.csv',index=False)
    for grouping,name in [(['region'],'overall'),(['fold','region'],'by_fold'),(['session','region'],'by_session')]:
        metrics.groupby(grouping).agg(n=('image_id','size'),n_defined=('iou','count'),mean_iou=('iou','mean'),min_iou=('iou','min'),mean_dice=('dice','mean')).to_csv(root/f'agreement_{name}.csv')
    status = dict(mask_generation='complete',images=418,comparison_rows=len(metrics),
        self_consistency=consistency_gate(self_rows),selfcheck_images=30,
        pseudo_label_quality='reported_not_automatically_approved_for_S5',
        human_box_prompted=True,not_fully_automatic=True,arm=arm,
        s3_status='review_required',s5_started=False)
    (root/'STATUS.json').write_text(json.dumps(status,indent=2))
    for path in list(root.glob('*.csv'))+[root/'STATUS.json']:
        publish(path)
    assert sess.finish(), 'Upload incomplete; rerun to retry publication, not regenerate valid records'
    latest = hf_read('final verification',lambda:HfApi().repo_info(REPO,repo_type='dataset',token=READ_TOKEN)).sha
    inventory = set(hf_read('verify artifacts',lambda:HfApi().list_repo_files(REPO,repo_type='dataset',revision=latest,token=READ_TOKEN)))
    required = [prefix+'/records/'+r.image_id+'.npz' for r in clean.itertuples()]
    required += [prefix+'/'+p.name for p in root.glob('*.csv')]+[prefix+'/STATUS.json',prefix+'/protocol.json']
    assert set(required)<=inventory, 'Some expected results missing on HF; rerun RUN to republish'
    print('VERIFIED HF:',latest,prefix)
    print(metrics.groupby('region')[['iou','dice']].mean())
    print(status)
else:
    print('Preparation only. S3 is NOT completed until independent inputs, GPU inference and review.')
'''

def build():
    cells = c.start('''# NB11 — S3 manual versus SAM2 mask agreement

**Two-pass notebook. First PREPARE on CPU, then RUN on one Kaggle T4 session.**
Attach Tire Dataset Prepared, enable Internet and HF_TOKEN. Do not run four copies.

This uses **SAM2.1 Hiera Small**, not a changed classifier and not a trained student.
It produces **human-box-prompted, unedited SAM2 masks**, NOT fully automatic semantic
segmentation. Boxes must be drawn from images without consulting manual polygons;
manual-derived prompts or best-reference-IoU mask selection would bias this comparison.
No mask edits or hidden replacement for failed predictions. All 418 clean images
are scored by fold/session; original manual masks are untouched. There are no epochs:
resume is per completed image. Normal HF pushes are every 30 minutes, at major
completion, and on catchable Stop. Hard kernel kills cannot flush unpublished work.

1. Keep MODE=PREPARE; Run All. Download `S3_inputs_to_annotate.zip` from Kaggle Output.
2. In labelme, do the 30-image **blind_pass first**, without viewing old labels.
   Draw tyre/tread polygons and visible marking/damage polygons independently.
3. In boxes/, draw two rectangles (`tyre`, `tread`) per image, not old polygon boxes.
   Keep original filenames/dimensions. These are human prompts, not ground truth masks.
4. Upload only JSON folders boxes/ and blind_pass/ as a **private** Kaggle dataset.
5. Attach it, set INPUT_ROOT to its folder, MODE=RUN and truthfully set both confirmations.
   Select T4 x2; only GPU 0 is used to bound memory. Run All.

The blind tread-IoU gate is the existing >0.90 rule, not a SAM2 acceptance threshold.
SAM2 quality is reported including failures; review it before adopting pseudo-labels
for S5. This notebook does not train S4b/S5/S9 or resolve tyre-fold leakage.
''','s3',True)
    helper = Path(__file__).with_name('s3_mask_audit.py').read_text()
    fingerprint = hashlib.sha256((helper+RUN).encode()).hexdigest()
    cells += [b.code(CONFIG+f'\nS3_IMPLEMENTATION_SHA = "{fingerprint}"\n'),b.code(helper),
              b.code(PREPARE), b.code(READS),b.code(RUN)]
    c.save('NB11_S3_Manual_SAM2_Agreement.ipynb',cells)

if __name__=='__main__':
    build()
