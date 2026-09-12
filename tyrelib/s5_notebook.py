"""S5 notebook orchestration: a single upload owner for the entire Kaggle session."""
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from filelock import FileLock
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

import s5_data as d
from s5_runtime import atomic_json, YOLO_POLICY

REPO = 'Shanmuk4622/tyre-wear-study'

# Reviewed, narrow compatibility amendment: only YOLO's unintended default
# Albumentations are removed; semantic/RT recipes and original data stay intact.
PRE_REPAIR_RUNTIME = '67c1a33b92adfdcaf7df7f280e38284c8e4de23c385c8b8612fa0489113906d4'
PRE_JOURNAL_RUNTIME = '284d30d0b4d4c936bccad2cd3bdcb1595814c28777cf4dba99311c4e2bffa03f'


def source_compatible(plan):
    expected = {name:d.digest(Path(d.__file__).parent/name) for name in ('s5_data.py','s5_runtime.py')}
    actual = plan.get('implementation_sha256', {})
    return (set(actual)==set(expected) and actual['s5_data.py']==expected['s5_data.py']
            and actual['s5_runtime.py'] in (expected['s5_runtime.py'], PRE_REPAIR_RUNTIME, PRE_JOURNAL_RUNTIME))


def pilot_prefix(prefix, name):
    base = f'{prefix}/pilots/{name}'
    return base+'/'+YOLO_POLICY if d.MODELS[name][0]=='yolo' else base


def retry(operation):
    for attempt in range(8):
        try:
            return operation()
        except Exception as exc:
            response = getattr(exc, 'response', None)
            status = getattr(response, 'status_code', None)
            if status not in (429, 500, 502, 503, 504) or attempt == 7:
                raise
            import tyrelib as tl
            delay = max(min(300, 5*2**attempt), tl.parse_retry_after(str(exc)) or 0)
            hint = getattr(response, 'headers', {}).get('Retry-After', '')
            if hint.isdigit():
                delay = max(delay, int(hint))
            print(f'HF temporarily unavailable ({status}); retry in {delay:.0f}s', flush=True)
            until = time.monotonic()+delay+2
            while time.monotonic() < until:
                time.sleep(max(0, min(5, until-time.monotonic())))


def token(sess):
    value = sess.uploader.token
    assert value and sess.uploader.enabled, 'Enable writable HF_TOKEN and Internet'
    return value


def revision(sess):
    return retry(lambda: HfApi(token=token(sess)).repo_info(REPO, repo_type='dataset').sha)


def pull(sess, rel, rev, out, optional=False):
    try:
        return Path(retry(lambda: hf_hub_download(REPO, rel, repo_type='dataset', revision=rev,
                         token=token(sess), local_dir=str(out))))
    except EntryNotFoundError:
        if optional:
            return None
        raise


def push(sess, reason):
    assert sess.push_now(reason) and sess.uploader.enabled, 'HF push failed. Stop here; retry without discarding local progress.'


def prepare(sess, root, annotations):
    print('Hashing and validating all 418 clean images/manual masks and three splits...', flush=True)
    plan = d.protocol(d.inspect_data(root, annotations))
    api = HfApi(token=token(sess))
    plan['model_revisions'] = {m: retry(lambda m=m: api.model_info(m).sha)
                              for m in ('nvidia/mit-b0','nvidia/mit-b2','PekingU/rtdetr_v2_r18vd')}
    prefix = f's5/{d.REVISION}/{d.signature(plan)}'
    out = Path(sess.stage_dir)/'s5_protocol'; out.mkdir(exist_ok=True)
    path = out/'protocol.json'; atomic_json(path, plan)
    sess.uploader.enqueue(path, prefix+'/protocol.json')
    push(sess, 'S5 frozen data and manual-label protocol')
    rev = revision(sess)
    assert json.loads(pull(sess, prefix+'/protocol.json', rev, out/'verify').read_text()) == plan
    print('Verified published protocol at HF revision:', rev)
    print('81 planned runs. Existing fold leakage flags retained. No annotation work required.')
    return plan, prefix


def plan_prefix(plan):
    return f's5/{plan["revision"]}/{d.signature(plan)}'


def resolve_prefix(sess, prefix, rev):
    prefix = (prefix or '').strip().rstrip('/')
    if prefix:
        if not prefix.startswith(f's5/{d.REVISION}/') or len(prefix.split('/')[-1])!=64:
            raise ValueError('PREFIX must be blank (automatic discovery) or the full s5/... path printed by NB13.')
        return prefix
    base = f's5/{d.REVISION}'
    try:
        entries = retry(lambda: list(HfApi(token=token(sess)).list_repo_tree(
            REPO, path_in_repo=base, revision=rev, repo_type='dataset', recursive=True)))
    except EntryNotFoundError:
        entries = []
    matches = []
    expected = {name:d.digest(Path(d.__file__).parent/name) for name in ('s5_data.py','s5_runtime.py')}
    for entry in entries:
        if not entry.path.endswith('/protocol.json'):
            continue
        candidate = json.loads(pull(sess, entry.path, rev, Path(sess.stage_dir)/'s5_read').read_text())
        candidate_prefix = entry.path.rsplit('/', 1)[0]
        if (candidate_prefix==plan_prefix(candidate) and source_compatible(candidate)
                and candidate.get('models')=={k:list(v) for k,v in d.MODELS.items()}
                and candidate.get('packages')==d.PACKAGES):
            matches.append(candidate_prefix)
    if not matches:
        raise RuntimeError('No compatible NB13 protocol found on HF. Run the matching NB13 on CPU first; no training started.')
    if len(matches)!=1:
        raise RuntimeError('Multiple compatible NB13 protocols found. Set PREFIX explicitly; no automatic latest selection:\n'+'\n'.join(sorted(matches)))
    print('Auto-discovered frozen NB13 protocol:', matches[0], flush=True)
    return matches[0]


def load_plan(sess, prefix, root, annotations):
    rev = revision(sess)
    prefix = resolve_prefix(sess, prefix, rev)
    plan = json.loads(pull(sess, prefix+'/protocol.json', rev, Path(sess.stage_dir)/'s5_read').read_text())
    assert d.signature(plan) == prefix.split('/')[-1]
    assert plan['models']=={k:list(v) for k,v in d.MODELS.items()} and plan['packages']==d.PACKAGES
    assert source_compatible(plan), 'Runtime source differs from NB13; use matching notebooks'
    print('Verifying this session uses the same image, mask and split bytes...', flush=True)
    assert d.inspect_data(root, annotations)==plan['data'], 'Dataset differs from frozen NB13 protocol'
    return plan


ARTIFACTS = ['state.pt','STATUS.json','epochs.csv','identity.json','polygon_audit.csv',
             'metrics.json','mask_metrics.csv','roi_metrics.csv','roi_predictions.csv','SMOKE.json','hardware.json',
             'worker_environment.json', 'resume_recovery.json', 'memory_stop.json']


def repair_local_metadata(out, allow_legacy=False):
    """Caller holds writer lock; recover only metadata matching intact bytes."""
    if not (out/'state.pt').exists():
        return
    sha = d.digest(out/'state.pt')
    status = json.loads((out/'STATUS.json').read_text()) if (out/'STATUS.json').exists() else None
    if status and status.get('evaluated'):
        assert status['checkpoint_sha256']==sha, 'Completed local checkpoint mismatch; refusing automatic repair'
        return
    journal = json.loads((out/'checkpoint_pending.json').read_text()) if (out/'checkpoint_pending.json').exists() else None
    if journal and journal['status']['checkpoint_sha256']==sha:
        pending = journal['status']
        assert [h['epoch'] for h in journal['history']]==list(range(1,pending['epoch']+1))
        if status:
            assert status['plan_hash']==pending['plan_hash'] and status['job']==pending['job']
        # Restore a potentially interrupted CSV even if STATUS already matches.
        pd.DataFrame(journal['history']).to_csv(out/'epochs.csv', index=False)
        if not status or status['checkpoint_sha256']!=sha:
            atomic_json(out/'STATUS.json', pending)
            atomic_json(out/'resume_recovery.json', dict(reason='Interrupted local publication recovered from journal',
                        original_status=status, checkpoint_sha256=sha, recovered_epoch=pending['epoch']))
            print('Recovered interrupted local save at epoch', pending['epoch'], flush=True)
        return
    if status and status['checkpoint_sha256']==sha:
        return
    if allow_legacy and status and (out/'request.json').exists():
        req = json.loads((out/'request.json').read_text())
        recover_sidecars(out/'state.pt', status, req['plan'], req['job'], 'local-stopped-child')
        return
    raise RuntimeError('Local checkpoint has no matching journal; preserved without publishing inconsistent files')


def snapshot(sess, out, remote, reason, allow_legacy=False):
    """Copy immutable upload files under the writer lock. Never upload an actively-written checkpoint."""
    staging = out.parent/(out.name+'_upload')/uuid.uuid4().hex
    staging.mkdir(parents=True)
    queued = []
    assert shutil.disk_usage(staging).free > 3*2**30, 'Scratch nearly full; stop and preserve local state'
    with FileLock(str(out/'snapshot.lock')):
        repair_local_metadata(out, allow_legacy)
        for name in ARTIFACTS:
            source = out/name
            if source.exists():
                target = staging/name
                shutil.copy2(source, target)
                queued.append((target, f'{remote}/{name}'))
        for source in (out/'predictions').glob('*.json') if (out/'predictions').exists() else []:
            target = staging/'predictions'/source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
            queued.append((target, f'{remote}/predictions/{source.name}'))
        for source in out.glob('*.log'):
            target = staging/source.name
            shutil.copy2(source, target)
            queued.append((target, f'{remote}/logs/{sess.account}_{sess.session_id}_{source.name}'))
        for source in (out/'telemetry').rglob('*.gz') if (out/'telemetry').exists() else []:
            rel = source.relative_to(out)
            target = staging/rel; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            queued.append((target, f'{remote}/{rel.as_posix()}'))
    if (staging/'state.pt').exists():
        saved = json.loads((staging/'STATUS.json').read_text())
        assert d.digest(staging/'state.pt')==saved['checkpoint_sha256'], 'Local snapshot mismatch; nothing queued'
    sess.uploader.enqueue_batch(queued)
    push(sess, reason)
    safe_remove_generated(staging, out.parent/(out.name+'_upload'))


def recover_sidecars(state_file, status, plan, job, rev):
    """Rebuild interrupted non-completed sidecars from the intact full checkpoint.

    Called only after verifying downloaded bytes against pinned HF LFS metadata.
    Keep original metadata and do not edit/reinitialize model or optimizer state.
    """
    if status.get('evaluated') or status['status'] not in ('resumable', 'trained'):
        raise RuntimeError('Completed checkpoint mismatch: refusing automatic recovery')
    request = state_file.parent/'recovery_request.json'
    atomic_json(request, dict(plan_hash=d.signature(plan), job=job, original=status, revision=rev))
    code = '''import json,sys,torch,pandas as pd
from pathlib import Path
import s5_data as d
from s5_runtime import atomic_json
p=Path(sys.argv[1]); req=json.loads(Path(sys.argv[2]).read_text())
s=torch.load(p,map_location='cpu',weights_only=False)
assert s['plan_hash']==req['plan_hash'] and s['job']==req['job'], 'Checkpoint identity mismatch'
assert 1<=s['epoch']<=60 and [h['epoch'] for h in s['history']]==list(range(1,s['epoch']+1)), 'Incomplete checkpoint history'
assert all(k in s for k in ('model','optimizer','scheduler','scaler','rng','runtime')), 'Missing resumable state'
assert req['job']['backend']=='semantic', 'Automatic sidecar repair is restricted to semantic checkpoints'
sha=d.digest(p)
pd.DataFrame(s['history']).to_csv(p.parent/'epochs.csv',index=False)
atomic_json(p.parent/'STATUS.json',dict(status='trained' if s['epoch']==60 else 'resumable',epoch=s['epoch'],plan_hash=s['plan_hash'],job=s['job'],checkpoint_sha256=sha,evaluated=False,yolo_policy=s.get('yolo_policy')))
atomic_json(p.parent/'resume_recovery.json',dict(reason='Published sidecars mismatched intact checkpoint',source_revision=req['revision'],original_status=req['original'],checkpoint_sha256=sha,recovered_epoch=s['epoch']))
print('Recovered intact checkpoint at epoch',s['epoch'],'; original status retained in resume_recovery.json')
'''
    result = subprocess.run([sys.executable, '-c', code, str(state_file), str(request)],
                            capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError('Checkpoint recovery validation failed; no fresh training allowed:\n'+result.stderr[-3000:])
    print(result.stdout, flush=True)
    return json.loads((state_file.parent/'STATUS.json').read_text())


def restore(sess, plan, job, out, remote):
    if (out/'state.pt').exists():
        with FileLock(str(out/'snapshot.lock')):
            repair_local_metadata(out, allow_legacy=True)
    rev = revision(sess)
    cache = out/'remote'
    status_file = pull(sess, remote+'/STATUS.json', rev, cache, optional=True)
    if status_file:
        status = json.loads(status_file.read_text())
        assert status['plan_hash']==d.signature(plan) and status['job']==job
        if job['backend']=='yolo':
            assert status.get('yolo_policy')==YOLO_POLICY, 'Old-policy YOLO result cannot be reused as corrected training'
        if status['status']=='completed' and status['evaluated']:
            info = retry(lambda: HfApi(token=token(sess)).get_paths_info(REPO, [remote+'/state.pt'], repo_type='dataset', revision=rev))
            assert len(info)==1 and info[0].lfs and info[0].lfs.sha256==status['checkpoint_sha256']
            return 'public_completed'  # Do not download completed model weights merely to skip them.
    state_file = pull(sess, remote+'/state.pt', rev, cache, optional=True)
    if status_file is None and state_file is None:
        # Never silently replace unpublished local progress by a fresh remote-missing run.
        if (out/'state.pt').exists():
            local = json.loads((out/'STATUS.json').read_text())
            assert local['plan_hash']==d.signature(plan) and local['job']==job
        return False
    assert status_file and state_file, 'Incomplete HF generation; do not restart. Recover the publishing session first.'
    status = json.loads(status_file.read_text())
    assert status['plan_hash']==d.signature(plan) and status['job']==job
    recovered = False
    if d.digest(state_file)!=status['checkpoint_sha256']:
        info = retry(lambda: HfApi(token=token(sess)).get_paths_info(REPO, [remote+'/state.pt'], repo_type='dataset', revision=rev))
        assert len(info)==1 and info[0].lfs and info[0].lfs.sha256==d.digest(state_file), 'Downloaded checkpoint differs from pinned HF bytes'
        status = recover_sidecars(state_file, status, plan, job, rev)
        recovered = True
    if (out/'STATUS.json').exists():
        local = json.loads((out/'STATUS.json').read_text())
        assert local['plan_hash']==d.signature(plan) and local['job']==job
        if local['epoch'] > status['epoch'] or (local.get('evaluated') and not status.get('evaluated')):
            print('Keeping newer unpublished local state:', job['run_id'], flush=True)
            return 'local_completed' if local.get('evaluated', False) else False
    shutil.copy2(state_file, out/'state.pt')
    shutil.copy2(status_file, out/'STATUS.json')
    for name in ARTIFACTS:
        if name in ('state.pt','STATUS.json'):
            continue
        if recovered and name in ('epochs.csv', 'resume_recovery.json'):
            shutil.copy2(state_file.parent/name, out/name)
            continue
        file = pull(sess, f'{remote}/{name}', rev, cache, optional=True)
        if file:
            shutil.copy2(file, out/name)
    # Restore resumable evaluation outputs, not just weights. Missing directory is normal before evaluation.
    try:
        entries = retry(lambda: list(HfApi(token=token(sess)).list_repo_tree(REPO,
                path_in_repo=remote+'/predictions', revision=rev, repo_type='dataset', recursive=True)))
    except EntryNotFoundError:
        entries = []
    for entry in entries:
        if entry.path.endswith('.json'):
            target = out/'predictions'/Path(entry.path).name; target.parent.mkdir(exist_ok=True)
            shutil.copy2(pull(sess, entry.path, rev, cache), target)
    if recovered:
        snapshot(sess, out, remote, 'S5 verified checkpoint sidecar recovery')
    return status['status']=='completed' and status['evaluated']


def worker_environment(out, env, expected=None):
    """Restore NumPy per child, without replacing notebook/CUDA packages or relaxing checks."""
    from s5_runtime import runtime_versions
    expected = expected or runtime_versions()
    checkpoint = Path(out)/'state.pt'
    if checkpoint.exists():
        # Read recorded versions in a short CPU process; don't retain a full checkpoint
        # in the notebook parent while the training child allocates its own state.
        source = "import json,sys,torch; s=torch.load(sys.argv[1],map_location='cpu',weights_only=False); print(json.dumps(s['runtime']))"
        p = subprocess.run([sys.executable,'-c',source,str(checkpoint)],env=env,
                           capture_output=True,text=True,timeout=120)
        if p.returncode:
            raise RuntimeError('Could not inspect saved runtime; checkpoint was not reset:\n'+p.stderr[-3000:])
        expected = json.loads(p.stdout.strip().splitlines()[-1])
    probe = ("import json,importlib.metadata as m,numpy; "
             "from s5_runtime import runtime_versions; "
             "print(json.dumps(dict(versions=runtime_versions(),loaded_numpy=numpy.__version__)))")
    def inspect(candidate_env):
        p = subprocess.run([sys.executable,'-c',probe],env=candidate_env,
                           capture_output=True,text=True,timeout=120)
        if p.returncode:
            raise RuntimeError('Worker runtime probe failed before training:\n'+p.stderr[-3000:])
        return json.loads(p.stdout.strip().splitlines()[-1])
    observed = inspect(env)
    differences = {k:(expected.get(k),observed['versions'].get(k)) for k in set(expected)|set(observed['versions'])
                   if expected.get(k)!=observed['versions'].get(k)}
    if set(differences)-{'numpy'}:
        raise RuntimeError(f'Non-NumPy runtime mismatch (saved, worker): {differences}. CUDA stack was not modified.')
    repaired_env = dict(env)
    if differences or observed['loaded_numpy']!=expected['numpy']:
        wanted = expected['numpy']
        import re
        assert re.fullmatch(r'\d+\.\d+\.\d+',wanted), 'Unexpected NumPy version string'
        target = Path(out).parent/'runtime_packages'/('numpy_'+wanted)
        target.mkdir(parents=True,exist_ok=True)
        print(f'[RUNTIME] Restoring NumPy {wanted} for this worker only; notebook and CUDA stack unchanged.',flush=True)
        if not (target/'numpy/__init__.py').exists():
            subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-deps','--only-binary=:all:',
                '--no-cache-dir','--disable-pip-version-check','--timeout','60',
                '--target',str(target),'numpy=='+wanted],env=env,check=True,timeout=600)
        repaired_env['PYTHONPATH'] = str(target)+os.pathsep+str(Path.cwd())+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        observed = inspect(repaired_env)
    if observed['versions']!=expected or observed['loaded_numpy']!=expected['numpy']:
        raise RuntimeError(f'Runtime restoration failed. Expected {expected}; got {observed}. Training not started.')
    record = dict(expected=expected,verified=observed,checkpoint_resume=checkpoint.exists(),
                  policy='isolated-numpy-exact-r1')
    atomic_json(Path(out)/'worker_environment.json',record)
    print('[RUNTIME] Worker verified against '+('checkpoint' if checkpoint.exists() else 'pilot/session')+
          ': NumPy '+expected['numpy']+', torch '+expected['torch'],flush=True)
    return repaired_env


def memory_pressure():
    """Exclude only reclaimable inactive clean file cache, not model RAM."""
    import tyrelib as tl
    used, limit, source = tl.container_memory()
    reclaimable = 0
    if source.startswith('cgroup:'):
        for file in (Path('/sys/fs/cgroup/memory.stat'), Path('/sys/fs/cgroup/memory/memory.stat')):
            try:
                stats = {k:int(v) for k,v in (line.split() for line in file.read_text().splitlines())}
                reclaimable = max(0, stats.get('total_inactive_file', stats.get('inactive_file',0))
                    - stats.get('total_dirty', stats.get('file_dirty',0))
                    - stats.get('total_writeback', stats.get('file_writeback',0)))
                break
            except (OSError, ValueError):
                continue
    return dict(raw=used, working=max(0,used-reclaimable), limit=limit, reclaimable=reclaimable, source=source)


def launch(sess, plan, job, root, annotations, out, remote, action):
    request = out/'request.json'
    atomic_json(request, dict(plan=plan, job=job, root=str(root), annotations=str(annotations), out=str(out), action=action))
    env = dict(os.environ, HF_TOKEN=token(sess), PYTHONUNBUFFERED='1')
    env = worker_environment(out, env, getattr(sess,'_s5_runtime_targets',{}).get(job['model']))
    log = out/(action+'.log')
    last_push = time.monotonic()
    print(action.upper(), job['run_id'], '— detailed progress below', flush=True)
    with log.open('w', encoding='utf-8') as stream:
        child = subprocess.Popen([sys.executable, '-u', str(Path.cwd()/'s5_runtime.py'), str(request)],
                                 stdout=stream, stderr=subprocess.STDOUT, env=env)
        import tyrelib as tl
        monitor = tl.HardwareMonitor(out/'telemetry'/f'{sess.account}_{sess.session_id}_{action}', gpu_hz=1., sys_hz=.2)
        if monitor._psutil:
            monitor._proc = monitor._psutil.Process(child.pid)
        atomic_json(out/'hardware.json', monitor.gpu_static())
        monitor.start()
        offset = 0
        last_output = time.monotonic()
        original_term = signal.getsignal(signal.SIGTERM)
        def stop_signal(signum, frame):
            raise KeyboardInterrupt('Platform termination: flushing last completed epoch')
        signal.signal(signal.SIGTERM, stop_signal)
        try:
            while child.poll() is None:
                time.sleep(2)
                with log.open(encoding='utf-8', errors='replace') as reader:
                    reader.seek(offset); text = reader.read(); offset = reader.tell()
                if text:
                    print(text, end='', flush=True)
                    last_output = time.monotonic()
                elif time.monotonic()-last_output > 60:
                    print(f'{job["run_id"]}: process still running ({action}); monitoring memory and checkpoint progress.', flush=True)
                    last_output = time.monotonic()
                if time.monotonic()-last_push >= 1800:
                    monitor.dump()
                    snapshot(sess, out, remote, 'S5 30-minute progress')
                    last_push = time.monotonic()
                memory = memory_pressure()
                if memory['limit'] and memory['working']/memory['limit'] > .90:
                    atomic_json(out/'memory_stop.json', memory)
                    raise KeyboardInterrupt('Container working RAM above90%; preserving last completed epoch (clean inactive file cache excluded)')
                if sess.guard.near_limit(margin_min=35):
                    raise KeyboardInterrupt('Session budget nearly used; preserving last completed epoch')
        except BaseException:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
            monitor.stop()
            snapshot(sess, out, remote, 'S5 catchable Stop/error', allow_legacy=True)
            raise
        finally:
            monitor.stop()
            signal.signal(signal.SIGTERM, original_term)
    tail = log.read_text(encoding='utf-8', errors='replace')[offset:]
    if tail:
        print(tail)
    # Major action completion includes failures: preserve evidence before raising.
    snapshot(sess, out, remote, f'S5 {action} ended: {job["run_id"]}')
    assert child.returncode == 0, f'{action} failed; see the final lines above. Do not reset the run or reduce the model.'


def safe_remove_generated(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    assert path != parent and path.is_relative_to(parent), 'Refusing broad scratch cleanup'
    if path.exists():
        shutil.rmtree(path)


def run_family(sess, plan, prefix, root, annotations, family, mode):
    assert mode in ('PILOT','TRAIN','AUTO')
    base = Path(sess.stage_dir)/'s5_jobs'/d.signature(plan)
    base.mkdir(parents=True, exist_ok=True)
    if sess.worker_id==0:
        amendment = base/'checkpoint_journal_amendment.json'
        atomic_json(amendment, dict(plan_hash=d.signature(plan),
            original_runtime=plan['implementation_sha256']['s5_runtime.py'],
            repaired_runtime=d.digest(Path(d.__file__).parent/'s5_runtime.py'),
            reason='Interrupted-save metadata journal and cache-aware RAM guard; no training recipe changes'))
        sess.uploader.enqueue(amendment, f'{prefix}/runtime_amendments/checkpoint-journal-r1.json')
        sess.uploader.enqueue(Path(d.__file__).parent/'s5_runtime.py', f'{prefix}/runtime_amendments/checkpoint-journal-r1.py')
        push(sess, 'S5 checkpoint-journal repair provenance')
    if family=='yolo' and sess.worker_id==0:
        amendment = base/'yolo_runtime_amendment.json'
        atomic_json(amendment, dict(policy=YOLO_POLICY, plan_hash=d.signature(plan),
            original_runtime=plan['implementation_sha256']['s5_runtime.py'],
            repaired_runtime=d.digest(Path(d.__file__).parent/'s5_runtime.py'),
            reason='Disable unintended default Albumentations; implement the frozen flip-only recipe',
            old_pilots='retained, not accepted as repaired-policy evidence'))
        sess.uploader.enqueue(amendment, f'{prefix}/runtime_amendments/{YOLO_POLICY}.json')
        sess.uploader.enqueue(Path(d.__file__).parent/'s5_runtime.py', f'{prefix}/runtime_amendments/{YOLO_POLICY}.py')
        push(sess, 'YOLO runtime correction provenance')
    if mode=='AUTO':
        assert family=='yolo', 'AUTO currently applies to the repaired YOLO notebook'
        def missing_pilots():
            rev = revision(sess)
            missing = []
            from s5_runtime import runtime_versions
            for name, spec in plan['models'].items():
                if spec[0]!=family:
                    continue
                file = pull(sess, pilot_prefix(prefix,name)+'/PASS.json', rev, base/'pilot_checks', optional=True)
                pilot = json.loads(file.read_text()) if file else {}
                if not (pilot.get('status')=='passed' and pilot.get('resume_verified')
                        and pilot.get('plan_hash')==d.signature(plan) and pilot.get('yolo_policy')==YOLO_POLICY
                        and pilot.get('runtime')==runtime_versions()):
                    missing.append(name)
            return missing
        missing = missing_pilots()
        if missing and sess.worker_id==0:
            print('AUTO: validating corrected YOLO policy, then continuing to full training.', flush=True)
            run_family(sess, plan, prefix, root, annotations, family, 'PILOT')
            missing = missing_pilots()
        started = time.monotonic()
        while missing:
            if sess.worker_id==0 or time.monotonic()-started>2400 or sess.guard.near_limit():
                raise RuntimeError('Corrected pilots not yet passed. Keep worker0 running; rerun AUTO to continue safely.')
            print('Waiting for worker0 to publish corrected pilots; no claim commits:', missing, flush=True)
            time.sleep(30)
            missing = missing_pilots()
        mode = 'TRAIN'
        print('AUTO: all four corrected pilots verified. Starting/resuming the assigned full runs.', flush=True)
    if mode=='PILOT':
        if sess.worker_id!=0:
            print('PILOT runs only on the first active account (worker0). This copy will not duplicate it. '
                  'Run PILOT there, then set MODE=TRAIN in all copies after it passes.', flush=True)
            return
        selected = [j for j in d.jobs(plan, family) if j['fold']==1 and j['seed']==1]
    else:
        selected = d.assigned(plan, family, sess.worker_id, sess.num_workers)
        rev = revision(sess)
        for name, spec in plan['models'].items():
            if spec[0]!=family:
                continue
            pilot = json.loads(pull(sess, pilot_prefix(prefix,name)+'/PASS.json', rev, base/'pilot_checks').read_text())
            assert pilot['plan_hash']==d.signature(plan) and pilot['status']=='passed' and pilot['resume_verified']
            if family=='yolo':
                assert pilot.get('yolo_policy')==YOLO_POLICY, 'Corrected policy pilot is required'
            from s5_runtime import runtime_versions
            current = runtime_versions()
            changed = {k for k in set(current)|set(pilot['runtime']) if current.get(k)!=pilot['runtime'].get(k)}
            assert not (changed-{'numpy'}), 'Pilot ran in another package/runtime image; rerun PILOT here first'
            if not hasattr(sess,'_s5_runtime_targets'):
                sess._s5_runtime_targets = {}
            # Fresh workers use the proven pilot NumPy; resumed workers use their own saved runtime.
            sess._s5_runtime_targets[name] = pilot['runtime']
        print(f'{len(selected)} statically owned jobs. No claim commits or stealing. Stop all copies before changing worker count.')
    for job in selected:
        if sess.guard.near_limit(margin_min=45):
            print('Session nearly used; restart TRAIN to continue remaining jobs.'); break
        out = base/(('pilot_' if mode=='PILOT' else '')+job['run_id']); out.mkdir(exist_ok=True)
        remote = pilot_prefix(prefix,job['model']) if mode=='PILOT' else f'{prefix}/runs/{job["run_id"]}'
        if mode=='PILOT':
            # Separate pilot state can never become one of the 81 scientific results.
            safe_remove_generated(out, base); out.mkdir()
            launch(sess, plan, job, root, annotations, out, remote, 'smoke')
            # A second isolated process must load the saved state and perform resumed updates.
            # Semantic/RT pilot saves an epoch only after a complete epoch; test that path explicitly.
            pilot = json.loads((out/'SMOKE.json').read_text())
            assert pilot['status']=='passed'
            launch(sess, plan, job, root, annotations, out, remote, 'resume_test')
            passed = dict(status='passed', plan_hash=d.signature(plan), resume_verified=True,
                          runtime=pilot['runtime'], completed_at=datetime.now(timezone.utc).isoformat(),
                          yolo_policy=YOLO_POLICY if family=='yolo' else None)
            atomic_json(out/'PASS.json', passed)
            sess.uploader.enqueue(out/'PASS.json', remote+'/PASS.json')
            push(sess, 'S5 pilot and cross-process resume check passed')
        else:
            restored = restore(sess, plan, job, out, remote)
            if restored:
                print('SKIP completed:', job['run_id']);
                if restored=='local_completed':
                    snapshot(sess, out, remote, 'S5 ensure completed local state is public')
            else:
                launch(sess, plan, job, root, annotations, out, remote, 'train')
                launch(sess, plan, job, root, annotations, out, remote, 'evaluate')
            rev = revision(sess)
            verified = json.loads(pull(sess, remote+'/STATUS.json', rev, base/'verification').read_text())
            assert verified['status']=='completed' and verified['epoch']==60 and verified['evaluated']
        # Only this generated job scratch, after successful verified upload; never the attached dataset.
        safe_remove_generated(out, base)
        safe_remove_generated(out.parent/(out.name+'_upload'), base)
    print('This worker finished its available assignments. NB17 checks all81 across workers.')


def report(sess, plan, prefix):
    import numpy as np
    from sklearn.metrics import f1_score
    rev = revision(sess)
    base = Path(sess.stage_dir)/'s5_report'/d.signature(plan); base.mkdir(parents=True, exist_ok=True)
    rows, roi_rows, dense_rows = [], [], []
    api = HfApi(token=token(sess))
    for job in d.jobs(plan):
        remote = f'{prefix}/runs/{job["run_id"]}'
        status_file = pull(sess, remote+'/STATUS.json', rev, base/'read', optional=True)
        if status_file is None:
            rows.append(dict(**job, status='not_started', epoch=0)); continue
        st = json.loads(status_file.read_text())
        assert st['plan_hash']==d.signature(plan) and st['job']==job
        if job['backend']=='yolo':
            assert st.get('yolo_policy')==YOLO_POLICY, 'YOLO result uses the old augmentation policy'
        if st['status']!='completed':
            rows.append(dict(**job, status=st['status'], epoch=st['epoch'])); continue
        assert st['epoch']==60 and st['evaluated']
        info = retry(lambda: api.get_paths_info(REPO, [remote+'/state.pt'], repo_type='dataset', revision=rev))
        assert len(info)==1 and info[0].lfs and info[0].lfs.sha256==st['checkpoint_sha256'], 'HF checkpoint hash mismatch'
        files = {name:pull(sess, remote+'/'+name, rev, base/'read') for name in st['artifact_sha256']}
        assert all(d.digest(files[name])==sha for name, sha in st['artifact_sha256'].items())
        history = pd.read_csv(files['epochs.csv'])
        assert history.epoch.tolist()==list(range(1,61))
        predictions = pd.read_csv(files['roi_predictions.csv'])
        metrics = pd.read_csv(files['roi_metrics.csv'])
        expected = set(plan['data']['splits'][str(job['fold'])]['validation'])
        assert set(predictions['mode'])==set(plan['downstream']['modes'])
        for mode, group in predictions.groupby('mode'):
            assert set(group.image_id)==expected and group.image_id.is_unique
            import tyrelib as tl
            truth = {x['image_id']:tl.C2I[x['proxy_label']] for x in plan['data']['records']}
            assert group.truth.tolist()==group.image_id.map(truth).tolist(), 'Prediction labels differ from frozen manifest'
            score = f1_score(group.truth, group.prediction, labels=[0,1,2], average='macro', zero_division=0)
            row = metrics.loc[metrics['mode'].eq(mode)].iloc[0]
            assert np.isclose(score, row.macro_f1) and len(group)==row.n
            baseline = metrics.loc[metrics['mode'].eq('full'),'macro_f1'].iloc[0]
            assert np.isclose(row.delta_macro_f1_vs_full, score-baseline)
            roi_rows.append(dict(**job, **row.to_dict()))
        loc = json.loads(files['metrics.json'].read_text())
        assert loc['job']==job and loc['plan_hash']==d.signature(plan) and loc['n_validation']==len(expected)
        entries = retry(lambda: list(api.list_repo_tree(REPO, path_in_repo=remote+'/predictions', revision=rev,
                                                        repo_type='dataset', recursive=True)))
        assert {Path(e.path).stem for e in entries if e.path.endswith('.json')}==expected, 'Missing native per-image localisation predictions'
        dense = dict(**job, **loc['localisation'])
        if job['backend']=='semantic' or job['model'].endswith('_seg'):
            masks = pd.read_csv(files['mask_metrics.csv'])
            assert set(masks.region)=={'tyre','tread'}
            for region, group in masks.groupby('region'):
                assert set(group.image_id)==expected and group.image_id.is_unique
                for metric in ('iou','dice','boundary_f1_2px'):
                    dense[region+'_'+metric] = float(group[metric].mean())
        dense_rows.append(dense)
        rows.append(dict(**job, status='completed_verified', epoch=60))
    status = pd.DataFrame(rows)
    status.to_csv(base/'inventory.csv', index=False)
    if roi_rows:
        roi = pd.DataFrame(roi_rows); roi.to_csv(base/'roi_by_run.csv', index=False)
        roi.groupby(['model','fold','mode']).agg(n_seeds=('seed','nunique'), mean_macro_f1=('macro_f1','mean'),
            mean_delta=('delta_macro_f1_vs_full','mean'), std_delta=('delta_macro_f1_vs_full','std')).reset_index().to_csv(base/'roi_summary.csv', index=False)
    if dense_rows:
        dense = pd.DataFrame(dense_rows)
        dense.to_csv(base/'localisation_by_run.csv', index=False)
        cols = [c for c in dense if c not in ('model','backend','fold','seed','run_id')]
        dense.groupby(['model','fold'])[cols].agg(['mean','std','count']).to_csv(base/'localisation_summary.csv')
    completed = int(status.status.eq('completed_verified').sum())
    atomic_json(base/'STATUS.json', dict(status='complete' if completed==81 else 'partial',
        verified_runs=completed, planned_runs=81, hf_revision=rev, plan_hash=d.signature(plan),
        limitations=plan['limitations'], sam2_comparison='deferred', s9='not_completed'))
    for file in base.glob('*'):
        if file.is_file():
            sess.uploader.enqueue(file, f'{prefix}/report/{file.name}')
    push(sess, 'S5 public-HF report')
    print(f'HF verified {completed}/81. SAM2 comparison remains deferred; S9 not completed.')
    return status
