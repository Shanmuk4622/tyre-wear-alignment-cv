"""
tyrelib.py -- Tyre-wear comparative study: experiment infrastructure.

Built for: Kaggle dual-T4 sessions, HuggingFace as the only permanent store,
N Kaggle accounts sharing ONE HuggingFace account (Shanmuk4622).

Design rules baked in (see docs/05):
  * workers never talk to each other -- ownership is arithmetic
  * one rate-limit bucket per TOKEN, process-wide          (Bug 1)
  * one registry shard per WRITER, merged on read          (Bug 2)
  * a worker may always resume its own run                 (Bug 3)
  * ownership uses a STATIC cost table, always             (Bug 7)
  * resume restores optimizer, scheduler, scaler, all RNG  (Bug 6)
  * NO EARLY STOPPING -- every run trains its full epoch budget

Generated into notebooks by build_notebooks.py. Edit THIS file, never the
base64 blob inside a notebook.
"""
from __future__ import annotations

__version__ = "v11"

import atexit
import csv
import contextlib
import gzip
import gc
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

NA = "NA"

# --------------------------------------------------------------------------
# 0. Small utilities
# --------------------------------------------------------------------------

def now() -> float:
    """Float epoch seconds. Never store only ISO strings -- second granularity
    makes same-second events across shards sort ambiguously."""
    return time.time()


def iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts if ts is not None else now()))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(Path(path), text.encode("utf-8"))


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=str))


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def release_host_memory() -> bool:
    """Return freed Python/PyTorch arenas to the Linux host when possible.

    Kaggle keeps one Python process alive for many models.  Large checkpoint
    serialisations and Hugging Face LFS uploads free their temporary buffers,
    but glibc can keep those arenas mapped in the process.  The public NB06
    telemetry showed that mapped RSS accumulating across epochs/runs until the
    kernel was killed even though both T4s had ample free VRAM.  ``malloc_trim``
    releases those already-free arenas without changing any live tensor.
    """
    gc.collect()
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes
        return bool(ctypes.CDLL(None).malloc_trim(0))
    except Exception:
        return False


def atomic_clone_file(source: Path, destination: Path) -> None:
    """Atomically snapshot one local file, using a hard link when possible."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with contextlib.suppress(FileNotFoundError):
        tmp.unlink()
    try:
        os.link(source, tmp)
    except OSError:
        shutil.copy2(source, tmp)
    os.replace(tmp, destination)


_KNOWN_EPOCH_SCHEMA_INSERTIONS = (
    # v5 added this field between memory and CUDA revisions while the old
    # writer was still appending positional rows under the v4 header.
    ("runtime_hf_commit_policy_revision", "runtime_memory_safety_revision"),
)


def read_epoch_history(path: Path, repair: bool = True) -> pd.DataFrame:
    """Read an epoch CSV and losslessly migrate known mixed-schema rows.

    CSV append is positional.  If telemetry gains one field but an existing
    file keeps its old header, every later value shifts one column and pandas
    raises a ParserError.  This reader recognises recorded schema insertions,
    inserts blanks into the older rows, and atomically rewrites one canonical
    table.  Unknown width changes still raise instead of silently dropping or
    mislabelling an epoch.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return pd.DataFrame()

    header, data = list(rows[0]), [list(r) for r in rows[1:]]
    changed = False
    for field, after in _KNOWN_EPOCH_SCHEMA_INSERTIONS:
        if field in header or after not in header:
            continue
        old_width = len(header)
        insert_at = header.index(after) + 1
        wider = [r for r in data if len(r) == old_width + 1]
        # A revision token at the insertion point makes this migration
        # unambiguous. Never guess where an arbitrary extra CSV value belongs.
        if not wider or not all(re.fullmatch(r"\d{4}-\d{2}-\d{2}-r\d+", r[insert_at] or "")
                                for r in wider):
            continue
        header.insert(insert_at, field)
        for i, row in enumerate(data):
            if len(row) == old_width:
                data[i] = row[:insert_at] + [""] + row[insert_at:]
        changed = True

    bad = [(i + 2, len(row)) for i, row in enumerate(data) if len(row) != len(header)]
    if bad:
        sample = ", ".join(f"line {line}: {width}" for line, width in bad[:8])
        raise ValueError(
            f"unrecognised epochs.csv schema drift in {path}: header has "
            f"{len(header)} fields; {sample}. The file is preserved unchanged."
        )

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(data)
    frame = pd.read_csv(io.StringIO(buf.getvalue()))
    if changed and repair:
        atomic_write_text(path, frame.to_csv(index=False))
        _print("HISTORY", f"repaired mixed telemetry schema: {path.name} "
                          f"({len(frame)} epoch rows, {len(frame.columns)} columns)")
    return frame


def append_epoch_row(path: Path, row: dict) -> pd.DataFrame:
    """Atomically append by column name, expanding the header when needed."""
    path = Path(path)
    old = read_epoch_history(path, repair=True) if path.exists() else pd.DataFrame()
    new = pd.DataFrame([row])
    columns = list(old.columns) + [c for c in new.columns if c not in old.columns]
    out = pd.concat([old.reindex(columns=columns), new.reindex(columns=columns)],
                    ignore_index=True)
    if "epoch" in out.columns:
        out = (out.drop_duplicates(subset=["epoch"], keep="last")
                  .sort_values("epoch", kind="stable"))
    atomic_write_text(path, out.to_csv(index=False))
    return out


def config_hash(cfg: dict) -> str:
    """Stable across processes. Debug-only keys (leading _) are excluded so a
    resumed run does not fail its own hash check."""
    clean = {k: v for k, v in sorted(cfg.items()) if not str(k).startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()[:12]


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng() -> dict:
    import torch
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    import torch
    if not state:
        return
    with contextlib.suppress(Exception):
        random.setstate(state["python"])
    with contextlib.suppress(Exception):
        np.random.set_state(state["numpy"])
    with contextlib.suppress(Exception):
        torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
    with contextlib.suppress(Exception):
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, "cpu") else s for s in state["cuda"]])


def human_time(sec: float) -> str:
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec/60:.1f}m"
    return f"{sec/3600:.2f}h"


def _print(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


# --------------------------------------------------------------------------
# 1. Rate limiting -- ONE BUCKET PER TOKEN, PROCESS-WIDE  (Bug 1)
# --------------------------------------------------------------------------

class SharedRateLimiter:
    """HuggingFace meters writes PER USER, not per repository.

    We run N Kaggle accounts against ONE HuggingFace account (Shanmuk4622),
    so every worker draws from the same 128/hour budget. A limiter living on
    the uploader object would multiply the apparent budget by the number of
    repos or uploader instances and the cap would be decorative.
    """
    _buckets: dict[str, "SharedRateLimiter"] = {}
    _registry_lock = threading.Lock()

    def __init__(self, limit: int):
        self.limit = int(limit)
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    @classmethod
    def for_token(cls, token: str | None, limit: int) -> "SharedRateLimiter":
        key = hashlib.sha256((token or "anon").encode()).hexdigest()[:16]
        with cls._registry_lock:
            b = cls._buckets.setdefault(key, cls(limit))
            b.limit = min(b.limit, int(limit))     # most conservative wins
            return b

    def count_last_hour(self) -> int:
        t = now()
        with self._lock:
            while self._times and t - self._times[0] >= 3600:
                self._times.popleft()
            return len(self._times)

    def wait_for_slot(self, stop: threading.Event | None = None) -> bool:
        while True:
            if stop is not None and stop.is_set():
                return False
            t = now()
            with self._lock:
                while self._times and t - self._times[0] >= 3600:
                    self._times.popleft()
                if len(self._times) < self.limit:
                    self._times.append(t)
                    return True
                oldest = self._times[0]
            wait = max(1.0, 3600 - (t - oldest) + 2.0)
            _print("RATE", f"budget spent ({self.limit}/hr); sleeping {wait:.0f}s")
            if stop is not None:
                stop.wait(wait)
            else:
                time.sleep(wait)


def parse_retry_after(err: str) -> float | None:
    """HF's 429 body carries a human-readable hint. Parsing it beats blind
    exponential backoff, which either wastes a window or hammers early."""
    m = re.search(r"retry after (\d+)\s*second", err, re.I)
    if m:
        return float(m.group(1)) + 2.0
    m = re.search(r"in about (\d+)\s*minute", err, re.I)
    if m:
        return float(m.group(1)) * 60.0 + 5.0
    m = re.search(r"in about (\d+)\s*hour", err, re.I)
    if m:
        return float(m.group(1)) * 3600.0 + 10.0
    return None


# --------------------------------------------------------------------------
# 2. Background uploader -- batched, deduped, never fatal
# --------------------------------------------------------------------------

class Uploader:
    """One background thread, one buffer keyed by repo path, one commit/cycle.

    A rolling checkpoint enqueued five times in one window produces ONE file in
    ONE commit -- create_commit with many operations is ONE rate-limit op.
    """

    def __init__(self, repo_id: str, token: str | None, repo_type: str = "dataset",
                 interval_s: int = 1800, rate_limit: int = 25, enabled: bool = True):
        self.repo_id = repo_id
        self.token = token
        self.repo_type = repo_type
        self.interval_s = int(interval_s)
        self.enabled = bool(enabled and token)
        self.limiter = SharedRateLimiter.for_token(token, rate_limit)

        self._buffer: dict[str, tuple[str, str]] = {}
        self._pushed: set[str] = set()
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._api = None
        self.commits = 0
        self.failures = 0
        self.last_push_ts: float | None = None
        self.bytes_pushed = 0

        if self.enabled:
            try:
                from huggingface_hub import HfApi
                self._api = HfApi(token=token)
                self._api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=True)
                who = self._api.whoami().get("name", "?")
                _print("HF", f"authenticated as {who}  ->  {repo_type}:{repo_id}")
                _print("HF", f"rate cap {self.limiter.limit}/hr (shared across all workers on this token)")
            except Exception as e:
                _print("HF", f"DISABLED -- {type(e).__name__}: {e}")
                self.enabled = False
        else:
            _print("HF", "DISABLED -- no token; running local-only")

    # -- public -----------------------------------------------------------
    def start(self) -> None:
        if not self.enabled or self._thread:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="uploader")
        self._thread.start()
        _print("HF", f"background uploader started ({self.interval_s//60} min cycle)")

    def enqueue(self, local_path, repo_path: str, force: bool = False) -> bool:
        p = Path(local_path)
        if not p.exists():
            return False
        try:
            st = p.stat()
            fp = f"{repo_path}|{st.st_size}|{st.st_mtime_ns}"
        except OSError:
            return False
        with self._lock:
            if not force and fp in self._pushed:
                return False                       # unchanged file -- free skip
            self._buffer[repo_path] = (str(p), fp)
        return True

    def enqueue_dir(self, local_dir, repo_prefix: str, patterns=("*",), force=False) -> int:
        n = 0
        base = Path(local_dir)
        if not base.exists():
            return 0
        for pat in patterns:
            for f in base.rglob(pat):
                if f.is_file():
                    rel = f.relative_to(base).as_posix()
                    n += bool(self.enqueue(f, f"{repo_prefix}/{rel}", force=force))
        return n

    def flush(self, timeout: float = 1800, reason: str = "manual") -> bool:
        """Push everything pending NOW and block until done."""
        if not self.enabled:
            return True
        with self._lock:
            pending = len(self._buffer)
        if pending == 0:
            return True
        _print("HF", f"flush ({reason}): {pending} file(s)")
        return self._push_batch(blocking=True, timeout=timeout)

    def stop(self) -> None:
        self._stop.set()
        self._wakeup.set()
        if self._thread:
            self._thread.join(timeout=10)

    def verify_present(self, repo_paths: list[str]) -> list[str]:
        """A flush that did not time out is NOT evidence the files arrived.
        Ask the repository."""
        if not self.enabled:
            return []
        try:
            files = set(self._api.list_repo_files(self.repo_id, repo_type=self.repo_type))
            return [p for p in repo_paths if p not in files]
        except Exception as e:
            _print("HF", f"verify failed: {e}")
            return list(repo_paths)

    # -- internals --------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wakeup.wait(timeout=self.interval_s)
            self._wakeup.clear()
            if self._stop.is_set():
                break
            with self._lock:
                if not self._buffer:
                    continue
            self._push_batch(blocking=False)

    def _push_batch(self, blocking: bool, timeout: float = 1800) -> bool:
        from huggingface_hub import CommitOperationAdd
        with self._lock:
            batch, self._buffer = dict(self._buffer), {}
        if not batch:
            return True

        ops, fps, total = [], {}, 0
        for repo_path, (local, fp) in batch.items():
            if not Path(local).exists():
                continue
            ops.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local))
            fps[repo_path] = fp
            total += Path(local).stat().st_size
        if not ops:
            return True

        deadline = now() + timeout
        for attempt in range(5):
            if not self.limiter.wait_for_slot(self._stop if not blocking else None):
                break
            try:
                t0 = now()
                self._api.create_commit(
                    repo_id=self.repo_id, repo_type=self.repo_type, operations=ops,
                    commit_message=f"{len(ops)} file(s) @ {iso()}")
                self.commits += 1
                self.bytes_pushed += total
                self.last_push_ts = now()
                with self._lock:
                    self._pushed.update(fps.values())
                _print("HF", f"commit #{self.commits}: {len(ops)} file(s), "
                             f"{total/1e6:.1f} MB, {now()-t0:.1f}s  "
                             f"[{self.limiter.count_last_hour()}/{self.limiter.limit} this hr]")
                # huggingface_hub/LFS can leave large, now-free upload arenas
                # mapped in a long-lived Kaggle process.  Trim after the batch
                # so those buffers cannot accumulate into a host-RAM kill.
                release_host_memory()
                return True
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                if any(k in msg.lower() for k in ("401", "403", "unauthorized", "forbidden")):
                    _print("HF", f"AUTH FAILURE -- not retrying. {msg}")
                    self.enabled = False
                    break                          # a read-only token never becomes writable
                wait = parse_retry_after(msg) or min(80.0, 5.0 * (2 ** attempt))
                self.failures += 1
                _print("HF", f"push failed (attempt {attempt+1}/5), retry in {wait:.0f}s -- {msg[:160]}")
                if now() + wait > deadline:
                    break
                time.sleep(wait)

        # failed: put it back, without clobbering anything newer that arrived
        with self._lock:
            for repo_path, val in batch.items():
                self._buffer.setdefault(repo_path, val)
        _print("HF", f"batch returned to buffer ({len(batch)} files) -- training continues")
        release_host_memory()
        return False


# --------------------------------------------------------------------------
# 3. Registry -- ONE SHARD PER WRITER, merged on read  (Bug 2)
# --------------------------------------------------------------------------

class Registry:
    """HuggingFace has no append operation.

    Every worker appending to a shared runs.jsonl and pushing means the last
    push silently destroys every other worker's lines. No error -- the file
    just forgets. And since work planning reads COMPLETION from the ledger, a
    lost 'completed' entry makes a finished 3-hour run look unfinished and
    someone retrains it.

    So: each writer owns one file nobody else touches. Reads merge all shards.
    """

    def __init__(self, local_dir: Path, uploader: Uploader | None,
                 account: str, worker_id: int, session_id: str):
        self.dir = Path(local_dir) / "registry" / "events"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.uploader = uploader
        self.shard_name = f"{account}_w{worker_id}_{session_id}.jsonl"
        self.shard = self.dir / self.shard_name
        self.shard.touch()
        self._lock = threading.Lock()

    def emit(self, run_id: str, state: str, **extra) -> None:
        rec = {"ts": now(), "iso": iso(), "run_id": run_id, "state": state, **extra}
        with self._lock:
            with open(self.shard, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        if self.uploader:
            # force=True: the shard changes every write, so the mtime dedup
            # would otherwise skip it inside one push window
            self.uploader.enqueue(self.shard, f"registry/events/{self.shard_name}", force=True)

    def entries(self) -> list[dict]:
        out = []
        for p in sorted(self.dir.glob("*.jsonl")):
            try:
                for line in p.read_text().splitlines():
                    if line.strip():
                        out.append(json.loads(line))
            except Exception:
                continue
        out.sort(key=lambda e: float(e.get("ts", 0.0)))
        return out

    def latest(self) -> dict[str, dict]:
        st: dict[str, dict] = {}
        for e in self.entries():
            rid = e.get("run_id")
            if not rid:
                continue
            # 'completed' is STICKY. A late heartbeat from a stale shard must
            # not resurrect a finished run, or it gets trained a second time.
            if st.get(rid, {}).get("state") == "completed" and e.get("state") != "completed":
                continue
            st[rid] = e
        return st

    def pull(self, uploader: Uploader) -> int:
        """Download every other worker's shards."""
        if not uploader.enabled:
            return 0
        try:
            from huggingface_hub import hf_hub_download
            files = [f for f in uploader._api.list_repo_files(uploader.repo_id, repo_type=uploader.repo_type)
                     if f.startswith("registry/events/") and f.endswith(".jsonl")]
            n = 0
            for f in files:
                if Path(f).name == self.shard_name:
                    continue                       # never overwrite our own live shard
                try:
                    p = hf_hub_download(uploader.repo_id, f, repo_type=uploader.repo_type,
                                        token=uploader.token, local_dir=str(self.dir.parent.parent))
                    n += 1
                except Exception:
                    continue
            return n
        except Exception as e:
            _print("REG", f"pull failed: {e}")
            return 0

    def can_claim(self, run_id: str, account: str, stale_s: float = 7200) -> tuple[bool, str]:
        """Bug 3: check OWNER before freshness. The most common case -- my
        session died and this is the new one -- must be the easy path."""
        st = self.latest().get(run_id)
        if st is None:
            return True, "unclaimed"
        if st["state"] == "completed":
            return False, "already completed"
        if st.get("account") == account:
            return True, "own run -- resuming"
        age = now() - float(st.get("ts", 0))
        # A recent failure/paused event is also evidence that the assigned
        # account is alive and about to retry.  The old test protected only
        # running/claimed events, so every other worker immediately stole the
        # failed run and several Kaggle notebooks converged on the same model.
        if age < stale_s:
            return False, (f"recent {st.get('state')} by {st.get('account')} "
                           f"({age/60:.0f} min ago)")
        return True, f"stale ({age/3600:.1f} h) -- stealing"


# --------------------------------------------------------------------------
# 3b. RemoteInventory -- what the REPOSITORY holds        (Bug 8, Bug 9)
# --------------------------------------------------------------------------

class RemoteInventory:
    """The registry records intentions. This records facts.

    Every field in the registry is relative to a session: which account
    claimed a run, which worker id, how many workers were configured. Change
    NUM_WORKERS from 4 to 1 and the ownership arithmetic reshuffles. Run on a
    different account and `can_claim` no longer recognises the run as yours.
    Lose a shard and a finished run looks unfinished.

    `runs/<run_id>/STATUS.json` has none of those problems. It either says
    epoch 34 or it does not, and it says the same thing to every worker on
    every account at every value of NUM_WORKERS. So:

        WORK PLANNING READS THIS.
        The registry is demoted to the one thing it is good at -- telling you
        whether somebody else is training this run *right now*.

    That is what "the workers concept is universal" means concretely: a run's
    state is a property of the run, not of who is looking at it.

    Bug 8 -- and this is the one that cost ten hours: `Trainer.try_resume`
    only ever looked at the LOCAL checkpoint. Kaggle wipes the session disk,
    so in a fresh session there is never a local checkpoint, so every run
    restarted at epoch 1 no matter how far it had got. The checkpoints were
    on HuggingFace the whole time. Nothing ever fetched them back.
    """

    TERMINAL_OK = "completed"

    def __init__(self, uploader, stage_dir: Path):
        self.uploader = uploader
        self.stage_dir = Path(stage_dir)
        self.files: set[str] = set()
        self.status: dict[str, dict] = {}
        self.fetched_at: float = 0.0

    # -- reading ----------------------------------------------------------
    def refresh(self, run_ids=None, verbose: bool = True) -> "RemoteInventory":
        """One listing call, then one tiny JSON per run that has one.

        `run_ids` narrows the STATUS.json downloads, not the listing. Statuses
        outside the narrowed set are kept, so `refresh([one_run])` is a cheap
        re-check of a single run just before starting it -- which is how a
        second worker finding out it was beaten to a run costs two requests
        instead of thirty-six.
        """
        self.files = set()
        if run_ids is None:
            self.status = {}
        if not self.uploader.enabled:
            if verbose:
                _print("INV", "HuggingFace off -- remote inventory empty")
            return self
        try:
            self.files = set(self.uploader._api.list_repo_files(
                self.uploader.repo_id, repo_type=self.uploader.repo_type))
        except Exception as e:
            _print("INV", f"listing failed ({type(e).__name__}: {e}) -- "
                          "falling back to the registry alone")
            return self

        present = {p.split("/")[1] for p in self.files
                   if p.startswith("runs/") and len(p.split("/")) > 2}
        want = present if run_ids is None else (present & set(run_ids))

        from huggingface_hub import hf_hub_download
        for rid in sorted(want):
            rp = f"runs/{rid}/STATUS.json"
            if rp not in self.files:
                continue
            try:
                p = hf_hub_download(self.uploader.repo_id, rp,
                                    repo_type=self.uploader.repo_type,
                                    token=self.uploader.token,
                                    local_dir=str(self.stage_dir))
                self.status[rid] = json.loads(Path(p).read_text())
            except Exception:
                continue
        self.fetched_at = now()
        if verbose:
            n_done = sum(1 for r in want if self.state(r) == "completed")
            n_res = sum(1 for r in want if self.state(r) == "resumable")
            scope = "in this notebook" if run_ids is not None else "in the whole repository"
            _print("INV", f"repository holds {len(present)} run(s); of the {len(want)} "
                          f"{scope}: {n_done} finished, {n_res} resumable")
        return self

    def has_ckpt(self, run_id: str) -> bool:
        return f"runs/{run_id}/checkpoints/ckpt_last.pt" in self.files

    def epoch(self, run_id: str) -> int:
        st = self.status.get(run_id, {})
        for k in ("epoch", "epochs_trained"):
            with contextlib.suppress(Exception):
                v = st.get(k)
                if v is not None:
                    return int(v)
        return 0

    def state(self, run_id: str) -> str:
        """'completed' | 'resumable' | 'absent'.

        Note what is NOT here: 'failed'. A run that raised at epoch 47 has a
        checkpoint at epoch 47, so it is resumable -- the same as one the
        watchdog paused. Treating 'failed' as a state to be re-run from
        scratch is how twenty-six runs got thrown away.
        """
        st = self.status.get(run_id, {})
        if st.get("status") == self.TERMINAL_OK:
            return "completed"
        if self.has_ckpt(run_id):
            return "resumable"
        return "absent"

    def reason(self, run_id: str) -> str:
        s = self.state(run_id)
        if s == "completed":
            return "finished"
        if s == "resumable":
            st = self.status.get(run_id, {})
            was = st.get("status", "interrupted")
            ep = self.epoch(run_id)
            with contextlib.suppress(Exception):
                planned = int(st.get("of", st.get("epochs_planned")))
                if planned > 0 and ep >= planned:
                    return f"finalise {ep}-epoch checkpoint (status was {was})"
            return f"resume from epoch {ep+1} (was {was})"
        return "not started"

    # -- writing back to the session disk ---------------------------------
    def fetch_run(self, run_id: str, verbose: bool = True) -> bool:
        """Bring a run's checkpoint and history back onto this machine.

        Without this, resume works only inside one Kaggle session, which is
        the same as not working.
        """
        if not (self.uploader.enabled and self.has_ckpt(run_id)):
            return False
        from huggingface_hub import hf_hub_download
        wanted = [f"runs/{run_id}/checkpoints/ckpt_last.pt",
                  f"runs/{run_id}/checkpoints/ckpt_best.pt",
                  f"runs/{run_id}/metrics/epochs.csv"]
        got = 0
        for rp in wanted:
            if rp not in self.files:
                continue
            try:
                hf_hub_download(self.uploader.repo_id, rp,
                                repo_type=self.uploader.repo_type,
                                token=self.uploader.token,
                                local_dir=str(self.stage_dir))
                got += 1
            except Exception as e:
                _print("INV", f"could not fetch {rp}: {type(e).__name__}: {e}")
        if got and verbose:
            _print("INV", f"{run_id}: pulled {got} file(s) from HuggingFace "
                          f"-- resuming at epoch {self.epoch(run_id)+1}")
        return got > 0

    def qwk(self, run_id: str):
        """`best_qwk` in a running STATUS.json, `best_val_qwk` in a finished
        one -- the summary is merged in at the end under a different name."""
        st = self.status.get(run_id, {})
        for k in ("best_qwk", "best_val_qwk"):
            v = st.get(k)
            if v is not None:
                with contextlib.suppress(Exception):
                    return round(float(v), 4)
        return NA

    def table(self, run_ids) -> pd.DataFrame:
        return pd.DataFrame([{"run_id": r, "state": self.state(r),
                              "epoch": self.epoch(r),
                              "status_file": self.status.get(r, {}).get("status", NA),
                              "best_qwk": self.qwk(r)}
                             for r in sorted(run_ids)])


# --------------------------------------------------------------------------
# 4. Sharding -- LPT bin packing on a STATIC cost table  (Bug 7)
# --------------------------------------------------------------------------

# Minutes per single run (1 fold, 1 seed, full epoch budget).
# Derived from measured T4 throughput scaled by relative FLOPs and resolution.
# CALIBRATE ONCE against two real runs, then FREEZE. Measurements refine the
# PRINTED plan only -- never the assignment, or two workers disagree about
# what they own and a job is trained twice while another is abandoned.
STATIC_COST_HINTS: dict[str, float] = {
    "mobilenetv4": 11, "swin_t": 12, "coatnet0": 13, "swin_s": 21,
    "regnety016": 24, "vit_s": 26, "deit3_s": 26, "resnet50": 27,
    "effnetv2s": 29, "dinov2_s": 30, "resnext50": 32, "convnextv2_t": 34,
    "densenet121": 37, "bcnn": 50, "convnextv2_s": 55, "hbp": 55,
    "csab": 55, "vgg16bn": 61, "coarse2fine": 61, "clip_b16": 69,
    "siglip_b16": 69, "maxvit_t": 72, "dinov2_b": 72, "resnet18": 12,
}
DEFAULT_COST = 30.0


def cost_of(run_id: str, costs: dict[str, float] | None = None) -> float:
    table = costs or STATIC_COST_HINTS
    for arch, c in sorted(table.items(), key=lambda kv: -len(kv[0])):
        if f"-{arch}-" in run_id:
            return float(c)
    return DEFAULT_COST


def assign_workers(run_ids, n_workers: int, mode: str = "cost",
                   costs: dict | None = None) -> dict[str, int]:
    ids = sorted(run_ids)                          # canonical order on every machine
    if n_workers <= 1:
        return {r: 0 for r in ids}
    if mode == "hash":
        return {r: int(hashlib.sha256(r.encode()).hexdigest(), 16) % n_workers for r in ids}
    if mode == "balanced":
        return {r: i % n_workers for i, r in enumerate(ids)}
    jobs = sorted(ids, key=lambda r: (-cost_of(r, costs), r))
    load, out = [0.0] * n_workers, {}
    for r in jobs:
        w = int(np.argmin(load))
        out[r] = w
        load[w] += cost_of(r, costs)
    return out


def shard_report(run_ids, n_workers: int, mode: str = "cost",
                 display_costs: dict | None = None) -> pd.DataFrame:
    owner = assign_workers(run_ids, n_workers, mode)       # STATIC table only
    rows = []
    for w in range(n_workers):
        mine = [r for r in run_ids if owner[r] == w]
        hrs = sum(cost_of(r, display_costs) for r in mine) / 60.0
        rows.append({"worker": w, "runs": len(mine), "est_hours": round(hrs, 2)})
    df = pd.DataFrame(rows)
    if len(df) and df.est_hours.min() > 0:
        df.attrs["imbalance"] = round(df.est_hours.max() / df.est_hours.min(), 2)
    return df


def estimate_phase(run_ids, num_workers: int = 1, display_costs: dict | None = None) -> dict:
    total_min = sum(cost_of(r, display_costs) for r in run_ids)
    owner = assign_workers(run_ids, num_workers, "cost")
    per = [sum(cost_of(r, display_costs) for r in run_ids if owner[r] == w) / 60.0
           for w in range(num_workers)]
    wall = max(per) if per else 0.0
    measured = set((display_costs or {}).keys()) - set()
    archs = {a for a in STATIC_COST_HINTS if any(f"-{a}-" in r for r in run_ids)}
    frac = len(archs & measured) / max(1, len(archs)) if display_costs else 0.0
    return {"n_runs": len(run_ids), "total_gpu_hours": total_min / 60.0,
            "wall_clock_hours": wall, "per_worker_hours": per,
            "sessions_needed": max(1, math.ceil(wall / 8.5)),
            "frac_measured": frac}


# --------------------------------------------------------------------------
# 5. Lifecycle guards -- all four ways a session ends
# --------------------------------------------------------------------------

class LifecycleGuard:
    """Kaggle usually sends SIGTERM. Catching only KeyboardInterrupt misses the
    platform kill entirely -- which is how you lose the last 30 minutes of a
    3-hour run."""

    def __init__(self, on_flush, session_limit_h: float = 8.5):
        self.on_flush = on_flush
        self.session_limit_s = session_limit_h * 3600
        self.t_start = now()
        self._fired = threading.Event()
        self._orig_term = None
        self._orig_int = None

    def install(self):
        with contextlib.suppress(Exception):
            self._orig_term = signal.signal(signal.SIGTERM, self._handle)
        with contextlib.suppress(Exception):
            self._orig_int = signal.signal(signal.SIGINT, self._handle)
        atexit.register(self._atexit)
        _print("LIFE", f"guards installed (SIGTERM, SIGINT, atexit, watchdog @ {self.session_limit_s/3600:.1f} h)")
        return self

    def _handle(self, signum, frame):
        self._fire(f"signal {signum}")
        if signum == signal.SIGINT:
            raise KeyboardInterrupt

    def _atexit(self):
        self._fire("atexit")

    def _fire(self, reason: str):
        if self._fired.is_set():
            return                                  # exactly once
        self._fired.set()
        _print("LIFE", f"flush triggered by {reason}")
        with contextlib.suppress(Exception):
            self.on_flush(reason)

    def reset(self):
        self._fired.clear()

    @property
    def elapsed_h(self) -> float:
        return (now() - self.t_start) / 3600

    def near_limit(self, margin_min: float = 20) -> bool:
        return (now() - self.t_start) > (self.session_limit_s - margin_min * 60)


# --------------------------------------------------------------------------
# 6. Telemetry -- record everything, because we train once
# --------------------------------------------------------------------------

CARBON_INTENSITY_G_PER_KWH = 713.0     # India grid average; recorded for reproducibility
HOST_RAM_PAUSE_PERCENT = 88.0          # checkpoint + push before Kaggle's OOM killer
HOST_RAM_RESUME_PERCENT = 80.0         # ...and carry on once the arenas come back
RAM_GUARD_REVISION = "2026-09-01-r2"


def container_memory() -> tuple[float, float, str]:
    """(used_bytes, limit_bytes, source) for the memory the OOM killer counts.

    ⚠ Bug 25. `psutil.virtual_memory()` reads `/proc/meminfo`, which inside a
    container reports the **host's** memory, not the cgroup limit the kernel
    actually enforces on us. So the percentage the guard was pausing on did not
    describe our own budget at all, and on a busy host it can sit near 90% no
    matter what this notebook does.

    The cgroup files are the number Kaggle's OOM killer uses. Read those and
    fall back to psutil only when they are absent.
    """
    for cur, mx in ((Path("/sys/fs/cgroup/memory.current"),
                     Path("/sys/fs/cgroup/memory.max")),                    # v2
                    (Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
                     Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))): # v1
        try:
            used = float(cur.read_text().strip())
            raw = mx.read_text().strip()
            limit = float("inf") if raw == "max" else float(raw)
            # An unset v1 limit is a huge sentinel, not a real budget.
            if limit and limit < 2**62:
                return used, limit, f"cgroup:{cur.parent.name or 'v2'}"
        except Exception:
            continue
    try:
        import psutil
        vm = psutil.virtual_memory()
        return float(vm.total - vm.available), float(vm.total), "psutil(host)"
    except Exception:
        return 0.0, 0.0, "unavailable"


def memory_report() -> dict:
    """Where the memory actually is. Printed per epoch so a pause is explainable
    instead of being one number nobody can act on."""
    used, limit, src = container_memory()
    out = {"used_gb": used / 1e9, "limit_gb": limit / 1e9, "source": src,
           "percent": (100.0 * used / limit) if limit else 0.0,
           "proc_rss_gb": 0.0, "children_rss_gb": 0.0, "n_children": 0}
    try:
        import psutil
        me = psutil.Process()
        out["proc_rss_gb"] = me.memory_info().rss / 1e9
        kids = me.children(recursive=True)
        out["n_children"] = len(kids)
        tot = 0.0
        for k in kids:
            with contextlib.suppress(Exception):
                tot += k.memory_info().rss / 1e9
        out["children_rss_gb"] = tot
    except Exception:
        pass
    return out


def host_ram_percent() -> float:
    """Memory in use RIGHT NOW as a percentage of the enforced limit.

    Uses the cgroup budget when there is one (Bug 25), so this is the same
    number the OOM killer is watching rather than the host's.

    ⚠ Bug 22. The guard used to read `ram_percent_peak` -- the MAXIMUM of the
    1 Hz samples taken during the epoch. Serialising a 300 MB checkpoint and
    handing it to the HuggingFace uploader spikes RSS for a second or two, and
    that spike alone crossed 88%. The run was then paused, and because a pause
    stops the whole worker, one transient buffer ended an eight-hour session
    with eighteen runs untouched.

    A peak answers "did we ever come close?". The question that matters before
    starting another epoch is "is there room now?" -- after the buffers have
    been freed and the arenas returned to the kernel. That is this.
    """
    used, limit, _ = container_memory()
    return (100.0 * used / limit) if limit else 0.0


def host_ram_headroom(release: bool = True) -> tuple[float, float]:
    """(percent_before, percent_after_release). Cheap; call it per epoch."""
    before = host_ram_percent()
    if release:
        release_host_memory()
    return before, host_ram_percent()
MEMORY_SAFETY_REVISION = "2026-08-31-r2"
CUDA_SAFETY_REVISION = "2026-08-31-r1"
SCHEDULER_SAFETY_REVISION = "2026-08-31-r2"
HF_COMMIT_POLICY_REVISION = "2026-08-31-r1"
EPOCH_HISTORY_SCHEMA_REVISION = "2026-09-01-r1"
PROCESS_ISOLATION_REVISION = "2026-09-03-r1"

# PyTorch 2.10.0+cu128 on Kaggle's T4 image reproducibly failed in the first
# RegNetY-16GF ROI batch when AMP, DataParallel, cuDNN autotuning, and NHWC
# (channels_last) were combined.  Two independent public runs failed in s2.conv
# with CUDNN_STATUS_EXECUTION_FAILED / CUDA misaligned-address while each GPU
# held only ~1.1 GB, so this is not an OOM and changing the model or batch is the
# wrong repair.  Keep the exact model/config/checkpoint format, but use cuDNN's
# conservative NCHW path for this architecture.  Other completed architectures
# keep the Stage-A channels_last path.
CUDA_CONTIGUOUS_ARCHS = frozenset({"regnety016"})
_FATAL_CUDA_MARKERS = (
    "misaligned address", "illegal memory access", "device-side assert",
    "cudnn_status_execution_failed", "unspecified launch failure",
)


def training_memory_format(arch: str) -> str:
    """Runtime tensor layout; deliberately excluded from scientific config."""
    return "contiguous" if arch in CUDA_CONTIGUOUS_ARCHS else "channels_last"


def fatal_cuda_error(exc: BaseException) -> bool:
    """Whether the CUDA context must be discarded before another run."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _FATAL_CUDA_MARKERS)


class HardwareMonitor:
    """Samples GPU power/util/temp/clocks and host CPU/RAM in the background.

    Per DEVICE, never aggregated: train on one of two GPUs and an aggregate
    reports ~50% utilisation, hiding that half the allocation is idle.
    """

    def __init__(self, out_dir: Path, gpu_hz: float = 10.0, sys_hz: float = 1.0):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_dt = 1.0 / gpu_hz
        self.sys_dt = 1.0 / sys_hz
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self.samples: list[dict] = []
        self.energy_rows: list[dict] = []
        self._energy_j = defaultdict(float)
        self._nvml = None
        self._handles = []
        self._psutil = None
        self._proc = None
        self.available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                             for i in range(pynvml.nvmlDeviceGetCount())]
            self.available = True
        except Exception:
            pass
        try:
            import psutil
            self._psutil = psutil
            self._proc = psutil.Process()
        except Exception:
            pass

    def gpu_static(self) -> dict:
        out = {}
        if not self._nvml:
            return out
        for i, h in enumerate(self._handles):
            with contextlib.suppress(Exception):
                name = self._nvml.nvmlDeviceGetName(h)
                out[f"gpu{i}_name"] = name.decode() if isinstance(name, bytes) else name
                out[f"gpu{i}_mem_total_mb"] = self._nvml.nvmlDeviceGetMemoryInfo(h).total / 1e6
                out[f"gpu{i}_power_limit_w"] = self._nvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000
                out[f"gpu{i}_uuid"] = self._nvml.nvmlDeviceGetUUID(h)
        with contextlib.suppress(Exception):
            v = self._nvml.nvmlSystemGetDriverVersion()
            out["gpu_driver"] = v.decode() if isinstance(v, bytes) else v
        return out

    def start(self):
        if not (self.available or self._psutil):
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True, name="hwmon")
        self._thread.start()
        return self

    def _loop(self):
        t_last_sys = 0.0
        t_prev = now()
        while not self._stop.is_set():
            t = now()
            dt = t - t_prev
            t_prev = t
            row = {"ts": t}
            if self._nvml:
                for i, h in enumerate(self._handles):
                    try:
                        pw = self._nvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                        self._energy_j[i] += pw * dt
                        u = self._nvml.nvmlDeviceGetUtilizationRates(h)
                        mem = self._nvml.nvmlDeviceGetMemoryInfo(h)
                        # UNDER THE LOCK. Bug 12: this append used to be
                        # unsynchronised, so `dump()` could hold the lock and
                        # still have the list grow underneath pandas.
                        with self._lock:
                            self.energy_rows.append({
                                "ts": t, "gpu_index": i, "power_w": pw,
                                "energy_joules_cumulative": self._energy_j[i],
                                "temp_c": self._nvml.nvmlDeviceGetTemperature(h, 0),
                                "util_pct": u.gpu})
                        if t - t_last_sys >= self.sys_dt:
                            row.update({
                                f"gpu{i}_util": u.gpu, f"gpu{i}_mem_util": u.memory,
                                f"gpu{i}_mem_used_mb": mem.used / 1e6,
                                f"gpu{i}_temp_c": self._nvml.nvmlDeviceGetTemperature(h, 0),
                                f"gpu{i}_power_w": pw,
                                f"gpu{i}_sm_clock": self._nvml.nvmlDeviceGetClockInfo(h, 0),
                                f"gpu{i}_mem_clock": self._nvml.nvmlDeviceGetClockInfo(h, 2),
                                f"gpu{i}_throttle": self._nvml.nvmlDeviceGetCurrentClocksThrottleReasons(h),
                            })
                    except Exception:
                        continue
            if self._psutil and t - t_last_sys >= self.sys_dt:
                with contextlib.suppress(Exception):
                    vm = self._psutil.virtual_memory()
                    row.update({"cpu_percent": self._psutil.cpu_percent(interval=None),
                                "ram_used_gb": vm.used / 1e9,
                                "ram_percent": vm.percent,
                                "proc_rss_gb": self._proc.memory_info().rss / 1e9,
                                "proc_vms_gb": self._proc.memory_info().vms / 1e9,
                                "swap_gb": self._psutil.swap_memory().used / 1e9})
            if t - t_last_sys >= self.sys_dt:
                with self._lock:
                    self.samples.append(row)
                t_last_sys = t
            self._stop.wait(self.gpu_dt)

    def window(self, t0: float, t1: float) -> dict:
        """Aggregate everything sampled inside [t0, t1] into epoch columns.

        Same rule as `dump()`: an observer must not be able to fail the run it
        is observing. A missing telemetry block costs some columns in one row
        of epochs.csv; an exception here costs the epoch.
        """
        try:
            return self._window(t0, t1)
        except Exception as e:
            _print("HWMON", f"telemetry window failed ({type(e).__name__}: {e}) "
                            "-- epoch recorded without hardware columns")
            return {}

    def _window(self, t0: float, t1: float) -> dict:
        with self._lock:
            rows = [r for r in self.samples if t0 <= r["ts"] <= t1]
            erows = [r for r in self.energy_rows if t0 <= r["ts"] <= t1]
        out: dict = {}
        if not rows and not erows:
            return out
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        n_gpu = len(self._handles)
        for i in range(n_gpu):
            def col(name, agg="mean"):
                c = f"gpu{i}_{name}"
                if c not in df or df[c].dropna().empty:
                    return NA
                return float(getattr(df[c].dropna(), agg)())
            out[f"gpu{i}_util_mean"] = col("util")
            out[f"gpu{i}_util_max"] = col("util", "max")
            out[f"gpu{i}_util_p50"] = float(df[f"gpu{i}_util"].dropna().median()) if f"gpu{i}_util" in df and not df[f"gpu{i}_util"].dropna().empty else NA
            out[f"gpu{i}_mem_used_mb_mean"] = col("mem_used_mb")
            out[f"gpu{i}_mem_used_mb_peak"] = col("mem_used_mb", "max")
            out[f"gpu{i}_temp_c_mean"] = col("temp_c")
            out[f"gpu{i}_temp_c_max"] = col("temp_c", "max")
            out[f"gpu{i}_power_w_mean"] = col("power_w")
            out[f"gpu{i}_power_w_max"] = col("power_w", "max")
            out[f"gpu{i}_sm_clock_mhz_mean"] = col("sm_clock")
            out[f"gpu{i}_mem_clock_mhz_mean"] = col("mem_clock")
            # non-zero means the card clocked down -- otherwise a slow epoch is
            # a permanent mystery
            out[f"gpu{i}_throttle_reasons"] = col("throttle", "max")
            ei = [r for r in erows if r["gpu_index"] == i]
            out[f"gpu{i}_energy_joules_epoch"] = (ei[-1]["energy_joules_cumulative"] - ei[0]["energy_joules_cumulative"]) if len(ei) > 1 else NA
            out[f"gpu{i}_energy_joules_cumulative"] = ei[-1]["energy_joules_cumulative"] if ei else NA
        if not df.empty:
            for src, dst, agg in [("cpu_percent", "cpu_percent_mean", "mean"),
                                  ("cpu_percent", "cpu_percent_max", "max"),
                                  ("ram_used_gb", "ram_used_gb_mean", "mean"),
                                  ("ram_used_gb", "ram_used_gb_peak", "max"),
                                  ("ram_percent", "ram_percent_peak", "max"),
                                  ("proc_rss_gb", "proc_rss_gb_mean", "mean"),
                                  ("proc_rss_gb", "proc_rss_gb_peak", "max"),
                                  ("proc_vms_gb", "proc_vms_gb_peak", "max"),
                                  ("swap_gb", "swap_used_gb_peak", "max")]:
                out[dst] = float(getattr(df[src].dropna(), agg)()) if src in df and not df[src].dropna().empty else NA
        ej = sum(v for k, v in out.items() if k.endswith("_energy_joules_epoch") and v != NA)
        out["energy_joules_epoch"] = ej
        out["energy_wh_epoch"] = ej / 3600.0
        out["co2_g_epoch"] = (ej / 3.6e6) * CARBON_INTENSITY_G_PER_KWH
        out["carbon_intensity_g_per_kwh"] = CARBON_INTENSITY_G_PER_KWH
        out["power_sample_count"] = len(erows)
        return out

    def dump(self):
        """Write the sample buffers to disk.

        ⚠ Bug 12 -- this crashed two runs after 43 and 66 minutes of training:

            ValueError: Length of values (35249) does not match length of index (35250)

        `pd.DataFrame(list_of_dicts)` walks the list while building columns. The
        10 Hz sampler thread appended one more row midway, so the last column
        came out one element short. The lock was already held here, but the
        sampler's append was NOT synchronised, so holding it achieved nothing.

        Two changes, and the second matters more than the first:

          1. Copy the buffers under the lock, build the DataFrames outside it.
             Correct, and it also stops a slow gzip write from stalling the
             sampler for a second.

          2. **Never raise.** Telemetry is an observer. An observer that can
             kill a three-hour training run is a liability, however good its
             data is. Losing a power trace is a nuisance; losing the run is not.

        ⚠ Bug 23 -- and this one grew until the kernel was killed.

        The buffers were snapshotted and rewritten in full every ten epochs,
        and **never cleared**. At 10 Hz per GPU a four-hour run accumulates
        roughly 300,000 dicts, and every dump rebuilt a DataFrame over all of
        them. Public NB06 telemetry shows host RSS climbing +0.54 GB per epoch,
        3.5 GB to 28 GB across one run, at which point Kaggle killed the kernel
        with no Python exception to catch.

        Now each dump writes only the rows added since the last one and then
        drops them. Concatenated gzip members are a valid gzip stream, so the
        file on disk still reads back as one table with `pd.read_csv`, while
        the process holds at most one dump-interval of samples.
        """
        try:
            with self._lock:
                erows, self.energy_rows = self.energy_rows, []
                srows, self.samples = self.samples, []
            for rows, name in ((erows, "energy_samples.csv.gz"),
                               (srows, "system_samples.csv.gz")):
                if not rows:
                    continue
                path = self.out_dir / name
                first = not path.exists()
                with gzip.open(path, "at", newline="") as fh:
                    pd.DataFrame(rows).to_csv(fh, index=False, header=first)
                del rows
            release_host_memory()
        except Exception as e:
            _print("HWMON", f"telemetry dump failed ({type(e).__name__}: {e}) "
                            "-- training continues, this epoch's trace is lost")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.dump()


# --------------------------------------------------------------------------
# 7. Metrics
# --------------------------------------------------------------------------

CLASSES = ["low_mileage_proxy", "mid_mileage_proxy", "high_mileage_proxy"]
CLASS_SHORT = ["low", "mid", "high"]
C2I = {c: i for i, c in enumerate(CLASSES)}


def quadratic_weighted_kappa(y_true, y_pred, n: int = 3) -> float:
    """The ORDINAL metric. Our classes are ordered, so confusing low<->high
    must cost more than low<->mid. Never report macro-F1 alone."""
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    if len(y_true) == 0:
        return float("nan")
    O = np.zeros((n, n))
    for a, b in zip(y_true, y_pred):
        O[a, b] += 1
    W = np.array([[((i - j) ** 2) / ((n - 1) ** 2) for j in range(n)] for i in range(n)])
    ha = np.bincount(y_true, minlength=n).astype(float)
    hb = np.bincount(y_pred, minlength=n).astype(float)
    E = np.outer(ha, hb)
    E = E * (O.sum() / max(E.sum(), 1e-12))
    den = (W * E).sum()
    return float(1.0 - (W * O).sum() / den) if den > 1e-12 else 0.0


def classification_report_dict(y_true, y_pred, probs=None, prefix="val_", n=3) -> dict:
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    out: dict = {}
    if len(y_true) == 0:
        return out, np.zeros((n, n), int)
    cm = np.zeros((n, n), int)
    for a, b in zip(y_true, y_pred):
        cm[a, b] += 1
    acc = float((y_true == y_pred).mean())
    precs, recs, f1s, sups = [], [], [], []
    for k in range(n):
        tp = cm[k, k]; fp = cm[:, k].sum() - tp; fn = cm[k, :].sum() - tp
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        precs.append(pr); recs.append(rc)
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
        sups.append(int(cm[k, :].sum()))
    out[prefix + "acc"] = acc
    out[prefix + "balanced_acc"] = float(np.mean([r for r, s in zip(recs, sups) if s > 0]) if any(sups) else 0.0)
    out[prefix + "f1_macro"] = float(np.mean(f1s))
    out[prefix + "f1_micro"] = acc
    tot = max(sum(sups), 1)
    out[prefix + "f1_weighted"] = float(sum(f * s for f, s in zip(f1s, sups)) / tot)
    out[prefix + "precision_macro"] = float(np.mean(precs))
    out[prefix + "recall_macro"] = float(np.mean(recs))
    for k, sh in enumerate(CLASS_SHORT[:n]):
        out[f"{prefix}f1_{sh}"] = float(f1s[k])
        out[f"{prefix}recall_{sh}"] = float(recs[k])
        out[f"{prefix}precision_{sh}"] = float(precs[k])
        out[f"{prefix}support_{sh}"] = sups[k]
    out[prefix + "qwk"] = quadratic_weighted_kappa(y_true, y_pred, n)
    out[prefix + "mae_class"] = float(np.abs(y_true - y_pred).mean())
    po = acc
    pe = float((np.bincount(y_true, minlength=n) * np.bincount(y_pred, minlength=n)).sum() / (len(y_true) ** 2))
    out[prefix + "cohen_kappa"] = float((po - pe) / (1 - pe)) if abs(1 - pe) > 1e-12 else 0.0
    t = cm.astype(float)
    c = np.trace(t); s = t.sum()
    pk = t.sum(0); tk = t.sum(1)
    num = c * s - (tk * pk).sum()
    den = math.sqrt(max((s ** 2 - (pk ** 2).sum()) * (s ** 2 - (tk ** 2).sum()), 0.0))
    out[prefix + "mcc"] = float(num / den) if den > 1e-12 else 0.0

    if probs is not None and len(probs):
        probs = np.asarray(probs, float)
        conf = probs.max(1)
        correct = (y_pred == y_true)
        eps = 1e-12
        out[prefix + "nll"] = float(-np.log(np.clip(probs[np.arange(len(y_true)), y_true], eps, 1)).mean())
        oh = np.eye(n)[y_true]
        out[prefix + "brier"] = float(((probs - oh) ** 2).sum(1).mean())
        out[prefix + "mean_confidence"] = float(conf.mean())
        out[prefix + "mean_confidence_correct"] = float(conf[correct].mean()) if correct.any() else NA
        out[prefix + "mean_confidence_incorrect"] = float(conf[~correct].mean()) if (~correct).any() else NA
        out[prefix + "overconfidence_gap"] = float(conf.mean() - acc)
        bins = np.linspace(0, 1, 16)
        ece = mce = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() == 0:
                continue
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += (m.sum() / len(conf)) * gap
            mce = max(mce, gap)
        out[prefix + "ece"] = float(ece)
        out[prefix + "mce"] = float(mce)
        out[prefix + "ace"] = float(ece)
    return out, cm


# --------------------------------------------------------------------------
# 8. Data
# --------------------------------------------------------------------------

def find_dataset_root(hint: str | None = None) -> Path | None:
    """Kaggle sometimes wraps an uploaded folder in an extra directory.
    Find the directory that actually contains images/, splits/ and manifests/."""
    cands = []
    if hint:
        cands.append(Path(hint))
    cands += [Path("/kaggle/input"), Path("/kaggle/temp/data"), Path.cwd()]
    for base in cands:
        if not base.exists():
            continue
        if (base / "images").is_dir() and (base / "splits").is_dir():
            return base
        for p in sorted(base.rglob("*")):
            if (p.is_dir() and (p / "images").is_dir()
                    and (p / "splits").is_dir() and (p / "manifests").is_dir()):
                return p
    return None


def find_annotations_root(data_root=None):
    """annotations/ is a SIBLING of FINAL/ inside the same uploaded package."""
    cands = []
    if data_root is not None:
        cands += [Path(data_root).parent / "annotations", Path(data_root) / "annotations"]
    cands += [Path("/kaggle/input")]
    for c in cands:
        if c.name == "annotations" and (c / "clean" / "masks").is_dir():
            return c
        if c.exists():
            for p in sorted(c.rglob("annotations")):
                if p.is_dir() and (p / "clean" / "masks").is_dir():
                    return p
    return None


def read_manifest(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    return df


def load_split(root: Path, fold: int):
    tr = read_manifest(root / f"splits/cv{fold}_train.csv")
    va = read_manifest(root / f"splits/cv{fold}_validation.csv")
    # The assertions that actually matter. A frame-level leak here would make
    # every number in the study meaningless, and it is silent.
    assert set(tr.session_group).isdisjoint(set(va.session_group)), "SESSION LEAK train/val"
    assert set(va.image_kind) == {"clean_original"}, "validation must be clean originals only"
    return tr, va


# `session_group` comes from a 12-second timestamp gap -- a PROXY for tyre
# identity, not a measurement. Photograph one tyre twice 20 s apart and it
# becomes two "sessions"; if they land in different folds the leak is silent.
# Found by scripts/tyre_identity_audit.py comparing tread pattern.
KNOWN_CROSS_FOLD_PAIRS = [
    ("mileage_070000__session_001", "mileage_090000__session_001", 0.90, "suspect"),
]


def split_health(tr, va, fold: int, verbose: bool = True) -> dict:
    """How many DISTINCT TYRES does this fold actually validate on?

    Image count is not the sample size. With ~1 tyre per class in validation, a
    model only has to tell three specific tyres apart -- a near-perfect score is
    the EXPECTED outcome, not evidence of learning wear.
    """
    per = va.groupby("proxy_label").session_group.nunique().to_dict()
    info = {"fold": fold, "val_images": len(va),
            "val_sessions": int(va.session_group.nunique()),
            "train_sessions": int(tr.session_group.nunique()),
            "val_sessions_per_class": {k: int(v) for k, v in per.items()},
            "cross_fold_tyre_flags": []}
    tr_s, va_s = set(tr.session_group), set(va.session_group)
    for a, b, ratio, verdict in KNOWN_CROSS_FOLD_PAIRS:
        if (a in tr_s and b in va_s) or (b in tr_s and a in va_s):
            info["cross_fold_tyre_flags"].append(
                {"train": a if a in tr_s else b, "val": b if b in va_s else a,
                 "ratio": ratio, "verdict": verdict})
    if verbose:
        _print("SPLIT", f"fold {fold}: {len(va)} val images from {info['val_sessions']} "
                        "sessions  " + "  ".join(
                            f"{k.replace('_mileage_proxy','')}={v}" for k, v in per.items()))
        if min(per.values(), default=9) <= 1:
            _print("SPLIT", "  ~1 tyre per class in validation -- a near-perfect score means "
                            "the model told 3 tyres apart, NOT that it learned wear")
        for f in info["cross_fold_tyre_flags"]:
            _print("SPLIT", f"  *** {f['verdict'].upper()} SAME TYRE ACROSS THE SPLIT "
                            f"(ratio {f['ratio']}) -- treat this fold as leak-inflated")
    return info


class TyreDataset:
    def __init__(self, df: pd.DataFrame, root: Path, tf, return_index=True,
                 roi_mode: str = "full_frame", annotation_roots=None):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.tf = tf
        self.return_index = return_index
        self.roi_mode = roi_mode
        self.annotation_roots = annotation_roots

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from PIL import Image
        r = self.df.iloc[i]
        # Always detach the converted image from its file handle.  The ROI
        # sweep opens every source image once per epoch; relying on PIL object
        # finalisation left thousands of mapped image buffers alive in long
        # Kaggle kernels.
        with Image.open(self.root / r.relative_path) as src:
            img = src.convert("RGB")
        if self.roi_mode == "tyre_crop":
            # We need only the non-background bounding box, not a dense mask
            # and not the coordinates of every tyre pixel.  The old
            # `np.where(mask > 0)` path allocated two full int64 coordinate
            # arrays per sample and the persistent/pinned loader retained RAM
            # across epochs (about 0.29 GB/epoch in the public NB06 traces).
            mp = mask_path(self.annotation_roots, r.image_id, r.image_kind)
            if not mp.exists():
                raise FileNotFoundError(f"ROI mask missing for {r.image_id}")
            with Image.open(mp) as mask_img:
                bbox = mask_img.getbbox()       # background is label 0
                mask_size = mask_img.size
            if bbox is None:
                raise ValueError(f"ROI mask contains no tyre pixels for {r.image_id}")
            # Five percent context avoids cutting the shoulder exactly at the
            # annotation boundary while still removing the frame-occupancy cue.
            x0, y0, x1, y1 = bbox
            # `getbbox` uses exclusive x1/y1. Subtract one here to reproduce
            # the old max-min padding exactly, so completed and future ROI
            # runs receive byte-for-byte-identical crop coordinates.
            pad = max(2, int(round(0.05 * max(y1 - y0 - 1, x1 - x0 - 1))))
            mw, mh = mask_size
            if img.size != mask_size:
                raise ValueError(
                    f"ROI image/mask size mismatch for {r.image_id}: "
                    f"image={img.size}, mask={mask_size}")
            cropped = img.crop((max(0, x0 - pad), max(0, y0 - pad),
                                min(mw, x1 + pad), min(mh, y1 + pad)))
            img.close()
            img = cropped
        try:
            x = self.tf(img)
        finally:
            img.close()
        y = C2I[r.proxy_label]
        return (x, y, i) if self.return_index else (x, y)


def build_transforms(img_size: int, train: bool, preprocessing: str = "raw"):
    import torchvision.transforms as T
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ops = []
    if preprocessing == "clahe":
        def _clahe(img):
            import cv2
            from PIL import Image
            a = np.asarray(img.convert("RGB"))
            lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
            lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
            return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
        ops.append(T.Lambda(_clahe))
    ops.append(T.Resize((img_size, img_size)))
    if preprocessing == "grayscale":
        ops.append(T.Grayscale(num_output_channels=3))   # a SHORTCUT TEST, not an improvement
    ops += [T.ToTensor(), T.Normalize(MEAN, STD)]
    # No stochastic augmentation anywhere: the derivatives are pre-generated by
    # the dataset package, and validation must never be augmented.
    return T.Compose(ops)


def build_loaders(root, tr_df, va_df, cfg):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    validate_config(cfg)
    ann = None
    if cfg.get("roi_mode", "full_frame") == "tyre_crop":
        ann = {"clean_masks": Path(cfg["clean_mask_root"]),
               "propagated_masks": Path(cfg["propagated_mask_root"])}
    tr_ds = TyreDataset(
        tr_df, root,
        build_transforms(cfg["input_resolution"], True, cfg.get("preprocessing", "raw")),
        roi_mode=cfg.get("roi_mode", "full_frame"), annotation_roots=ann)
    va_ds = TyreDataset(
        va_df, root,
        build_transforms(cfg["input_resolution"], False, cfg.get("preprocessing", "raw")),
        roi_mode=cfg.get("roi_mode", "full_frame"), annotation_roots=ann)

    sampler_name = cfg.get("sampler_name", "session_balanced")
    if sampler_name == "session_balanced":
        w = tr_df["class_session_balanced_weight"].astype(float).values
        sampler, shuffle = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), True), False
    elif sampler_name == "class_weighted":
        counts = tr_df.proxy_label.value_counts()
        w = tr_df.proxy_label.map(lambda y: 1.0 / max(1, counts[y])).astype(float).values
        sampler, shuffle = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), True), False
    else:
        sampler, shuffle = None, True

    requested_nw = int(cfg.get("num_workers", 2))
    roi_loader = cfg.get("roi_mode", "full_frame") == "tyre_crop"

    # ⚠ Bug 26. The ROI arms were moved to the synchronous loader when their
    # host RAM climbed 3 -> 20 GB; the full-frame arms kept two persistent,
    # pinned workers. Then a full-frame `wd_low` run paused on the RAM guard at
    # epoch 36 with 89.6%, and every single epoch of it had logged **`dl 0%`**.
    #
    # `dataload_frac` was 0% for 49 consecutive epochs. The workers were buying
    # nothing at all -- the GPU is the bottleneck at 4.2 min/epoch -- while
    # costing two forked processes whose RSS counts against the same cgroup,
    # plus PyTorch's pinned-host allocator, which caches and does not return.
    #
    # So the measurement already said the answer. Synchronous everywhere, and
    # if a future arm is genuinely loader-bound its `dataload_frac` will say so
    # and can be given workers back deliberately.
    nw = 0 if (roi_loader or requested_nw == 0) else requested_nw
    if nw and dataloading_is_free(cfg):
        nw = 0
    pin = bool(torch.cuda.is_available() and nw > 0)
    _print("LOADER", f"workers={nw} pin_memory={pin} "
                     f"({'ROI memory-safe path' if roi_loader else 'standard path'}) "
                     "-- these are CPU input helpers, NOT the Kaggle/GPU worker count; "
                     "GPU training remains active")
    tr_dl = DataLoader(tr_ds, batch_size=cfg["batch_size"], sampler=sampler, shuffle=shuffle,
                       num_workers=nw, pin_memory=pin, drop_last=True,
                       persistent_workers=nw > 0)
    va_dl = DataLoader(va_ds, batch_size=cfg["batch_size"], shuffle=False,
                       num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)
    return tr_dl, va_dl


def dataloading_is_free(cfg: dict) -> bool:
    """Is this configuration GPU-bound enough that loader workers buy nothing?

    Kept as an explicit, named decision rather than a bare `nw = 0`, because
    the honest justification is a measurement and it should be readable:
    every epoch of the 384px and 512px Stage-B arms logged `dl 0%` or `dl 1%`
    at 4+ minutes per epoch. Two worker processes cannot speed up an epoch that
    spends none of its time waiting for data, and their RSS counts against the
    same cgroup budget the OOM killer enforces.

    Small, fast configurations are the case where prefetching can genuinely
    matter, so they keep their workers.
    """
    res = int(cfg.get("input_resolution", 384))
    return res >= 320


def validate_config(cfg: dict) -> None:
    """Fail before training when an OFAT arm is misspelled or unsupported.

    Silent no-ops are especially dangerous in an ablation: they produce two
    differently named runs with identical behaviour and look like a null result.
    """
    allowed = {
        "head_type": {"coral", "ce"},
        "preprocessing": {"raw", "grayscale", "clahe"},
        "roi_mode": {"full_frame", "tyre_crop"},
        "sampler_name": {"session_balanced", "class_weighted", "uniform"},
        "finetune_depth": {"full", "frozen"},
    }
    for key, values in allowed.items():
        val = cfg.get(key, RECIPE.get(key))
        if val not in values:
            raise ValueError(f"unsupported {key}={val!r}; choose one of {sorted(values)}")
    if cfg.get("roi_mode") == "tyre_crop":
        for key in ("clean_mask_root", "propagated_mask_root"):
            if not cfg.get(key):
                raise ValueError(f"roi_mode='tyre_crop' requires {key}")


# --------------------------------------------------------------------------
# 9. Model zoo
# --------------------------------------------------------------------------

ZOO: dict[str, dict] = {
    # key                 timm name                                           res  bs   cam target
    "resnet18":      dict(timm="resnet18",                                     res=384, bs=32, cam="layer4"),
    "resnet50":      dict(timm="resnet50",                                     res=384, bs=32, cam="layer4"),
    "resnext50":     dict(timm="resnext50_32x4d",                              res=384, bs=32, cam="layer4"),
    "densenet121":   dict(timm="densenet121",                                  res=384, bs=32, cam="features_norm5"),
    "vgg16bn":       dict(timm="vgg16_bn",                                     res=384, bs=16, cam="features"),
    "convnextv2_t":  dict(timm="convnextv2_tiny.fcmae_ft_in22k_in1k",          res=384, bs=32, cam="stages"),
    # timm defines the Small topology but publishes no pretrained Small
    # checkpoint.  An older registry entry appended the non-existent
    # ``fcmae_ft_in22k_in1k`` tag; the old emergency ResNet-18 fallback then
    # made nine completed runs look like ConvNeXt-V2-S runs.  Keep the base
    # topology here only so those checkpoints can be audited/rejected cleanly.
    # It is deliberately absent from new Stage-A training plans.
    "convnextv2_s":  dict(timm="convnextv2_small",                             res=384, bs=16, cam="stages",
                           pretrained_available=False, stage_a_valid=False),
    "effnetv2s":     dict(timm="tf_efficientnetv2_s.in21k_ft_in1k",            res=384, bs=32, cam="conv_head"),
    "regnety016":    dict(timm="regnety_016",                                  res=384, bs=32, cam="s4"),
    "mobilenetv4":   dict(timm="mobilenetv4_conv_medium.e500_r256_in1k",       res=384, bs=64, cam="blocks"),
    "vit_s":         dict(timm="vit_small_patch16_384.augreg_in21k_ft_in1k",   res=384, bs=32, cam="blocks"),
    "deit3_s":       dict(timm="deit3_small_patch16_384.fb_in22k_ft_in1k",     res=384, bs=32, cam="blocks"),
    "swin_t":        dict(timm="swin_tiny_patch4_window7_224",                 res=224, bs=32, cam="layers"),
    "swin_s":        dict(timm="swin_small_patch4_window7_224",                res=224, bs=16, cam="layers"),
    "coatnet0":      dict(timm="coatnet_0_rw_224.sw_in1k",                     res=224, bs=32, cam="stages"),
    "maxvit_t":      dict(timm="maxvit_tiny_tf_384.in1k",                      res=384, bs=16, cam="stages"),
    "dinov2_s":      dict(timm="vit_small_patch14_dinov2.lvd142m",             res=392, bs=32, cam="blocks"),
    "dinov2_b":      dict(timm="vit_base_patch14_dinov2.lvd142m",              res=392, bs=16, cam="blocks"),
    "clip_b16":      dict(timm="vit_base_patch16_clip_384.laion2b_ft_in12k_in1k", res=384, bs=16, cam="blocks"),
}
# Swin and CoAtNet are FIXED-WINDOW at 224. Do not silently feed them 384 --
# that is the "architecture cannot do what the sweep assumes" bug. They are
# declared 224-only and excluded from the resolution sweep.
FIXED_224 = {"swin_t", "swin_s", "coatnet0"}


def _timm_model_candidates(model_name: str, pretrained: bool) -> list[str]:
    """Return model identifiers appropriate for the requested weight source.

    Text after the first dot is a timm *pretrained-weight tag*, not part of the
    network topology.  Checkpoint reconstruction supplies its own weights, so
    ``pretrained=False`` must instantiate the untagged topology.  This also
    makes old checkpoints readable after timm retires or renames a weight tag.
    """
    name = str(model_name)
    if not pretrained and "." in name:
        return [name.split(".", 1)[0]]
    return [name]


def infer_checkpoint_architecture(state_dict: dict) -> str:
    """Infer a known backbone from saved tensor names/shapes.

    This is an integrity check, not a model loader.  It deliberately returns
    ``"unknown"`` rather than guessing when the signature is ambiguous.
    """
    sd = {str(k).removeprefix("module."): v for k, v in state_dict.items()}
    keys = set(sd)
    if {"conv1.weight", "layer1.0.conv1.weight", "layer4.0.conv1.weight"} <= keys:
        if "layer1.0.conv3.weight" not in keys:
            return "resnet18"
        conv2 = sd.get("layer1.0.conv2.weight")
        if getattr(conv2, "ndim", 0) == 4 and int(conv2.shape[1]) <= 8:
            return "resnext50"
        return "resnet50"
    if any(k.startswith("features.denseblock") for k in keys):
        return "densenet121"
    if any(k.startswith("stages.2.blocks.") for k in keys):
        stage2 = []
        for k in keys:
            m = re.match(r"stages\.2\.blocks\.(\d+)\.", k)
            if m:
                stage2.append(int(m.group(1)))
        stem = sd.get("stem.0.weight")
        width = int(stem.shape[0]) if getattr(stem, "ndim", 0) == 4 else None
        depth = max(stage2, default=-1) + 1
        if depth == 9 and width == 96:
            return "convnextv2_t"
        if depth == 27 and width == 96:
            return "convnextv2_s"
    return "unknown"


def build_model(arch: str, n_classes: int = 3, pretrained: bool = True,
                head: str = "coral", drop_path: float = 0.0,
                img_size: int | None = None, verify: bool = True):
    """Build one architecture, at the resolution it will actually be fed.

    ⚠ Bug 15 -- this cost 18 runs and half a day. The old version never told
    timm what resolution the images would be:

        m = timm.create_model(spec["timm"], pretrained=..., num_classes=...)

    Most models do not care. `vit_*_patch14_dinov2` does: it is created with
    `img_size=518` and its patch embedding asserts an exact match, so every
    dinov2 run died on the first batch with

        AssertionError: Input height (392) doesn't match model (518).

    Note where it died -- in `forward`, not in `create_model`. The old
    fallback-to-resnet18 `except` only wrapped construction, so it never fired,
    and the failure surfaced 100 lines later as a training crash rather than as
    "this architecture cannot take this input".

    Fix, in order of preference: tell timm the size, let it interpolate the
    position embeddings, and then **prove it with a real forward pass** before
    returning. A model that cannot forward at its own configured resolution is
    a build failure, and it should say so here rather than during training.
    """
    import torch
    spec = ZOO.get(arch)
    if spec is None:
        raise KeyError(f"unknown arch '{arch}'. known: {sorted(ZOO)}")
    res = int(img_size or spec.get("res", 384))
    out_dim = (n_classes - 1) if head == "coral" else n_classes

    if pretrained and spec.get("pretrained_available") is False:
        raise RuntimeError(
            f"{arch} has no published pretrained checkpoint in the current "
            "timm registry. It is excluded from the pretrained Stage-A sweep; "
            "do not substitute another architecture under this run id."
        )

    base = dict(pretrained=pretrained, num_classes=out_dim)
    if drop_path:
        base["drop_path_rate"] = drop_path

    # Most specific first. `img_size` re-interpolates the position embeddings
    # at construction; `dynamic_img_size` does it per forward. Plenty of models
    # accept neither, which is why the plain call is still last.
    attempts = [
        ("img_size + dynamic", dict(base, img_size=res, dynamic_img_size=True)),
        ("img_size", dict(base, img_size=res)),
        ("dynamic", dict(base, dynamic_img_size=True)),
        ("plain", dict(base)),
    ]

    errors = []
    try:
        import timm
    except Exception as e:
        raise RuntimeError(
            f"timm is required to build {arch}; import failed with "
            f"{type(e).__name__}: {e}. No architecture fallback is allowed."
        ) from e

    for model_name in _timm_model_candidates(spec["timm"], pretrained):
        for label, kw in attempts:
            try:
                m = timm.create_model(model_name, **kw)
            except Exception as e:
                errors.append(
                    f"{model_name} / {label}: create failed -- "
                    f"{type(e).__name__}: {e}"
                )
                continue
            if not verify:
                return m
            try:
                m.eval()
                with torch.no_grad():
                    out = m(torch.zeros(1, 3, res, res))
                if out.shape[-1] != out_dim:
                    raise RuntimeError(f"head produced {tuple(out.shape)}, expected (..., {out_dim})")
                if label != "plain" or model_name != spec["timm"]:
                    _print("ZOO", f"{arch}: built {model_name} at {res}px via {label}")
                return m.train()
            except Exception as e:
                errors.append(
                    f"{model_name} / {label}: forward at {res}px failed -- "
                    f"{type(e).__name__}: {e}"
                )

    raise RuntimeError(
        f"{arch} ({spec['timm']}) cannot run at {res}px. Attempts:\n  "
        + "\n  ".join(errors)
        + f"\n\nEither pick a resolution the checkpoint supports, or drop {arch} "
          f"from the sweep. Do NOT let this reach training -- it fails on the "
          f"first batch, after the dataloaders and the pretrained download."
    )


def verify_zoo(archs=None, pretrained: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Build every architecture at its own configured resolution.

    ⚠ NB00 already reported `dinov2_s` and `dinov2_b` as FAIL, printed
    "17/19 architectures build", and said "fix them BEFORE Stage A" -- and then
    carried on and returned success. Four accounts then spent a session
    discovering the same thing at a cost of 18 runs.

    **A preflight that reports but does not block is not a preflight.** This
    returns a table; `assert_zoo_ok` is what callers should use.
    """
    import torch
    rows = []
    for arch in (archs or list(ZOO)):
        spec = ZOO[arch]
        r = {"arch": arch, "res": spec["res"], "bs": spec["bs"],
             "fixed_224": arch in FIXED_224}
        try:
            m = build_model(arch, 3, pretrained=pretrained, head="coral")
            with torch.no_grad():
                out = m(torch.zeros(2, 3, spec["res"], spec["res"]))
            r.update(ok=True, out_shape=tuple(out.shape),
                     params_M=round(sum(p.numel() for p in m.parameters()) / 1e6, 1), err="")
            del m
        except Exception as e:
            r.update(ok=False, out_shape=None, params_M=np.nan,
                     err=f"{type(e).__name__}: {str(e).splitlines()[0][:120]}")
        if verbose:
            print(("  OK   " if r["ok"] else "  FAIL ") + f"{arch:14s} {r['err']}")
        rows.append(r)
    return pd.DataFrame(rows)


def assert_zoo_ok(archs=None, pretrained: bool = False) -> pd.DataFrame:
    """Same as `verify_zoo`, but raises. Use this in preflight and at the top
    of any notebook that is about to spend GPU-hours."""
    df = verify_zoo(archs, pretrained=pretrained, verbose=True)
    bad = df[~df.ok]
    if len(bad):
        raise RuntimeError(
            f"{len(bad)} architecture(s) cannot run at their configured resolution:\n"
            + bad[["arch", "res", "err"]].to_string(index=False)
            + "\n\nFix or remove them before starting. Every run of a broken "
              "architecture fails on its first batch, and 27 of those still "
              "look like a notebook that ran."
        )
    print(f"\nall {len(df)} architecture(s) build and forward at their configured resolution")
    return df


class CoralHead:
    """Rank-consistent ordinal regression (CORAL).

    K-1 cumulative binary tasks: P(y>0), P(y>1). Confusing low with high then
    costs more than confusing low with mid, which is what we want -- the
    classes are ordered.
    """

    @staticmethod
    def loss(logits, targets, n_classes=3):
        import torch
        import torch.nn.functional as F
        lev = torch.zeros(targets.size(0), n_classes - 1, device=logits.device)
        for k in range(n_classes - 1):
            lev[:, k] = (targets > k).float()
        return F.binary_cross_entropy_with_logits(logits, lev)

    @staticmethod
    def predict(logits):
        import torch
        return (torch.sigmoid(logits) > 0.5).sum(1)

    @staticmethod
    def probs(logits, n_classes=3):
        import torch
        cum = torch.sigmoid(logits)                     # [P(y>0), P(y>1)]
        p = torch.zeros(logits.size(0), n_classes, device=logits.device)
        p[:, 0] = 1 - cum[:, 0]
        for k in range(1, n_classes - 1):
            p[:, k] = cum[:, k - 1] - cum[:, k]
        p[:, -1] = cum[:, -1]
        return p.clamp_min(1e-8) / p.clamp_min(1e-8).sum(1, keepdim=True)


# --------------------------------------------------------------------------
# 10. Training -- fixed epoch budget, NO early stopping, tqdm per epoch
# --------------------------------------------------------------------------

def _autocast(dev):
    """torch.cuda.amp.autocast is deprecated in torch>=2.4."""
    import torch
    en = dev.type == "cuda"
    try:    return torch.amp.autocast("cuda", enabled=en)
    except (AttributeError, TypeError): return torch.cuda.amp.autocast(enabled=en)


def _grad_scaler(dev):
    import torch
    en = dev.type == "cuda"
    try:    return torch.amp.GradScaler("cuda", enabled=en)
    except (AttributeError, TypeError): return torch.cuda.amp.GradScaler(enabled=en)


def _tqdm(*a, **k):
    try:
        from tqdm.auto import tqdm
        return tqdm(*a, **k)
    except Exception:
        class _Dummy:
            def __init__(self, it=None, **kw): self.it = it or []
            def __iter__(self): return iter(self.it)
            def set_postfix(self, *a, **k): pass
            def update(self, *a): pass
            def close(self): pass
        return _Dummy(*a, **k)


def _shutdown_loader(loader) -> None:
    """Stop persistent workers explicitly instead of waiting for GC."""
    it = getattr(loader, "_iterator", None)
    if it is not None:
        with contextlib.suppress(Exception):
            it._shutdown_workers()
        with contextlib.suppress(Exception):
            loader._iterator = None


class Trainer:
    """One run = one (arch, technique, fold, seed).

    NO EARLY STOPPING. Every run trains its full epoch budget. Equal budget for
    every architecture keeps the comparison fair, and it means a run's length
    is known in advance -- which is what makes the work-shard estimate honest.
    """

    def __init__(self, cfg: dict, session: "Session"):
        self.cfg = dict(cfg)
        self.sess = session
        self.run_id = cfg["run_id"]
        self.run_dir = Path(session.stage_dir) / "runs" / self.run_id
        for sub in ("metrics", "telemetry", "checkpoints", "per_sample", "env"):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        self.hist_path = self.run_dir / "metrics" / "epochs.csv"
        self.ckpt_last = self.run_dir / "checkpoints" / "ckpt_last.pt"
        self.ckpt_best = self.run_dir / "checkpoints" / "ckpt_best.pt"
        self.cfg["config_hash"] = config_hash(self.cfg)
        self.mon: HardwareMonitor | None = None
        self.start_epoch = 0
        # Epochs actually COMPLETED. Distinct from start_epoch: a run that
        # resumed at 30 and died at 47 started at 30 and completed 47, and
        # reporting the former is how a resume silently loses 17 epochs.
        self.last_epoch = 0
        self.best_qwk = -9e9
        self.wall_seconds = 0.0
        self.energy_joules = 0.0

    # -- repo paths -------------------------------------------------------
    def rp(self, rel: str) -> str:
        return f"runs/{self.run_id}/{rel}"

    def enqueue_light(self):
        u = self.sess.uploader
        u.enqueue(self.run_dir / "config.yaml", self.rp("config.yaml"))
        u.enqueue(self.run_dir / "STATUS.json", self.rp("STATUS.json"), force=True)
        # ⚠ Bug 14: summary.json was written locally and never enqueued, while
        # confirm_on_hf treated its absence as "not finished". Every one of 36
        # completed runs was therefore reported as RESUMABLE. Two bugs whose
        # only symptom was a report that could never say FINISHED.
        u.enqueue(self.run_dir / "summary.json", self.rp("summary.json"), force=True)
        u.enqueue(self.run_dir / "split_health.json", self.rp("split_health.json"))
        u.enqueue(self.hist_path, self.rp("metrics/epochs.csv"), force=True)
        for f in (self.run_dir / "metrics").glob("*.csv"):
            u.enqueue(f, self.rp(f"metrics/{f.name}"), force=True)
        u.enqueue(self.run_dir / "env" / "environment.json", self.rp("env/environment.json"))

    def enqueue_heavy(self):
        u = self.sess.uploader
        if self.ckpt_last.exists():
            u.enqueue(self.ckpt_last, self.rp("checkpoints/ckpt_last.pt"), force=True)
        if self.ckpt_best.exists():
            u.enqueue(self.ckpt_best, self.rp("checkpoints/ckpt_best.pt"), force=True)

    def enqueue_bulk(self):
        u = self.sess.uploader
        u.enqueue_dir(self.run_dir / "telemetry", self.rp("telemetry"), force=True)
        u.enqueue_dir(self.run_dir / "per_sample", self.rp("per_sample"), force=True)

    # -- checkpointing ----------------------------------------------------
    def save_ckpt(self, path: Path, model, opt, sched, scaler, epoch: int, metrics: dict):
        import torch
        # DataParallel is a runtime detail. Saving the unwrapped module keeps
        # checkpoints portable to one GPU, two GPUs, CPU inference, and XAI.
        core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        state = {
            "epoch": epoch,                                  # last COMPLETED epoch
            "model": core_model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict() if sched else None,
            "scaler": scaler.state_dict() if scaler else None,   # omit -> AMP scale resets
            "rng": capture_rng(),                            # ALL FOUR streams
            "config": self.cfg,
            "config_hash": self.cfg["config_hash"],
            "metrics_at_save": metrics,
            "best_qwk": self.best_qwk,
            "wall_seconds": self.wall_seconds,               # cumulative across restarts
            "energy_joules": self.energy_joules,
            "arch": self.cfg["arch"],
            "classes": CLASSES,
            "input_resolution": self.cfg["input_resolution"],
            "normalisation": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "lib_version": __version__,
            "torch_version": torch.__version__,
            "dataset_version": "final_v1",
        }
        tmp = path.with_suffix(".tmp")
        try:
            torch.save(state, tmp)
            os.replace(tmp, path)                            # atomic
        finally:
            # The state dict only borrows live tensors. Drop the container and
            # return serialization buffers to the OS before the next epoch.
            del state
            release_host_memory()

    def fetch_remote_state(self) -> bool:
        """Bring this run's checkpoint back from HuggingFace before training.

        THIS IS THE FIX for the ten hours that got retrained. Kaggle wipes the
        session disk between sessions, so `ckpt_last.exists()` is False in
        every fresh session and `try_resume` gave up without ever asking
        whether a checkpoint existed anywhere else. It always did -- we push
        one every epoch.
        """
        if self.ckpt_last.exists():
            return True                       # already here; nothing to do
        inv = getattr(self.sess, "inventory", None)
        if inv is None:
            return False
        if not inv.files:                     # never listed, or listing failed
            inv.refresh([self.run_id], verbose=False)
        return inv.fetch_run(self.run_id)

    def try_resume(self, model, opt, sched, scaler) -> bool:
        import torch
        self.fetch_remote_state()
        if not self.ckpt_last.exists():
            return False
        try:
            ck = torch.load(self.ckpt_last, map_location="cpu", weights_only=False)
        except Exception as e:
            _print("RESUME", f"checkpoint unreadable ({e}) -- starting fresh")
            return False
        if ck.get("config_hash") != self.cfg["config_hash"]:
            _print("RESUME", f"config_hash mismatch "
                             f"({ck.get('config_hash')} != {self.cfg['config_hash']}) -- starting fresh")
            del ck
            release_host_memory()
            return False
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])              # load to CPU first, then move
        if sched and ck.get("scheduler"):
            sched.load_state_dict(ck["scheduler"])
        if scaler and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        restore_rng(ck.get("rng"))
        self.start_epoch = self.last_epoch = int(ck["epoch"])
        self.best_qwk = float(ck.get("best_qwk", -9e9))
        self.wall_seconds = float(ck.get("wall_seconds", 0.0))
        self.energy_joules = float(ck.get("energy_joules", 0.0))
        # A milestone push can land AFTER the checkpoint was written, so the log
        # may contain epochs the checkpoint does not know about. Without this,
        # duplicate epoch numbers make every cumulative statistic wrong.
        if self.hist_path.exists():
            h = read_epoch_history(self.hist_path, repair=True)
            if "epoch" in h.columns:
                atomic_write_text(
                    self.hist_path,
                    h[h.epoch <= self.start_epoch].to_csv(index=False),
                )
        if self.start_epoch >= int(self.cfg.get("max_epochs", self.start_epoch + 1)):
            _print("RESUME", f"{self.run_id}: checkpoint already contains all "
                             f"{self.start_epoch} epochs; finalising repaired metadata "
                             "without another training epoch")
        else:
            _print("RESUME", f"{self.run_id}: continuing from epoch {self.start_epoch+1}"
                             f" (best QWK so far {self.best_qwk:.4f})")
        del ck
        release_host_memory()
        return True

    # -- the loop ---------------------------------------------------------
    def run(self) -> dict:
        import torch
        import torch.nn as nn

        cfg = self.cfg
        seed_everything(cfg["seed"])
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        memory_format_name = training_memory_format(cfg["arch"])
        memory_format = (torch.contiguous_format if memory_format_name == "contiguous"
                         else torch.channels_last)
        # RegNet's conservative profile avoids a reproducible T4/cuDNN NHWC
        # kernel failure. This changes only runtime layout/algorithm selection;
        # model, weights, input resolution, batch and optimiser remain locked.
        torch.backends.cudnn.benchmark = memory_format_name == "channels_last"

        atomic_write_text(self.run_dir / "config.yaml",
                          "\n".join(f"{k}: {v}" for k, v in sorted(cfg.items())))
        atomic_write_text(self.run_dir / "config_hash.txt", cfg["config_hash"])
        atomic_write_json(self.run_dir / "env" / "environment.json", self.sess.environment())

        tr_df, va_df = load_split(self.sess.data_root, cfg["fold"])
        self.split_info = split_health(tr_df, va_df, cfg["fold"])
        atomic_write_json(self.run_dir / "split_health.json", self.split_info)
        tr_dl, va_dl = build_loaders(self.sess.data_root, tr_df, va_df, cfg)

        # img_size is passed, not assumed. See Bug 15 in build_model.
        validate_config(cfg)
        model = build_model(cfg["arch"], 3, cfg.get("pretrained", True), cfg["head_type"],
                            img_size=cfg["input_resolution"]).to(dev)

        if cfg.get("finetune_depth", "full") == "frozen":
            for p in model.parameters():
                p.requires_grad = False
            head = model.get_classifier() if hasattr(model, "get_classifier") else None
            if head is None or not hasattr(head, "parameters"):
                raise RuntimeError(f"{cfg['arch']} does not expose get_classifier(); cannot freeze safely")
            for p in head.parameters():
                p.requires_grad = True
            if not any(p.requires_grad for p in model.parameters()):
                raise RuntimeError("frozen arm left no trainable classifier parameters")

        model = model.to(memory_format=memory_format)
        n_all = sum(p.numel() for p in model.parameters())
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)

        decay, no_decay = [], []
        for n_, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or n_.endswith(".bias") else decay).append(p)
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                                 {"params": no_decay, "weight_decay": 0.0}],
                                lr=cfg["lr_initial"])
        total_steps = max(1, cfg["max_epochs"] * len(tr_dl))
        warm = max(1, cfg.get("warmup_epochs", 5) * len(tr_dl))

        def lr_lambda(step):
            if step < warm:
                return step / warm
            p = (step - warm) / max(1, total_steps - warm)
            return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        scaler = _grad_scaler(dev)                       # fp16: T4 has no bf16

        resumed = self.try_resume(model, opt, sched, scaler)
        model = model.to(dev).to(memory_format=memory_format)
        gpu_count = torch.cuda.device_count() if dev.type == "cuda" else 0
        if gpu_count > 1:
            model = torch.nn.DataParallel(model)
        for st in opt.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(dev)

        self.mon = HardwareMonitor(self.run_dir / "telemetry").start()
        gpu_static = self.mon.gpu_static()

        self.sess.registry.emit(self.run_id, "running", account=self.sess.account,
                                worker=self.sess.worker_id, epoch=self.start_epoch,
                                arch=cfg["arch"], fold=cfg["fold"], seed=cfg["seed"])
        atomic_write_json(self.run_dir / "STATUS.json",
                          {"status": "running", "epoch": self.start_epoch, "iso": iso()})

        n_ep = cfg["max_epochs"]
        _print("TRAIN", f"{self.run_id}  |  {cfg['arch']}  fold {cfg['fold']}  seed {cfg['seed']}  "
                        f"|  {n_ep} epochs (no early stopping)  |  {n_all/1e6:.1f} M params")
        _print("TRAIN", f"devices {max(1, gpu_count)}  |  trainable {n_tr/1e6:.1f}/{n_all/1e6:.1f} M params")
        _print("CUDA", f"layout={memory_format_name} cudnn_benchmark="
                       f"{torch.backends.cudnn.benchmark} safety={CUDA_SAFETY_REVISION}")
        _print("TRAIN", f"train {len(tr_df)} imgs / {len(tr_dl)} batches   "
                        f"val {len(va_df)} imgs / {va_df.session_group.nunique()} sessions")
        _print("LIVE", "Plain-text epoch heartbeats are authoritative; a saved Kaggle "
                       "progress widget can remain at 0% while the cell is running.")

        step_traces: list[dict] = []
        status = "completed"
        pause_reason = None
        cuda_restart_required = False
        err_type = err_msg = None
        try:
            for ep in range(self.start_epoch, n_ep):
                ep_t0 = now()
                model.train()
                run_loss = run_corr = run_n = 0
                data_s = fwd_s = bwd_s = opt_s = 0.0
                gnorms, step_times = [], []
                nan_batches = clip_hits = 0
                scale_before = float(scaler.get_scale()) if dev.type == "cuda" else 1.0
                scale_drops = 0

                bar = _tqdm(total=len(tr_dl), desc=f"ep {ep+1:>3}/{n_ep}", leave=False,
                            unit="b", dynamic_ncols=True)
                _print("LIVE", f"{self.run_id}: epoch {ep+1}/{n_ep} started "
                               f"({len(tr_dl)} training batches)")
                t_last = now()
                for step, (x, y, _) in enumerate(tr_dl):
                    t_s = now(); data_s += t_s - t_last
                    x = x.to(dev, non_blocking=True).to(memory_format=memory_format)
                    y = y.to(dev, non_blocking=True)

                    opt.zero_grad(set_to_none=True)
                    t_f = now()
                    with _autocast(dev):
                        logits = model(x)
                        loss = (CoralHead.loss(logits, y) if cfg["head_type"] == "coral"
                                else nn.functional.cross_entropy(
                                    logits, y, label_smoothing=cfg.get("label_smoothing", 0.0)))
                    t_b = now(); fwd_s += t_b - t_f

                    if not torch.isfinite(loss):
                        nan_batches += 1                     # silent under AMP otherwise
                        bar.update(1); t_last = now(); continue

                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 5.0))
                    gnorms.append(float(gn))
                    clip_hits += int(float(gn) > cfg.get("grad_clip", 5.0))
                    t_o = now(); bwd_s += t_o - t_b
                    s_pre = float(scaler.get_scale()) if dev.type == "cuda" else 1.0
                    scaler.step(opt); scaler.update()
                    s_post = float(scaler.get_scale()) if dev.type == "cuda" else 1.0
                    scale_drops += int(s_post < s_pre)       # each = a DISCARDED step
                    sched.step()
                    opt_s += now() - t_o

                    with torch.no_grad():
                        pred = (CoralHead.predict(logits) if cfg["head_type"] == "coral"
                                else logits.argmax(1))
                        run_corr += int((pred == y).sum())
                    run_loss += float(loss.detach()) * y.size(0); run_n += y.size(0)
                    step_times.append(now() - t_s)

                    if len(step_traces) < 2000:      # per EPOCH now; cleared each epoch
                        step_traces.append({"epoch": ep + 1, "step": step,
                                            "t_data": round(t_s - t_last, 4),
                                            "t_fwd": round(t_b - t_f, 4),
                                            "t_bwd": round(t_o - t_b, 4),
                                            "loss": round(float(loss.detach()), 5),
                                            "grad_norm": round(float(gn), 4),
                                            "lr": sched.get_last_lr()[0],
                                            "amp_scale": s_post})
                    bar.set_postfix(loss=f"{run_loss/max(run_n,1):.4f}",
                                    acc=f"{run_corr/max(run_n,1):.3f}",
                                    lr=f"{sched.get_last_lr()[0]:.2e}")
                    bar.update(1)
                    if step == 0:
                        _print("LIVE", f"{self.run_id}: epoch {ep+1}/{n_ep} "
                                       f"batch 1/{len(tr_dl)} completed in "
                                       f"{human_time(now() - ep_t0)} -- training is active")
                    t_last = now()
                bar.close()
                train_s = now() - ep_t0

                # ---- validate ----
                v_t0 = now()
                model.eval()
                P, Y, PR, IDX = [], [], [], []
                v_loss = v_n = 0
                vbar = _tqdm(total=len(va_dl), desc="   val", leave=False, unit="b", dynamic_ncols=True)
                with torch.no_grad():
                    for x, y, idx in va_dl:
                        x = x.to(dev, non_blocking=True).to(memory_format=memory_format)
                        yd = y.to(dev, non_blocking=True)
                        with _autocast(dev):
                            logits = model(x)
                            l = (CoralHead.loss(logits, yd) if cfg["head_type"] == "coral"
                                 else nn.functional.cross_entropy(logits, yd))
                        pr = (CoralHead.probs(logits.float()) if cfg["head_type"] == "coral"
                              else logits.float().softmax(1))
                        P.append(pr.argmax(1).cpu().numpy()); Y.append(y.numpy())
                        PR.append(pr.cpu().numpy()); IDX.append(idx.numpy())
                        v_loss += float(l) * y.size(0); v_n += y.size(0)
                        vbar.update(1)
                vbar.close()
                val_s = now() - v_t0
                y_pred = np.concatenate(P); y_true = np.concatenate(Y)
                probs = np.concatenate(PR); vidx = np.concatenate(IDX)
                vm, cm = classification_report_dict(y_true, y_pred, probs, "val_")

                ep_s = now() - ep_t0
                self.wall_seconds += ep_s
                hw = self.mon.window(ep_t0, now()) if self.mon else {}
                self.energy_joules += float(hw.get("energy_joules_epoch", 0) or 0)

                # Detach explicitly. PyTorch 2.10 warns when float(tensor)
                # implicitly crosses an autograd boundary; the norm is
                # telemetry only and must never build or retain a graph.
                with torch.no_grad():
                    wn = math.sqrt(sum(float(p.detach().norm().item()) ** 2
                                       for p in model.parameters()))
                row = {
                    "run_id": self.run_id, "stage": cfg["stage"], "arch": cfg["arch"],
                    "technique": cfg["technique"], "fold": cfg["fold"], "seed": cfg["seed"],
                    "epoch": ep + 1, "global_step": (ep + 1) * len(tr_dl),
                    "samples_seen": (ep + 1) * len(tr_dl) * cfg["batch_size"],
                    "ts_start": ep_t0, "ts_end": now(), "iso_start": iso(ep_t0), "iso_end": iso(),
                    "account": self.sess.account, "worker_id": self.sess.worker_id,
                    "session_id": self.sess.session_id, "host": self.sess.host,
                    "config_hash": cfg["config_hash"], "lib_version": __version__,
                    "train_loss": run_loss / max(run_n, 1),
                    "train_acc": run_corr / max(run_n, 1),
                    "val_loss": v_loss / max(v_n, 1),
                    "lr_group0": sched.get_last_lr()[0],
                    "grad_norm_mean": float(np.mean(gnorms)) if gnorms else NA,
                    "grad_norm_max": float(np.max(gnorms)) if gnorms else NA,
                    "grad_norm_p50": float(np.percentile(gnorms, 50)) if gnorms else NA,
                    "grad_norm_p95": float(np.percentile(gnorms, 95)) if gnorms else NA,
                    "grad_norm_p99": float(np.percentile(gnorms, 99)) if gnorms else NA,
                    "grad_clip_hit_rate": clip_hits / max(len(gnorms), 1),
                    "weight_norm_total": wn,
                    "update_to_weight_ratio": (float(np.mean(gnorms)) * sched.get_last_lr()[0] / wn) if (gnorms and wn) else NA,
                    "amp_scale": float(scaler.get_scale()) if dev.type == "cuda" else NA,
                    "amp_scale_decreases": scale_drops,
                    "nan_or_inf_batches": nan_batches,
                    "epoch_seconds": ep_s, "train_seconds": train_s, "val_seconds": val_s,
                    "dataload_seconds": data_s, "compute_seconds": fwd_s + bwd_s,
                    "backward_seconds": bwd_s, "optimizer_seconds": opt_s,
                    "dataload_frac": data_s / max(ep_s, 1e-9),
                    "step_time_mean": float(np.mean(step_times)) if step_times else NA,
                    "step_time_p50": float(np.percentile(step_times, 50)) if step_times else NA,
                    "step_time_p90": float(np.percentile(step_times, 90)) if step_times else NA,
                    "step_time_p99": float(np.percentile(step_times, 99)) if step_times else NA,
                    "images_per_second": run_n / max(train_s, 1e-9),
                    "n_params_total": n_all, "n_params_trainable": n_tr,
                    "runtime_loader_num_workers": int(tr_dl.num_workers),
                    "runtime_loader_pin_memory": bool(tr_dl.pin_memory),
                    "runtime_memory_safety_revision": MEMORY_SAFETY_REVISION,
                    "runtime_hf_commit_policy_revision": HF_COMMIT_POLICY_REVISION,
                    "runtime_epoch_history_schema_revision": EPOCH_HISTORY_SCHEMA_REVISION,
                    "runtime_cuda_memory_format": memory_format_name,
                    "runtime_cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                    "runtime_cuda_safety_revision": CUDA_SAFETY_REVISION,
                    "runtime_scheduler_safety_revision": SCHEDULER_SAFETY_REVISION,
                    "runtime_process_isolation_revision": PROCESS_ISOLATION_REVISION,
                    "runtime_isolated_child": bool(cfg.get("_isolated_child", False)),
                    "runtime_host_ram_pause_percent": HOST_RAM_PAUSE_PERCENT,
                    "wall_seconds_cumulative": self.wall_seconds,
                    "energy_joules_cumulative": self.energy_joules,
                    "epochs_planned": n_ep,
                    **{f"cfg_{k}": v for k, v in cfg.items() if k not in ("run_id",)},
                    **vm, **hw, **gpu_static,
                }
                # per-session validation accuracy -- how single-tyre
                # memorisation becomes visible
                vsub = va_df.reset_index(drop=True).iloc[vidx]
                for sg, grp in pd.DataFrame({"s": vsub.session_group.values,
                                             "ok": (y_pred == y_true)}).groupby("s"):
                    row[f"val_acc_session_{sg}"] = float(grp.ok.mean())
                    row[f"val_n_session_{sg}"] = int(len(grp))

                append_epoch_row(self.hist_path, row)

                is_best = vm["val_qwk"] > self.best_qwk
                if is_best:
                    self.best_qwk = vm["val_qwk"]
                    pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_SHORT],
                                 columns=[f"pred_{c}" for c in CLASS_SHORT]).to_csv(
                        self.run_dir / "metrics" / "confusion_matrix.csv")
                    pd.DataFrame({"image_id": vsub.image_id.values,
                                  "session_group": vsub.session_group.values,
                                  "true": y_true, "pred": y_pred,
                                  **{f"prob_{c}": probs[:, i] for i, c in enumerate(CLASS_SHORT)}
                                  }).to_parquet(self.run_dir / "per_sample" / "predictions.parquet",
                                                index=False)
                # Serialize the full state once. When this is the best epoch,
                # ckpt_best snapshots that exact ckpt_last instead of doing a
                # second 125--300 MB torch.save in the same Python process.
                self.save_ckpt(self.ckpt_last, model, opt, sched, scaler, ep + 1, vm)
                if is_best:
                    atomic_clone_file(self.ckpt_last, self.ckpt_best)
                self.last_epoch = ep + 1
                atomic_write_json(self.run_dir / "STATUS.json",
                                  {"status": "running", "epoch": ep + 1, "of": n_ep,
                                   "best_qwk": self.best_qwk, "iso": iso()})

                warn = ""
                if vm["val_qwk"] >= 0.995 or vm["val_acc"] >= 0.995:
                    warn = (f"   <-- PERFECT on {self.split_info['val_sessions']} tyres. "
                            "NOT a success signal; see split_health.json")
                print(f"  ep {ep+1:>3}/{n_ep}  loss {row['train_loss']:.4f}  "
                      f"val_acc {vm['val_acc']:.3f}  val_F1 {vm['val_f1_macro']:.3f}  "
                      f"val_QWK {vm['val_qwk']:.4f}{'  * best' if is_best else ''}  "
                      f"| {human_time(ep_s)}  dl {row['dataload_frac']:.0%}{warn}", flush=True)

                # push cadence: light every epoch, heavy+bulk every 10
                self.enqueue_light()
                self.enqueue_heavy()

                # Flush telemetry EVERY epoch, not every ten (Bug 23). Both
                # writers now append only what is new and then drop it, so the
                # process holds at most one epoch of samples instead of the
                # whole run. Doing it per epoch also means a hard kill loses
                # one epoch of trace rather than nine.
                if step_traces:
                    with open(self.run_dir / "telemetry" / "step_traces.jsonl", "a") as f:
                        for r in step_traces:
                            f.write(json.dumps(r) + "\n")
                    step_traces.clear()
                self.mon.dump()
                if (ep + 1) % 10 == 0 or (ep + 1) == n_ep:
                    self.enqueue_bulk()
                self.sess.registry.emit(self.run_id, "running", account=self.sess.account,
                                        epoch=ep + 1, best_qwk=self.best_qwk,
                                        wall_s=self.wall_seconds)
                self.sess.maybe_push(f"epoch {ep+1}")

                # A hard host-RAM kill produces no Python exception and hence
                # no emergency callback. Stop while we still have enough
                # headroom to publish the just-written checkpoint.
                #
                # Bug 22: measure NOW, after returning freed arenas to the
                # kernel -- not the epoch's transient peak. The checkpoint we
                # just wrote and handed to the uploader is exactly the spike
                # that used to trip this, and it is released by the time the
                # next epoch starts.
                try:
                    ram_peak = float(row.get("ram_percent_peak", 0.0))
                except (TypeError, ValueError):
                    ram_peak = 0.0
                ram_before, ram_now = host_ram_headroom()
                row["ram_percent_after_release"] = ram_now
                mem = memory_report()
                row["mem_used_gb"] = mem["used_gb"]
                row["mem_limit_gb"] = mem["limit_gb"]
                row["mem_source"] = mem["source"]
                row["mem_proc_rss_gb"] = mem["proc_rss_gb"]
                row["mem_children_rss_gb"] = mem["children_rss_gb"]
                # The first append protects metrics if checkpointing is killed.
                # Update that same epoch by name now that the post-checkpoint,
                # post-release memory fields exist (Bug 28 telemetry gap).
                append_epoch_row(self.hist_path, row)
                if ram_now >= HOST_RAM_PAUSE_PERCENT:
                    # Say WHERE the memory is. "89.6%" alone is not actionable;
                    # "this process holds 4 GB and something else holds 24" is.
                    _print("RAM", f"{ram_now:.1f}% of {mem['limit_gb']:.0f} GB "
                                  f"[{mem['source']}] after releasing (epoch peak "
                                  f"{ram_peak:.1f}%) -- this process "
                                  f"{mem['proc_rss_gb']:.1f} GB, {mem['n_children']} "
                                  f"child proc {mem['children_rss_gb']:.1f} GB, "
                                  f"rest {max(0.0, mem['used_gb'] - mem['proc_rss_gb'] - mem['children_rss_gb']):.1f} GB")
                if ep + 1 < n_ep and ram_now >= HOST_RAM_PAUSE_PERCENT:
                    status = "paused"
                    pause_reason = "host_ram_guard"
                    _print("RAM", f"host RAM {ram_now:.1f}% after epoch {ep+1}; "
                                  "pausing before the kernel is killed. Re-run to resume.")
                    break
                if ram_peak >= HOST_RAM_PAUSE_PERCENT and ram_now < HOST_RAM_PAUSE_PERCENT:
                    _print("RAM", f"epoch {ep+1} peaked at {ram_peak:.1f}% but sits at "
                                  f"{ram_now:.1f}% now -- transient, continuing")

                if self.sess.guard.near_limit():
                    _print("WATCHDOG", f"{self.sess.guard.elapsed_h:.1f} h elapsed -- pausing cleanly")
                    status = "paused"
                    pause_reason = "session_watchdog"
                    break
        except KeyboardInterrupt:
            status = "paused"
            pause_reason = "keyboard_interrupt"
            _print("TRAIN", "interrupted -- flushing")
        except Exception as e:
            status = "failed"
            cuda_restart_required = fatal_cuda_error(e)
            # Record WHAT failed, not just that something did. Twenty-six runs
            # were marked 'failed' with no way to tell a disk-full from a CUDA
            # OOM from a bad batch, so there was nothing to fix.
            err_type, err_msg = type(e).__name__, str(e)[:400]
            traceback.print_exc()
            atomic_write_text(self.run_dir / "ERROR.txt", traceback.format_exc())
            atomic_write_json(self.run_dir / "ERROR.json",
                              {"type": err_type, "message": err_msg,
                                "epoch": self.start_epoch, "iso": iso(),
                                "cuda_restart_required": cuda_restart_required,
                                "runtime_cuda_memory_format": memory_format_name,
                                "runtime_cuda_safety_revision": CUDA_SAFETY_REVISION,
                                "disk_free_gb_stage": round(
                                    shutil.disk_usage(self.sess.stage_dir).free / 1e9, 2)})
            self.sess.uploader.enqueue(self.run_dir / "ERROR.json",
                                       self.rp("ERROR.json"), force=True)
            self.sess.uploader.enqueue(self.run_dir / "ERROR.txt",
                                       self.rp("ERROR.txt"), force=True)
            _print("TRAIN", f"FAILED with {err_type}: {err_msg[:160]}")
            if cuda_restart_required:
                _print("CUDA", "the CUDA context is no longer safe. The failure was "
                               "pushed to HF; restart the Kaggle session before retrying.")
            else:
                _print("TRAIN", "the checkpoint is intact -- re-run this notebook and "
                                "it resumes from the last completed epoch")
        finally:
            if self.mon:
                self.mon.stop()
            _shutdown_loader(tr_dl)
            _shutdown_loader(va_dl)
            if step_traces:
                # APPEND. Bug 23: this used to open "w" and rewrite, which
                # truncated everything the per-epoch flush had already written.
                with open(self.run_dir / "telemetry" / "step_traces.jsonl", "a") as f:
                    for r in step_traces:
                        f.write(json.dumps(r) + "\n")
                step_traces.clear()
            release_host_memory()

        summary = {"run_id": self.run_id, "status": status, "arch": cfg["arch"],
                   "technique": cfg["technique"], "fold": cfg["fold"], "seed": cfg["seed"],
                   "stage": cfg["stage"], "best_val_qwk": self.best_qwk,
                   "epochs_trained": n_ep if status == "completed" else self.last_epoch,
                   "epochs_planned": n_ep, "n_params_total": n_all,
                   "total_wall_seconds": self.wall_seconds,
                   "total_energy_wh": self.energy_joules / 3600.0,
                   "config_hash": cfg["config_hash"], "account": self.sess.account,
                   "pause_reason": pause_reason,
                   "runtime_loader_num_workers": int(tr_dl.num_workers),
                   "runtime_loader_pin_memory": bool(tr_dl.pin_memory),
                   "runtime_memory_safety_revision": MEMORY_SAFETY_REVISION,
                   "runtime_hf_commit_policy_revision": HF_COMMIT_POLICY_REVISION,
                   "runtime_epoch_history_schema_revision": EPOCH_HISTORY_SCHEMA_REVISION,
                   "runtime_cuda_memory_format": memory_format_name,
                   "runtime_cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                   "runtime_cuda_safety_revision": CUDA_SAFETY_REVISION,
                   "runtime_scheduler_safety_revision": SCHEDULER_SAFETY_REVISION,
                   "runtime_process_isolation_revision": PROCESS_ISOLATION_REVISION,
                   "runtime_isolated_child": bool(cfg.get("_isolated_child", False)),
                   "cuda_restart_required": cuda_restart_required,
                   "lib_version": __version__, "finished_iso": iso(),
                   "val_sessions": self.split_info["val_sessions"],
                   "val_images": self.split_info["val_images"],
                   "cross_fold_tyre_flags": len(self.split_info["cross_fold_tyre_flags"])}
        if self.hist_path.exists():
            h = read_epoch_history(self.hist_path, repair=True)
            if len(h):
                b = h.loc[h.val_qwk.idxmax()]
                summary.update({
                    "best_epoch": int(b.epoch),
                    "best_val_f1_macro": float(b.val_f1_macro),
                    "best_val_acc": float(b.val_acc),
                    "best_val_mae_class": float(b.val_mae_class),
                    "final_val_qwk": float(h.iloc[-1].val_qwk),
                    "final_val_f1_macro": float(h.iloc[-1].val_f1_macro),
                    "nan_or_inf_batches_total": int(h.nan_or_inf_batches.sum()),
                    "amp_scale_decreases_total": int(h.amp_scale_decreases.sum()),
                    "peak_ram_gb": float(h.get("proc_rss_gb_peak", pd.Series([np.nan])).max()),
                    "mean_dataload_frac": float(h.dataload_frac.mean()),
                })
        pd.DataFrame([summary]).to_csv(self.run_dir / "metrics" / "final.csv", index=False)
        atomic_write_json(self.run_dir / "summary.json", summary)
        # 'epoch' explicitly, not only summary's 'epochs_trained' -- STATUS.json
        # is what RemoteInventory reads to decide where a resume starts, and it
        # must not depend on which of several near-synonyms happens to be there.
        atomic_write_json(self.run_dir / "STATUS.json",
                          {"status": status, "iso": iso(), "epoch": self.last_epoch,
                           "of": n_ep, "error_type": err_type, **summary})

        self.enqueue_light(); self.enqueue_heavy(); self.enqueue_bulk()
        self.sess.registry.emit(self.run_id, status, account=self.sess.account,
                                worker=self.sess.worker_id, best_qwk=self.best_qwk,
                                epochs=summary.get("epochs_trained"), wall_s=self.wall_seconds,
                                error_type=err_type, error_msg=err_msg)
        # a model finishing is a major step -- push now, do not wait for the cycle
        self.sess.uploader.flush(reason=f"run {status}: {self.run_id}")
        _print("TRAIN", f"{self.run_id}  ->  {status}  best QWK {self.best_qwk:.4f}  "
                        f"({human_time(self.wall_seconds)})")
        # Release model/optimizer/DataParallel and CUDA caches before the next
        # architecture is constructed in this same long-lived notebook.
        del model, opt, sched, scaler, tr_dl, va_dl
        release_host_memory()
        if torch.cuda.is_available():
            # A fatal asynchronous CUDA fault poisons the context; even
            # empty_cache can then raise a second, misleading exception and
            # hide the already-published root failure.
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
        return summary


# --------------------------------------------------------------------------
# 11. Session -- the façade the notebooks talk to
# --------------------------------------------------------------------------

HF_REPO_DEFAULT = "Shanmuk4622/tyre-wear-study"

# Standard recipe. Held FIXED across the whole architecture sweep -- if the
# recipe changes mid-sweep the comparison stops being a comparison.
RECIPE = dict(
    input_resolution=384,
    batch_size=32,
    head_type="coral",
    loss_name="coral_bce",
    label_smoothing=0.0,
    sampler_name="session_balanced",
    optimizer_name="adamw",
    lr_initial=3e-4,
    weight_decay=0.05,
    scheduler_name="cosine",
    warmup_epochs=5,
    max_epochs=60,          # EQUAL BUDGET. No early stopping, ever.
    grad_clip=5.0,
    pretrained=True,
    finetune_depth="full",
    preprocessing="raw",
    roi_mode="full_frame",
    augment_policy="dataset_v1_1",
    precision="fp16",
    num_workers=2,
)


def staging_root() -> Path:
    """Where checkpoints and telemetry are written during a session.

    `/kaggle/working` is capped at 20 GB and that cap is the size of your
    OUTPUT, not your scratch. A vgg16bn checkpoint is ~1.6 GB and we keep two
    per run, so nine vgg runs staged there is 29 GB and the session dies with
    a disk error partway through -- which is what turned finished training
    into `status: failed`.

    `/kaggle/temp` is on the big disk and is not part of the output cap. The
    previous version only used it `if Path("/kaggle/temp").exists()`, and on
    the current Kaggle image it does not exist until something creates it, so
    every session silently fell back to `./_work` inside /kaggle/working.
    Create it instead of testing for it.
    """
    for cand in ("/kaggle/temp", "/tmp", "."):
        try:
            p = Path(cand) / "tyre_study"
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".writable"
            probe.write_text("ok")
            probe.unlink()
            free = shutil.disk_usage(p).free / 1e9
            _print("DISK", f"staging {p}  ({free:.0f} GB free)")
            if free < 20:
                _print("DISK", "WARNING: under 20 GB free. Large checkpoints "
                               "(vgg16bn, maxvit) may not fit.")
            return p
        except Exception:
            continue
    raise RuntimeError("no writable staging directory found")


class Session:
    def __init__(self, account: str, worker_id: int = 0, num_workers: int = 1,
                 stage: str = "a", hf_repo: str = HF_REPO_DEFAULT,
                 enable_hf: bool = True, session_limit_h: float = 8.5,
                 push_interval_min: int = 30, rate_limit: int | None = None,
                 data_hint: str | None = None):
        self.account = account
        self.worker_id = int(worker_id)
        self.num_workers = int(num_workers)
        self.stage = stage
        self.session_id = hashlib.sha256(f"{account}{now()}".encode()).hexdigest()[:6]
        self.host = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "local")

        # One HuggingFace account for the whole team, so the 128/hr budget is
        # SHARED. Cap each worker at 128/num_workers with headroom.
        if rate_limit is None:
            rate_limit = max(6, int(100 / max(1, num_workers)))

        self.stage_dir = staging_root()

        token = None
        if enable_hf:
            try:
                from kaggle_secrets import UserSecretsClient
                token = UserSecretsClient().get_secret("HF_TOKEN")
            except Exception:
                token = os.environ.get("HF_TOKEN")

        self.uploader = Uploader(hf_repo, token, "dataset",
                                 interval_s=push_interval_min * 60,
                                 rate_limit=rate_limit, enabled=enable_hf)
        self.uploader.start()
        self.registry = Registry(self.stage_dir, self.uploader, account, worker_id, self.session_id)
        self.inventory = RemoteInventory(self.uploader, self.stage_dir)
        self.guard = LifecycleGuard(self._emergency_flush, session_limit_h).install()
        self.data_root: Path | None = None
        self._last_manual_push = now()

        if not (0 <= self.worker_id < max(1, self.num_workers)):
            raise ValueError(
                f"WORKER_ID={self.worker_id} is outside 0..{self.num_workers - 1}. "
                f"With NUM_WORKERS={self.num_workers} nothing would ever be assigned to you.")

        print()
        _print("SESSION", f"account={account}  worker={worker_id}/{num_workers}  "
                          f"stage={stage}  id={self.session_id}")
        if self.num_workers == 1:
            _print("SESSION", "MODE=ONE NOTEBOOK: this session owns every unfinished run; "
                              "there are no reserved shards or takeover waits")
        else:
            _print("SESSION", f"MODE={self.num_workers} PARALLEL NOTEBOOKS: each account "
                              "starts with one static shard, then safely helps when idle")
        _print("SESSION", f"staging {self.stage_dir}  |  hf {'ON' if self.uploader.enabled else 'OFF'}  "
                          f"|  cap {rate_limit}/hr  |  push every {push_interval_min} min")
        _print("SESSION", "NUM_WORKERS assigns each FRESH run to one static owner. "
                          "Completed/resumable state still comes from HuggingFace.")
        print()

    # -- lifecycle --------------------------------------------------------
    def _emergency_flush(self, reason: str):
        _print("FLUSH", f"emergency flush ({reason})")
        with contextlib.suppress(Exception):
            self.uploader.flush(timeout=900, reason=reason)

    def maybe_push(self, reason: str = "", min_gap_min: float = 30.0):
        """Background thread pushes on its own cycle; this is the explicit
        'a major step just finished' push."""
        if now() - self._last_manual_push >= min_gap_min * 60:
            self._last_manual_push = now()
            self.uploader.flush(timeout=600, reason=reason or "interval")

    def push_now(self, reason: str = "cell complete"):
        """Call at the end of every important cell."""
        self._last_manual_push = now()
        return self.uploader.flush(timeout=900, reason=reason)

    def finish(self):
        _print("SESSION", "final flush -- blocking until HuggingFace confirms")
        ok = self.uploader.flush(timeout=1800, reason="session finish")
        self.uploader.stop()
        _print("SESSION", f"done. commits={self.uploader.commits} "
                          f"failures={self.uploader.failures} "
                          f"pushed={self.uploader.bytes_pushed/1e6:.0f} MB")
        return ok

    def confirm_on_hf(self, run_ids):
        """Draining the upload queue is NOT the same as the files being on
        HuggingFace. Ask the repository before you close the tab.

        Completion is judged the same way everywhere else judges it -- by
        `STATUS.json`'s status field, via RemoteInventory -- rather than by the
        presence of a file. Presence was the old test, and because
        `summary.json` was never uploaded (Bug 14) it reported all 36 finished
        runs as merely RESUMABLE.
        """
        self.inventory.refresh(list(run_ids), verbose=False)
        rows = []
        for rid in run_ids:
            want = [f"runs/{rid}/metrics/epochs.csv", f"runs/{rid}/metrics/final.csv",
                    f"runs/{rid}/checkpoints/ckpt_last.pt", f"runs/{rid}/STATUS.json"]
            missing = [p for p in want if p not in self.inventory.files]
            st = self.inventory.state(rid)
            if st == "completed":
                state = "FINISHED"
            elif st == "resumable":
                state = "RESUMABLE"
            else:
                state = "AT RISK"
            rows.append({"run_id": rid, "on_hf": state, "epoch": self.inventory.epoch(rid),
                         "missing_files": len(missing)})
        df = pd.DataFrame(rows)
        n_risk = int((df.on_hf == "AT RISK").sum())
        print(df.to_string(index=False))
        print(f"\nFINISHED {int((df.on_hf=='FINISHED').sum())}   "
              f"RESUMABLE {int((df.on_hf=='RESUMABLE').sum())}   AT RISK {n_risk}")
        print("FINISHED and RESUMABLE are both safe to close.")
        return df

    def aggregate_remote(self, run_ids=None, verbose: bool = True) -> pd.DataFrame:
        """The real results table: every worker's `final.csv`, pulled from HF.

        `aggregate()` globs the local staging directory, so on a four-account
        run each account produces a table of the eleven runs it happened to do.
        Nobody ever sees all thirty-six in one place, which is the only view
        that answers anything.

        Runs from before lib v2 lack `val_sessions` / `cross_fold_tyre_flags`,
        so the concat is deliberately outer-joined and those cells come back
        NaN rather than the rows being dropped.
        """
        if not self.uploader.enabled:
            _print("AGG", "HuggingFace off -- use aggregate() for local runs")
            return pd.DataFrame()
        from huggingface_hub import hf_hub_download
        files = set(self.uploader._api.list_repo_files(
            self.uploader.repo_id, repo_type=self.uploader.repo_type))
        want = sorted(p for p in files
                      if p.startswith("runs/") and p.endswith("/metrics/final.csv")
                      and (run_ids is None or p.split("/")[1] in set(run_ids)))
        rows = []
        for rp in want:
            try:
                p = hf_hub_download(self.uploader.repo_id, rp,
                                    repo_type=self.uploader.repo_type,
                                    token=self.uploader.token, local_dir=str(self.stage_dir))
                rows.append(pd.read_csv(p))
            except Exception as e:
                _print("AGG", f"{rp}: {type(e).__name__}: {e}")
        if not rows:
            return pd.DataFrame()
        df = pd.concat(rows, ignore_index=True, sort=False)
        out = self.stage_dir / "tables"
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "all_runs_remote.csv", index=False)
        self.uploader.enqueue(out / "all_runs_remote.csv", "tables/all_runs_remote.csv", force=True)
        if verbose:
            _print("AGG", f"{len(df)} run(s) from {df.account.nunique()} account(s)")
            dup = df[df.duplicated("run_id", keep=False)]
            if len(dup):
                _print("AGG", f"WARNING: {dup.run_id.nunique()} run_id(s) trained more than "
                              f"once -- {sorted(dup.run_id.unique())}")
        return df

    def honest_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage A results with the leak-flagged folds separated out.

        `best_val_*` is chosen by looking at the validation fold, and that fold
        is four tyres. Selecting on it and then reporting it is circular. The
        fixed-budget number -- `final_val_*` at epoch 60, chosen by nobody --
        is the one that can be compared with a baseline, so both are shown
        side by side and the gap between them is a result in its own right.
        """
        if not len(df):
            return df
        d = df.copy()
        d["leak_flagged"] = d.get("cross_fold_tyre_flags", 0).fillna(0) > 0
        g = (d.groupby(["arch", "fold"])
               .agg(n=("run_id", "nunique"),
                    leak=("leak_flagged", "max"),
                    best_qwk=("best_val_qwk", "mean"),
                    best_f1=("best_val_f1_macro", "mean"),
                    final_f1=("final_val_f1_macro", "mean"),
                    best_epoch=("best_epoch", "median"))
               .round(3).reset_index())
        print(g.to_string(index=False))
        clean = g[~g.leak.astype(bool)]
        if len(clean):
            print(f"\nOn folds with NO cross-fold tyre flag:")
            print(f"  mean best  macro-F1 (selected on the val fold) {clean.best_f1.mean():.3f}")
            print(f"  mean final macro-F1 (fixed 60 epochs)          {clean.final_f1.mean():.3f}")
            print(f"  strongest trivial baseline on those folds      "
                  f"{max(BASELINES['frame_occupancy'][f'f{int(f)}'] for f in clean.fold.unique()):.3f}")
            print("\nThe gap between the two model rows is selection, not learning.")
        return g

    # -- data -------------------------------------------------------------
    def prepare_data(self, hint: str | None = None) -> Path:
        root = find_dataset_root(hint)
        if root is None:
            raise FileNotFoundError(
                "Dataset not found. Sidebar -> Add Input -> shanmuk4622/tire-dataset-prepared")
        self.data_root = root
        v = read_json(root / "VERSION.json", {})
        _print("DATA", f"root {root}")
        _print("DATA", f"{v.get('clean_images','?')} clean / {v.get('synthetic_derivatives','?')} derivatives"
                       f" / {v.get('provisional_session_groups','?')} sessions")
        return root

    def environment(self) -> dict:
        import torch
        env = {"python": sys.version.split()[0], "torch": torch.__version__,
               "cuda": torch.version.cuda, "numpy": np.__version__, "pandas": pd.__version__,
               "lib_version": __version__, "account": self.account,
               "worker_id": self.worker_id, "session_id": self.session_id,
               "host": self.host, "iso": iso()}
        with contextlib.suppress(Exception):
            import timm; env["timm"] = timm.__version__
        with contextlib.suppress(Exception):
            env["gpus"] = [{"name": torch.cuda.get_device_name(i),
                            "mem_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
                           for i in range(torch.cuda.device_count())]
        return env

    # -- configs ----------------------------------------------------------
    def config(self, arch: str, fold: int, seed: int, technique: str = "base",
               stage: str | None = None, **overrides) -> dict:
        stage = stage or self.stage
        spec = ZOO.get(arch, {})
        cfg = dict(RECIPE)
        cfg["input_resolution"] = spec.get("res", cfg["input_resolution"])
        cfg["batch_size"] = spec.get("bs", cfg["batch_size"])
        cfg.update(overrides)
        cfg.update(dict(arch=arch, fold=int(fold), seed=int(seed),
                        technique=technique, stage=stage))
        cfg["run_id"] = f"{stage}-{arch}-{technique}-f{fold}-s{seed}"
        cfg["config_hash"] = config_hash(cfg)
        return cfg

    def configs(self, archs, folds=(0, 1, 2), seeds=(1, 2, 3), technique="base", **ov):
        return [self.config(a, f, s, technique, **ov) for a in archs for f in folds for s in seeds]

    # -- planning ---------------------------------------------------------
    def sync_state(self, run_ids=None, verbose: bool = True) -> int:
        n = self.registry.pull(self.uploader)
        if verbose:
            st = self.registry.latest()
            done = sum(1 for v in st.values() if v["state"] == "completed")
            _print("SYNC", f"pulled {n} shard(s); registry knows {len(st)} run(s), {done} completed")
        self.inventory.refresh(run_ids, verbose=verbose)
        return n

    def reconcile(self, run_ids) -> pd.DataFrame:
        """What the repository actually holds for these runs, and what this
        session will therefore do with each one.

        Run it whenever a plan surprises you. It answers the only question
        that matters -- am I about to redo work that is already done -- from
        the files rather than from anybody's bookkeeping.
        """
        self.inventory.refresh(run_ids, verbose=False)
        df = self.inventory.table(run_ids)
        reg = self.registry.latest()
        df["registry"] = df.run_id.map(lambda r: reg.get(r, {}).get("state", "-"))
        df["action"] = df.run_id.map(
            lambda r: {"completed": "skip", "resumable": "resume", "absent": "train"}[
                self.inventory.state(r)])
        counts = df.action.value_counts().to_dict()
        print(df.to_string(index=False))
        print(f"\nskip {counts.get('skip', 0)}   resume {counts.get('resume', 0)}   "
              f"train from scratch {counts.get('train', 0)}")
        if (df.registry == "failed").any():
            n = int((df.registry == "failed").sum())
            print(f"\n{n} run(s) the registry calls 'failed' -- look at the `state` "
                  "column, not that one.\nA failure at epoch 47 still has a checkpoint "
                  "at epoch 47 and resumes from there.")
        return df

    def status(self) -> pd.DataFrame:
        st = self.registry.latest()
        if not st:
            print("registry empty -- nothing has run yet")
            return pd.DataFrame()
        df = pd.DataFrame([{"run_id": k, "state": v["state"], "account": v.get("account"),
                            "epoch": v.get("epoch"), "best_qwk": v.get("best_qwk")}
                           for k, v in sorted(st.items())])
        print(df.to_string(index=False))
        return df

    def claim_or_yield(self, run_id: str, settle_s: float = 25.0) -> tuple[bool, str]:
        """Claim a run another worker owns, without a lock server.

        Taking work off another account's shard is the only way to stop a
        worker idling while its neighbours have twenty runs left (Bug 24). It
        is also exactly how v2 trained `a-vgg16bn-base-f1-s1` twice (Bug 13),
        so it needs more than "the registry looked free a moment ago".

        Two phases, which is the standard answer when there is nowhere to put
        a lock:

          1. Pull the registry, check nobody holds it, write our claim, and
             **flush it immediately** so it is visible to everyone.
          2. Wait out the race window, pull again, and look at every claim
             written for this run in that window. If more than one account
             claimed it, the lowest account name wins.

        Both sides compute step 2 from the same bytes and reach the same
        answer, so exactly one proceeds and the other moves on. The cost is one
        commit and ~30 s, paid only by a worker that would otherwise be idle.
        """
        self.registry.pull(self.uploader)
        if self.inventory.refresh([run_id], verbose=False).state(run_id) == "completed":
            return False, "finished while I was deciding"
        ok, why = self.registry.can_claim(run_id, self.account, stale_s=2700)
        if not ok:
            return False, why

        self.registry.emit(run_id, "claimed", account=self.account, worker=self.worker_id)
        self.uploader.flush(timeout=120, reason=f"claim {run_id}")

        t_claim = now()
        time.sleep(settle_s + random.uniform(0.0, 10.0))
        self.registry.pull(self.uploader)

        rivals = [e for e in self.registry.entries()
                  if e.get("run_id") == run_id and e.get("state") == "claimed"
                  and abs(float(e.get("ts", 0.0)) - t_claim) < 600.0
                  and e.get("account")]
        if rivals:
            winner = min(str(e["account"]) for e in rivals)
            if winner != self.account:
                return False, f"yielded to {winner} (claimed the same run)"
        return True, "claimed after settling"

    def plan(self, run_ids, title: str = "plan", steal_stale: bool = False,
             refresh: bool = True, takeover_when_idle: bool = True):
        """Decide what to do this session.

        Ownership is computed over the FULL run list, never over the
        outstanding subset, so a fresh run keeps the same owner as its
        neighbours finish. Ownership reserves fresh work; completion and
        progress still come from `self.inventory`, which is identical for
        every worker. Changing NUM_WORKERS changes the fresh-work owner map,
        never whether completed work is skipped or a checkpoint is resumed.
        """
        if refresh:
            self.inventory.refresh(run_ids, verbose=True)
        inv = self.inventory
        owner = assign_workers(run_ids, self.num_workers, "cost")   # STATIC costs
        if self.num_workers > 1 and (steal_stale or takeover_when_idle):
            # Planning against a registry that was never pulled is how fresh
            # absent work was mistaken for abandoned work. One pull gives every
            # worker the same recent claims before ownership/takeover decisions.
            self.registry.pull(self.uploader)
        latest = self.registry.latest()

        # The repository is authoritative; the registry can only ADD
        # completions (for a run whose STATUS.json push was lost).
        done = {r for r in run_ids if inv.state(r) == "completed"}
        done |= {r for r in run_ids if latest.get(r, {}).get("state") == "completed"}

        mine, stolen, busy = [], [], []
        for r in sorted(run_ids):
            if r in done:
                continue
            if owner[r] == self.worker_id:
                mine.append(r)
            elif takeover_when_idle and self.num_workers > 1 and not steal_stale:
                # ⚠ Bug 24. `steal_stale=False` made every run owned by someone
                # else permanently untouchable, so a worker that finished its
                # 27-run shard printed "will run 0 run(s)" and the notebook
                # ended -- while the other accounts still had twenty runs each.
                # Reported as "out of 4, 2 are running and 2 stopped".
                #
                # The shard is LPT-balanced on ESTIMATED cost and skewed further
                # by pauses and resumes, so shards always finish at different
                # times. Some worker always runs dry first.
                #
                # These go in a separate pool that is only touched once `mine`
                # is empty, and only through the two-phase claim in
                # `claim_or_yield`. That is what makes it safe: v2 stole
                # aggressively and trained vgg16bn-f1-s1 twice; v4 fixed that by
                # refusing all takeover, which is how we got here.
                ev = latest.get(r)
                if ev is not None and ev.get("state") in ("running", "claimed") \
                        and now() - float(ev.get("ts", 0)) < 2700:
                    busy.append(r)          # someone is genuinely on it
                else:
                    stolen.append(r)
            elif steal_stale and self.num_workers > 1:
                # An absent run is not stale work: it is fresh work reserved by
                # the static owner map.  Treating "no event" as "dead worker"
                # made all four accounts select the same first outstanding run
                # during a simultaneous start.  Only a real, old registry event
                # is eligible for takeover.
                event = latest.get(r)
                if event is None:
                    busy.append(r)
                else:
                    ok, why = self.registry.can_claim(r, self.account, stale_s=2700)
                    (stolen if ok else busy).append(r)
            elif steal_stale:
                mine.append(r)          # single worker: everything is mine
            else:
                busy.append(r)

        # Finish what is half-done before starting anything new. A run at
        # epoch 52 of 60 is eight minutes from being a result; a fresh one is
        # half an hour from being anything at all.
        key = lambda r: (0 if inv.state(r) == "resumable" else 1, -inv.epoch(r), r)
        mine.sort(key=key)
        stolen.sort(key=key)

        plan = type("Plan", (), {})()
        plan.mine, plan.stolen, plan.busy = mine, stolen, busy
        plan.scheduler_revision = SCHEDULER_SAFETY_REVISION
        plan.done = sorted(done & set(run_ids))
        # Offset each worker's scan of the shared pool by its own id, so two
        # workers going idle at the same moment do not both reach for the same
        # run before the two-phase claim has to arbitrate.
        if stolen and self.num_workers > 1:
            k = self.worker_id % len(stolen)
            stolen = stolen[k:] + stolen[:k]
        plan.stolen = stolen
        plan.order = mine + stolen                    # own work ALWAYS first
        plan.n_mine = len(mine)                       # everything after is takeover
        plan.resumable = [r for r in plan.order if inv.state(r) == "resumable"]

        remaining = sum(cost_of(r) * (1 - min(0.98, inv.epoch(r) / 60.0)) for r in plan.order)
        print(f"\n=== {title} ===")
        print(f"  total in this notebook : {len(run_ids)}")
        print(f"  already finished       : {len(plan.done)}   (skipped)")
        print(f"  resuming mid-run       : {len(plan.resumable)}")
        print(f"  starting from scratch  : {len(plan.order) - len(plan.resumable)}")
        if stolen:
            print(f"  available if I go idle : {len(stolen)}   "
                  f"(claimed one at a time, only after my own {len(mine)})")
        if busy:
            label = ("another worker is on/reserved it" if steal_stale else
                     "reserved for other static owners")
            print(f"  {label:<31}: {len(busy)}")
        print(f"  est. GPU time for me   : ~{remaining/60:.1f} h "
              f"(credits partly-done runs)")
        print(f"  -> will run {len(plan.order)} run(s) this session\n")
        return plan

    # -- execution --------------------------------------------------------
    def _run_one_isolated(self, cfg: dict) -> dict:
        """Train one model in a disposable Python process.

        Public NB06 telemetry showed the long-lived Jupyter kernel retaining
        0.17--0.30 GB of RSS after every epoch despite loader shutdown,
        ``gc.collect`` and ``malloc_trim``. After two completed models the
        third reached the 88% guard and the whole cell stopped. A child process
        gives Linux a hard reclamation boundary: model, optimiser, checkpoint
        serialization buffers, CUDA context and library caches all disappear
        when that one run exits. The parent keeps the plan and immediately
        resumes the same HF checkpoint if the child paused under pressure.
        """
        rid = cfg["run_id"]
        iso_dir = Path(self.stage_dir) / "_isolated" / self.session_id
        iso_dir.mkdir(parents=True, exist_ok=True)
        nonce = hashlib.sha256(f"{rid}{now()}{random.random()}".encode()).hexdigest()[:10]
        payload_path = iso_dir / f"{nonce}.input.json"
        result_path = iso_dir / f"{nonce}.result.json"
        elapsed = now() - self.guard.t_start
        remaining_h = max(0.25, (self.guard.session_limit_s - elapsed) / 3600.0)
        payload = {
            "cfg": cfg,
            "account": self.account,
            "worker_id": self.worker_id,
            "num_workers": self.num_workers,
            "stage": self.stage,
            "hf_repo": self.uploader.repo_id,
            "enable_hf": self.uploader.enabled,
            "rate_limit": self.uploader.limiter.limit,
            "push_interval_min": self.uploader.interval_s / 60.0,
            "session_limit_h": remaining_h,
            "data_root": str(self.data_root),
        }
        atomic_write_json(payload_path, payload)
        _print("ISOLATE", f"{rid}: starting a clean child process "
                          f"(memory isolation {PROCESS_ISOLATION_REVISION}, "
                          f"{remaining_h:.1f} h session time left)")
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--isolated-train", str(payload_path), str(result_path)]
        child_env = os.environ.copy()
        if self.uploader.token:
            # Environment inheritance avoids putting the secret on the command
            # line/process list while guaranteeing the clean child can publish.
            child_env["HF_TOKEN"] = self.uploader.token
        proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parent),
                                env=child_env)
        try:
            returncode = proc.wait()
        except KeyboardInterrupt:
            # Give the child the same graceful-stop path as an interactive
            # notebook: checkpoint, publish, then let the interrupt return.
            with contextlib.suppress(Exception):
                proc.send_signal(signal.SIGINT)
            with contextlib.suppress(Exception):
                proc.wait(timeout=900)
            self.push_now(f"parent interrupted during {rid}")
            raise

        summary = read_json(result_path, None)
        with contextlib.suppress(Exception):
            payload_path.unlink()
        with contextlib.suppress(Exception):
            result_path.unlink()
        release_host_memory()

        # A hard-killed child may not have time to write its tiny result file,
        # while its previous epoch checkpoint is already public. Reconcile the
        # repository before deciding whether any work was lost.
        self.inventory.refresh([rid], verbose=False)
        if summary is None:
            state = self.inventory.state(rid)
            epoch = self.inventory.epoch(rid)
            status = "completed" if state == "completed" else (
                "paused" if state == "resumable" else "failed")
            summary = {
                "run_id": rid, "arch": cfg["arch"], "fold": cfg["fold"],
                "seed": cfg["seed"], "status": status,
                "epochs_trained": epoch, "pause_reason": "isolated_child_exit",
                "cuda_restart_required": False,
                "error_type": f"child_exit_{returncode}",
            }
        _print("ISOLATE", f"{rid}: child exited rc={returncode}; "
                          f"status={summary.get('status')} epoch="
                          f"{summary.get('epochs_trained', self.inventory.epoch(rid))}. "
                          "Its process memory is now fully reclaimed.")
        return summary

    def run_all(self, cfgs, title: str = "training", steal_stale: bool = False,
                takeover_when_idle: bool = True, isolate_runs: bool = False) -> list[dict]:
        by_id = {c["run_id"]: c for c in cfgs}
        plan = self.plan(list(by_id), title=title, steal_stale=steal_stale,
                         takeover_when_idle=takeover_when_idle)
        out = []
        n_mine = getattr(plan, "n_mine", len(plan.order))
        announced_idle = False
        for i, rid in enumerate(plan.order, 1):
            # This guard must apply to own work too. Isolated children have
            # fresh clocks of their own, but the Kaggle session does not.
            if self.guard.near_limit(margin_min=45):
                _print("WATCHDOG", "less than 45 minutes remain in this Kaggle "
                                   "session; not starting another model")
                break
            # The repository decides. Only ask the registry whether somebody
            # is on it RIGHT NOW, and only when more than one worker exists.
            if self.num_workers > 1:
                # Another account may have finished this in the last few hours.
                # Narrowed to one run: one listing + one small download.
                self.inventory.refresh([rid], verbose=False)
            if self.inventory.state(rid) == "completed":
                _print("SKIP", f"{rid}: already finished on HuggingFace")
                continue
            if i > n_mine and not announced_idle:
                announced_idle = True
                print("\n" + "-" * 74)
                _print("IDLE", f"my own {n_mine} run(s) are done or running elsewhere. "
                               f"Taking work from the shared pool so this GPU is not "
                               f"parked while other accounts still have runs left.")
                print("-" * 74)
            if i > n_mine and self.num_workers > 1:
                # Takeover: two-phase claim (Bug 24). Costs one commit and ~30 s,
                # and only an otherwise-idle worker ever pays it.
                if self.guard.near_limit(margin_min=90):
                    _print("IDLE", "not enough session time left to start another "
                                   "model; stopping cleanly instead of half-training one")
                    break
                ok, whyc = self.claim_or_yield(rid)
                if not ok:
                    _print("SKIP", f"{rid}: {whyc}")
                    continue
                _print("IDLE", f"{rid}: {whyc}")
            elif self.num_workers > 1 and rid in getattr(plan, "stolen", ()):
                # ⚠ Bug 13. `can_claim` reads the LOCAL copy of the other
                # workers' registry shards, and those were last downloaded in
                # `sync_state` -- hours ago. So a run another account started
                # twenty minutes ago still looked idle, and got stolen.
                #
                # It happened: a-vgg16bn-base-f1-s1 was trained to completion
                # by acct1 AND acct2, same config_hash, ~1.4 GPU-hours burnt
                # twice. Only shows up if you notice one run has two owners.
                #
                # Own runs do not need this -- nobody else using the repaired
                # static schedule can be on them -- so pay the requests and
                # publish an immediate claim only when takeover was explicitly
                # enabled and this run is genuinely stolen.
                self.registry.pull(self.uploader)
                ok, held = self.registry.can_claim(rid, self.account, stale_s=2700)
                if not ok:
                    _print("SKIP", f"{rid}: {held}")
                    continue
            why = self.inventory.reason(rid)
            print("\n" + "=" * 74)
            _print("RUN", f"{i}/{len(plan.order)}  {rid}   ({why})")
            print("=" * 74)
            if i <= n_mine:
                self.registry.emit(rid, "claimed", account=self.account,
                                   worker=self.worker_id)
            if i <= n_mine and rid in getattr(plan, "stolen", ()):
                # A claim nobody can read is not a claim. `emit` only enqueues,
                # and the background cycle is 30 minutes -- long enough for a
                # second worker to start the same run and for both to be right
                # about what they could see. One commit, at the only moment it
                # buys anything.
                self.uploader.flush(timeout=120, reason=f"stolen claim {rid}")
            self.guard.reset()
            if isolate_runs:
                last_epoch = -1
                s = None
                for restart in range(1, 9):
                    s = self._run_one_isolated(by_id[rid])
                    why_pause = s.get("pause_reason")
                    epoch_now = int(s.get("epochs_trained") or
                                    self.inventory.epoch(rid) or 0)
                    if not (s.get("status") == "paused" and
                            why_pause == "host_ram_guard"):
                        break
                    if epoch_now <= last_epoch:
                        _print("ISOLATE", f"{rid}: RAM pause made no epoch progress; "
                                          "not retrying in a loop")
                        break
                    last_epoch = epoch_now
                    if self.guard.near_limit(margin_min=45):
                        _print("WATCHDOG", f"{rid}: checkpoint is safe at epoch "
                                           f"{epoch_now}; session is nearly over")
                        break
                    _print("ISOLATE", f"{rid}: child paused at epoch {epoch_now}. "
                                      "That process has exited, so its retained RAM is "
                                      "gone; resuming the SAME run in a fresh child.")
                assert s is not None
            else:
                s = Trainer(by_id[rid], self).run()
            out.append(s)
            if s["status"] == "completed":
                self.prune_local(rid)
            if s["status"] == "paused":
                why = s.get("pause_reason") or "safety pause"

                # Not every pause means the session is finished.
                #
                # v5 stopped the worker after ANY pause, to stop the old loop
                # marching into dozens of models after a host-RAM pause and
                # burning one HF commit on each. That was right about the
                # cascade and wrong about the scope: a RAM pause is a statement
                # about this moment, not about the session. Combined with the
                # peak-based trigger of Bug 22, one checkpoint-sized spike
                # ended an eight-hour session with eighteen runs untouched.
                #
                # So: free the run's memory, look again, and only stop if the
                # pressure is real. A watchdog pause or an interrupt still ends
                # the cell -- those genuinely mean there is no time left.
                if why == "host_ram_guard" and not isolate_runs:
                    release_host_memory()
                    ram_now = host_ram_percent()
                    if ram_now < HOST_RAM_RESUME_PERCENT:
                        _print("RUN", f"host RAM back to {ram_now:.1f}% (under "
                                      f"{HOST_RAM_RESUME_PERCENT:.0f}%) once this model was "
                                      "released -- continuing with the next run")
                        continue
                    _print("RUN", f"host RAM still {ram_now:.1f}% after releasing this "
                                  f"model. Stopping so the kernel is not killed.")
                elif why == "host_ram_guard" and isolate_runs:
                    _print("RUN", f"isolated child remained RAM-blocked at epoch "
                                  f"{s.get('epochs_trained')}; checkpoint is safe")
                _print("RUN", f"stopping worker after {why}. The checkpoint is on "
                              "HuggingFace; use a fresh Kaggle session and re-run "
                              "this notebook to resume at the next epoch.")
                break
            if s.get("cuda_restart_required"):
                # CUDA launch faults are process-fatal in practice. Continuing
                # would only mark unrelated models failed in a poisoned context.
                if isolate_runs:
                    _print("RUN", "fatal CUDA fault was contained inside the disposable "
                                  "child; the parent is clean and will continue with the "
                                  "next run. This run remains recorded for retry.")
                    continue
                _print("RUN", "stopping after a fatal CUDA fault. The error and "
                              "available checkpoint are on HuggingFace; restart "
                              "the Kaggle session before retrying.")
                break
        if out:
            df = pd.DataFrame([{k: s.get(k) for k in
                                ("run_id", "arch", "fold", "seed", "status",
                                 "best_val_qwk", "best_val_f1_macro", "best_val_acc",
                                 "epochs_trained", "total_wall_seconds", "total_energy_wh")}
                               for s in out])
            print("\n" + df.to_string(index=False))
        self.push_now("run_all complete")
        return out

    def prune_local(self, run_id: str) -> int:
        """Delete a finished run's local checkpoints, but only once the
        repository confirms it has them.

        Thirty-six runs staged at once is tens of gigabytes, and a session that
        runs out of disk at run 20 loses the GPU time for run 20 -- which is a
        silly way to lose an afternoon. Verify first, then delete: the point of
        keeping one copy is that there is always one copy.
        """
        want = [f"runs/{run_id}/checkpoints/ckpt_best.pt",
                f"runs/{run_id}/checkpoints/ckpt_last.pt"]
        missing = self.uploader.verify_present(want) if self.uploader.enabled else want
        if missing:
            _print("DISK", f"{run_id}: keeping local checkpoints -- "
                           f"{len(missing)} not confirmed on HuggingFace yet")
            return 0
        freed = 0
        for rel in ("checkpoints/ckpt_last.pt", "checkpoints/ckpt_best.pt"):
            p = self.stage_dir / "runs" / run_id / rel
            if p.exists():
                freed += p.stat().st_size
                with contextlib.suppress(Exception):
                    p.unlink()
        if freed:
            _print("DISK", f"{run_id}: freed {freed/1e9:.2f} GB locally "
                           f"(both checkpoints confirmed on HuggingFace)")
        return freed

    # -- aggregation ------------------------------------------------------
    def aggregate(self) -> pd.DataFrame:
        rows = []
        for f in (self.stage_dir / "runs").glob("*/metrics/final.csv"):
            with contextlib.suppress(Exception):
                rows.append(pd.read_csv(f))
        if not rows:
            return pd.DataFrame()
        df = pd.concat(rows, ignore_index=True)
        out = self.stage_dir / "tables"
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "all_runs.csv", index=False)
        self.uploader.enqueue(out / "all_runs.csv", "tables/all_runs.csv", force=True)
        return df


def _isolated_train_child(payload_path: str, result_path: str) -> int:
    """CLI entry for one disposable Stage-B training process."""
    payload = read_json(Path(payload_path), None)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid isolated-training payload: {payload_path}")
    cfg = dict(payload["cfg"])
    cfg["_isolated_child"] = True       # excluded from the scientific config hash
    child = Session(
        account=payload["account"],
        worker_id=int(payload["worker_id"]),
        num_workers=int(payload["num_workers"]),
        stage=payload["stage"],
        hf_repo=payload["hf_repo"],
        enable_hf=bool(payload["enable_hf"]),
        session_limit_h=float(payload["session_limit_h"]),
        push_interval_min=float(payload["push_interval_min"]),
        rate_limit=int(payload["rate_limit"]),
    )
    child.data_root = Path(payload["data_root"])
    rid = cfg["run_id"]
    _print("ISOLATE", f"child pid={os.getpid()} owns only {rid}")
    try:
        child.inventory.refresh([rid], verbose=True)
        summary = Trainer(cfg, child).run()
        child.finish()
        atomic_write_json(Path(result_path), summary)
        return 0
    except BaseException as exc:
        # Trainer catches ordinary training exceptions. This covers setup and
        # process-level failures so the parent can make a repository-backed
        # decision instead of silently losing the rest of its plan.
        with contextlib.suppress(Exception):
            child.finish()
        atomic_write_json(Path(result_path), {
            "run_id": rid, "arch": cfg.get("arch"), "fold": cfg.get("fold"),
            "seed": cfg.get("seed"), "status": "failed",
            "epochs_trained": child.inventory.epoch(rid),
            "pause_reason": "isolated_child_exception",
            "error_type": type(exc).__name__, "error_message": str(exc)[:500],
            "cuda_restart_required": fatal_cuda_error(exc),
        })
        traceback.print_exc()
        return 1


# --------------------------------------------------------------------------
# 12. Trivial baselines -- the floor every model must beat
# --------------------------------------------------------------------------

BASELINES = {
    # macro-F1 on the supplied folds, clean images, no deep learning.
    # Each is near-perfect on a DIFFERENT fold: four shortcuts, four folds.
    "frame_occupancy": {"f0": 0.181, "f1": 0.455, "f2": 0.968, "mean": 0.535},
    "colour_probe": {"f0": 0.952, "f1": 0.399, "f2": 0.123, "mean": 0.491},
    "structure_probe": {"f0": 0.354, "f1": 0.119, "f2": 0.976, "mean": 0.483},
    "annotation_sidechannel": {"f0": 0.978, "f1": 0.159, "f2": 0.108, "mean": 0.415},
    "majority_class_acc": {"f0": 0.360, "f1": 0.484, "f2": 0.423, "mean": 0.423},
}
FLOOR = 0.535   # highest trivial baseline. Beat it or nothing was learned.


def baseline_table() -> pd.DataFrame:
    return pd.DataFrame([{"baseline": k, **v} for k, v in BASELINES.items()])


def selftest() -> bool:
    """Offline, no GPU, no network. Run before anything else."""
    ok = True

    def t(name, cond):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + name)
        ok = ok and bool(cond)

    print("=== tyrelib selftest ===")
    t("config_hash stable", config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1}))
    t("config_hash ignores _debug keys",
      config_hash({"a": 1}) == config_hash({"a": 1, "_debug_interrupt_after_epoch": 2}))
    t("checkpoint reconstruction strips retired timm weight tags",
      _timm_model_candidates("convnextv2_small.retired_tag", False) ==
      ["convnextv2_small"])
    t("training preserves the requested timm weight tag",
      _timm_model_candidates("convnextv2_tiny.fcmae", True) ==
      ["convnextv2_tiny.fcmae"])
    fake_r18 = {
        "conv1.weight": np.empty((64, 3, 7, 7)),
        "layer1.0.conv1.weight": np.empty((64, 64, 3, 3)),
        "layer4.0.conv1.weight": np.empty((512, 256, 3, 3)),
    }
    t("checkpoint signature catches ResNet-18 substitution",
      infer_checkpoint_architecture(fake_r18) == "resnet18")
    t("invalid ConvNeXt-V2-S pretrained arm is quarantined",
      ZOO["convnextv2_s"].get("stage_a_valid") is False and
      ZOO["convnextv2_s"].get("pretrained_available") is False)
    t("QWK perfect == 1", abs(quadratic_weighted_kappa([0, 1, 2], [0, 1, 2]) - 1.0) < 1e-9)
    t("QWK penalises distance",
      quadratic_weighted_kappa([0, 1, 2, 0], [0, 1, 1, 0]) > quadratic_weighted_kappa([0, 1, 2, 0], [0, 1, 0, 2]))
    ids = [f"a-{a}-base-f{f}-s{s}" for a in ("resnet50", "maxvit_t", "mobilenetv4")
           for f in range(3) for s in (1, 2, 3)]
    a1 = assign_workers(ids, 4, "cost")
    a2 = assign_workers(list(reversed(ids)), 4, "cost")
    t("sharding deterministic & order-independent", a1 == a2)
    loads = [sum(cost_of(r) for r in ids if a1[r] == w) for w in range(4)]
    t(f"sharding balanced (imbalance {max(loads)/min(loads):.2f}x)", max(loads) / min(loads) < 1.35)
    t("static table used, not measured", cost_of("a-maxvit_t-base-f0-s1") == STATIC_COST_HINTS["maxvit_t"])
    t("retry-after parsed", abs((parse_retry_after("retry after 30 seconds") or 0) - 32.0) < 1e-6)
    t("retry-after minutes parsed", abs((parse_retry_after("in about 5 minutes") or 0) - 305.0) < 1e-6)
    rl = SharedRateLimiter.for_token("tok", 25)
    t("rate limiter is per-token singleton", rl is SharedRateLimiter.for_token("tok", 25))
    m, cm = classification_report_dict([0, 1, 2, 0], [0, 1, 2, 1], None, "val_")
    t("metrics produce qwk + f1", "val_qwk" in m and "val_f1_macro" in m)
    t("confusion matrix shape", cm.shape == (3, 3))
    t("recipe has no early stopping", "patience" not in RECIPE and "min_epochs" not in RECIPE)
    t("zoo non-empty", len(ZOO) >= 15)
    t("RegNet uses conservative contiguous CUDA layout",
      training_memory_format("regnety016") == "contiguous")
    t("other CNNs retain channels_last CUDA layout",
      training_memory_format("resnet50") == "channels_last")
    t("fatal CUDA launch faults require a fresh context",
      fatal_cuda_error(RuntimeError("cuDNN error: CUDNN_STATUS_EXECUTION_FAILED")))
    t("floor matches strongest baseline",
      abs(FLOOR - max(v["mean"] for v in BASELINES.values())) < 1e-9)
    t("cross-fold tyre pairs recorded", len(KNOWN_CROSS_FOLD_PAIRS) >= 1)
    import numpy as _np
    _m = _np.zeros((40, 40), _np.uint8); _m[10:30, 10:30] = 2
    _s = _np.zeros((40, 40), _np.float32); _s[15:25, 15:25] = 1
    _e = evidence_metrics(_s, _m)
    t("evidence_metrics: TER high inside tread", _e["ter"] > 0.99)
    t("evidence_metrics: TER_norm > 1 when focused", _e["ter_norm"] > 1.0)
    t("region_tyre is not raw index 1", region_tyre(_m).sum() == 400)

    # --- the worker/resume invariants (Bug 8, Bug 9) ----------------------
    class _FakeUp:
        enabled = False
        repo_id = "x/y"; repo_type = "dataset"; token = None
    inv = RemoteInventory(_FakeUp(), Path("."))
    inv.files = {"runs/r-done/checkpoints/ckpt_last.pt", "runs/r-done/STATUS.json",
                 "runs/r-mid/checkpoints/ckpt_last.pt", "runs/r-mid/STATUS.json",
                 "runs/r-full/checkpoints/ckpt_last.pt", "runs/r-full/STATUS.json"}
    inv.status = {"r-done": {"status": "completed", "epochs_trained": 60},
                  "r-mid": {"status": "failed", "epoch": 47},
                  "r-full": {"status": "running", "epoch": 60, "of": 60}}
    t("inventory: completed run is completed", inv.state("r-done") == "completed")
    t("inventory: FAILED run is resumable, not lost", inv.state("r-mid") == "resumable")
    t("inventory: resume epoch read from STATUS", inv.epoch("r-mid") == 47)
    t("inventory: full checkpoint is finalised, not called epoch 61 training",
      inv.reason("r-full").startswith("finalise 60-epoch checkpoint"))
    t("inventory: unknown run is absent", inv.state("r-nothing") == "absent")

    # The heart of it: a run's state must not depend on NUM_WORKERS.
    states = {nw: {r: inv.state(r) for r in ("r-done", "r-mid", "r-nothing")}
              for nw in (1, 2, 4)}
    t("run state identical at NUM_WORKERS 1, 2 and 4",
      states[1] == states[2] == states[4])
    # ...while ownership may legitimately differ, it reserves only fresh work.
    t("ownership covers every run at any worker count",
      all(set(assign_workers(ids, nw, "cost")) == set(ids) for nw in (1, 2, 3, 4, 8)))
    t("single worker owns everything",
      set(assign_workers(ids, 1, "cost").values()) == {0})
    t("staging never lands in /kaggle/working",
      "kaggle/working" not in str(staging_root()))

    # --- Bug 12: telemetry must never be able to fail the run --------------
    import tempfile
    mon = HardwareMonitor(Path(tempfile.mkdtemp()))
    stop = threading.Event()

    def _hammer():                       # stands in for the 10 Hz sampler
        i = 0
        while not stop.is_set():
            with mon._lock:
                mon.energy_rows.append({"ts": now(), "gpu_index": 0, "power_w": 1.0,
                                        "energy_joules_cumulative": float(i),
                                        "temp_c": 40, "util_pct": 50})
                mon.samples.append({"ts": now(), "cpu_percent": 10.0})
            i += 1
            time.sleep(0.0005)           # bounded, or the buffers reach millions
    th = threading.Thread(target=_hammer, daemon=True); th.start()
    crashed = False
    try:
        for _ in range(15):              # dump WHILE the sampler is appending
            mon.dump()
    except Exception:
        crashed = True
    stop.set(); th.join(timeout=2)
    t("telemetry dump survives a concurrent sampler", not crashed)
    mon.energy_rows = [{"bad": object()}]          # unserialisable on purpose
    try:
        mon.dump(); swallowed = True
    except Exception:
        swallowed = False
    t("telemetry dump swallows its own errors", swallowed)
    t("telemetry window swallows its own errors",
      HardwareMonitor(Path(tempfile.mkdtemp())).window(float("nan"), None) == {})

    # --- Bug 14: summary.json must be in the uploaded set ------------------
    import inspect as _insp
    _src = _insp.getsource(Trainer.enqueue_light)
    t("summary.json is enqueued for upload", "summary.json" in _src)
    t("confirm_on_hf judges completion by state, not file presence",
      "inventory.state" in _insp.getsource(Session.confirm_on_hf))
    t("stolen runs re-pull the registry before claiming",
      "registry.pull" in _insp.getsource(Session.run_all))
    t("work stealing is opt-in, not the default",
      _insp.signature(Session.run_all).parameters["steal_stale"].default is False and
      _insp.signature(Session.plan).parameters["steal_stale"].default is False)
    _run_all_src = _insp.getsource(Session.run_all)
    t("only a genuinely stolen claim forces an immediate HF commit",
      'rid in getattr(plan, "stolen", ())' in _run_all_src and
      'reason=f"stolen claim {rid}"' in _run_all_src)
    t("a run from my own shard is never double-claimed by the takeover path",
      "if i <= n_mine:" in _run_all_src)
    t("a paused model stops the worker instead of cascading into more runs",
      'if s["status"] == "paused"' in _run_all_src)
    _iso_src = _insp.getsource(Session._run_one_isolated)
    t("per-run isolation uses a fresh Python process",
      "subprocess.Popen" in _iso_src and "--isolated-train" in _iso_src)
    t("parent reconciles HF after an isolated child exits",
      "self.inventory.refresh([rid]" in _iso_src)
    t("a RAM-paused child resumes the same run after process reclamation",
      "for restart in range(1, 9)" in _run_all_src and
      "self._run_one_isolated(by_id[rid])" in _run_all_src and
      'why_pause == "host_ram_guard"' in _run_all_src)
    t("session deadline protects own runs as well as takeover work",
      'near_limit(margin_min=45)' in _run_all_src)
    t("fatal CUDA in an isolated child cannot poison the parent",
      "fatal CUDA fault was contained" in _run_all_src and
      "if isolate_runs:" in _run_all_src)

    # --- Bug 24: an idle worker must not sit parked ------------------------
    class _TInv:
        files = set(); status = {}
        def refresh(self, ids=None, verbose=True): return self
        def state(self, r): return "completed" if r in _t_done else "absent"
        def epoch(self, r): return 0
        def reason(self, r): return "not started"
    class _TReg:
        def latest(self): return {}
        def pull(self, u): return 0
        def can_claim(self, r, a, stale_s=2700): return True, "unclaimed"
    _t_ids = [f"b-a{a}-t{k}-f1-s{s}" for a in range(3) for k in range(4) for s in (1, 2, 3)]
    _t_owner = assign_workers(_t_ids, 4, "cost")
    _t_done = {r for r, w in _t_owner.items() if w == 0}      # worker 0 finished its shard
    _ts = Session.__new__(Session)
    _ts.inventory, _ts.registry, _ts.uploader = _TInv(), _TReg(), None
    _ts.num_workers, _ts.worker_id, _ts.account = 4, 0, "acct1"
    _tp = Session.plan(_ts, _t_ids, title="selftest idle takeover", refresh=False,
                       steal_stale=False, takeover_when_idle=True)
    t("a worker with an empty shard still has work to do",
      _tp.n_mine == 0 and len(_tp.order) == len(_t_ids) - len(_t_done))
    t("its own runs are always ordered before any takeover",
      list(_tp.order[:_tp.n_mine]) == list(_tp.mine))
    _tp_off = Session.plan(_ts, _t_ids, title="", refresh=False, steal_stale=False,
                           takeover_when_idle=True)
    _ts.worker_id = 2
    _tp2 = Session.plan(_ts, _t_ids, title="", refresh=False, steal_stale=False,
                        takeover_when_idle=True)
    t("two idle workers do not start the pool at the same run",
      not _tp_off.stolen or not _tp2.stolen or _tp_off.stolen[0] != _tp2.stolen[0])
    t("takeover can be switched off",
      len(Session.plan(_ts, _t_ids, title="", refresh=False, steal_stale=False,
                       takeover_when_idle=False).stolen) == 0)
    t("takeover claims go through the two-phase protocol",
      "claim_or_yield" in _run_all_src and "near_limit(margin_min=90)" in _run_all_src)
    _coy = _insp.getsource(Session.claim_or_yield)
    t("two-phase claim flushes, settles, then re-reads",
      "uploader.flush" in _coy and "time.sleep" in _coy and _coy.count("registry.pull") >= 2)
    t("two-phase claim breaks ties deterministically, not by luck",
      'min(str(e["account"]) for e in rivals)' in _coy)

    # --- Bug 22/23: the RAM guard must not end a session over a spike ------
    _trainer_run = _insp.getsource(Trainer.run)
    t("training has a plain-text first-batch heartbeat",
      'batch 1/{len(tr_dl)} completed' in _trainer_run and
      'training is active' in _trainer_run)
    t("weight-norm telemetry is detached from autograd",
      "p.detach().norm().item()" in _trainer_run)
    t("RAM guard reads a live post-release value, not the epoch peak",
      "host_ram_headroom()" in _trainer_run and "ram_now >= HOST_RAM_PAUSE_PERCENT" in _trainer_run)
    t("RAM guard no longer pauses on ram_percent_peak alone",
      "ep + 1 < n_ep and ram_peak >= HOST_RAM_PAUSE_PERCENT" not in _trainer_run)
    t("a recovered RAM pause continues instead of ending the cell",
      'why == "host_ram_guard"' in _run_all_src and "continue" in _run_all_src)
    t("resume threshold sits below the pause threshold",
      HOST_RAM_RESUME_PERCENT < HOST_RAM_PAUSE_PERCENT)
    t("host_ram_percent returns a sane number",
      0.0 <= host_ram_percent() <= 100.0)

    # --- Bug 25: measure the budget the OOM killer enforces -----------------
    _used, _limit, _src = container_memory()
    t(f"container_memory reports a budget [{_src}]",
      _limit > 0 and 0 <= _used <= _limit * 1.05)
    t("container_memory prefers the cgroup when one exists",
      "cgroup" in _insp.getsource(container_memory) and
      "memory.current" in _insp.getsource(container_memory))
    t("host_ram_percent is measured against that budget, not /proc/meminfo",
      "container_memory()" in _insp.getsource(host_ram_percent))
    _mr = memory_report()
    t("memory_report splits this process from its children",
      {"proc_rss_gb", "children_rss_gb", "limit_gb", "source"} <= set(_mr))
    t("a RAM pause says where the memory actually is",
      "child proc" in _trainer_run and "mem['proc_rss_gb']" in _trainer_run)
    t("post-release memory fields are persisted to epoch history",
      _trainer_run.count("append_epoch_row(self.hist_path, row)") == 2 and
      'row["mem_source"]' in _trainer_run)

    # --- Bug 26: loader workers that buy nothing ---------------------------
    t("GPU-bound configurations get no loader workers",
      dataloading_is_free({"input_resolution": 384})
      and dataloading_is_free({"input_resolution": 512}))
    t("small fast configurations keep their workers",
      not dataloading_is_free({"input_resolution": 224}))
    _bl = _insp.getsource(build_loaders)
    t("pin_memory follows the worker count instead of being forced on",
      "pin = bool(torch.cuda.is_available() and nw > 0)" in _bl)
    t("the worker decision is a named, measured rule",
      "dataloading_is_free(cfg)" in _bl)

    _dump_src = _insp.getsource(HardwareMonitor.dump)
    t("telemetry dump drains its buffers instead of accumulating",
      "self.energy_rows = self.energy_rows, []" in _dump_src)
    t("telemetry dump appends rather than rewriting the whole run",
      'gzip.open(path, "at"' in _dump_src)
    t("step traces are capped per epoch and appended, never rewritten",
      "len(step_traces) < 2000:" in _trainer_run
      and 'step_traces.jsonl", "w"' not in _trainer_run)

    import tempfile as _tf
    _mon = HardwareMonitor(Path(_tf.mkdtemp()))
    for _ in range(3):
        with _mon._lock:
            for i in range(50):
                _mon.energy_rows.append({"ts": now(), "gpu_index": 0, "power_w": 1.0,
                                         "energy_joules_cumulative": float(i),
                                         "temp_c": 40, "util_pct": 50})
        _mon.dump()
    t("telemetry buffer is empty after a dump", len(_mon.energy_rows) == 0)
    _back = pd.read_csv(Path(_mon.out_dir) / "energy_samples.csv.gz")
    t(f"appended gzip members read back as one table ({len(_back)} rows)", len(_back) == 150)
    _trainer_src = _insp.getsource(Trainer.run)
    t("each epoch serialises one full checkpoint, not best plus last",
      _trainer_src.count("self.save_ckpt(") == 1 and
      "atomic_clone_file(self.ckpt_last, self.ckpt_best)" in _trainer_src)
    _hist = Path(tempfile.mkdtemp()) / "epochs.csv"
    _buf = io.StringIO(); _cw = csv.writer(_buf, lineterminator="\n")
    _cw.writerow(["epoch", "runtime_memory_safety_revision",
                  "runtime_cuda_memory_format", "val_qwk"])
    _cw.writerow([1, "2026-08-31-r1", "channels_last", 0.5])
    _cw.writerow([2, "2026-08-31-r2", "2026-08-31-r1", "channels_last", 0.6])
    atomic_write_text(_hist, _buf.getvalue())
    _hh = read_epoch_history(_hist, repair=True)
    t("mixed epoch schemas are repaired without dropping or shifting rows",
      len(_hh) == 2 and
      "runtime_hf_commit_policy_revision" in _hh.columns and
      pd.isna(_hh.loc[0, "runtime_hf_commit_policy_revision"]) and
      _hh.loc[1, "runtime_cuda_memory_format"] == "channels_last" and
      abs(float(_hh.loc[1, "val_qwk"]) - 0.6) < 1e-9)
    append_epoch_row(_hist, {"epoch": 3, "runtime_memory_safety_revision": "r2",
                             "runtime_epoch_history_schema_revision": "r1",
                             "runtime_cuda_memory_format": "channels_last",
                             "val_qwk": 0.7})
    _hh2 = read_epoch_history(_hist)
    t("epoch writer expands columns atomically and remains readable",
      len(_hh2) == 3 and
      "runtime_epoch_history_schema_revision" in _hh2.columns and
      list(_hh2.epoch.astype(int)) == [1, 2, 3])
    t("fresh absent work is reserved for its static owner",
      "if event is None" in _insp.getsource(Session.plan))
    t("takeover planning refreshes registry claims first",
      "registry.pull" in _insp.getsource(Session.plan))
    class _PlanInventory:
        def refresh(self, *args, **kwargs): return self
        def state(self, run_id): return "absent"
        def epoch(self, run_id): return 0
    class _PlanRegistry:
        def pull(self, uploader): return 0
        def latest(self): return {}
        def can_claim(self, *args, **kwargs): return True, "unclaimed"
    _ps = Session.__new__(Session)
    _ps.inventory, _ps.registry, _ps.uploader = _PlanInventory(), _PlanRegistry(), None
    _ps.num_workers, _ps.worker_id, _ps.account = 4, 0, "acct1"
    _pp = Session.plan(_ps, ids, title="selftest fresh ownership", refresh=False)
    _owned = {r for r, w in assign_workers(ids, 4, "cost").items() if w == 0}
    # Bug 13's guarantee, restated for the takeover era: at a simultaneous cold
    # start every worker must do its OWN fresh runs first. The pool exists, but
    # nothing in it is reachable until `mine` is exhausted, so four accounts
    # starting together still cannot collide.
    t("an all-absent four-worker plan does this worker's own fresh runs first",
      set(_pp.mine) == _owned and set(_pp.order[:_pp.n_mine]) == _owned)
    _pp_noto = Session.plan(_ps, ids, title="", refresh=False, takeover_when_idle=False)
    t("with takeover off, an all-absent plan is exactly this worker's shard",
      set(_pp_noto.order) == _owned and not _pp_noto.stolen)
    import tempfile as _tempfile
    _reg = Registry(Path(_tempfile.mkdtemp()), None, "acct1", 0, "selftest")
    _reg.emit("recent-failure", "failed", account="acct2")
    t("recent failed work cannot be stolen immediately",
      not _reg.can_claim("recent-failure", "acct1", stale_s=2700)[0])
    t("the same account can immediately retry its failed work",
      _reg.can_claim("recent-failure", "acct2", stale_s=2700)[0])

    # --- Bug 15: the resolution contract ----------------------------------
    # No timm here, so this checks the arithmetic and the plumbing rather than
    # the models. `assert_zoo_ok` in the notebooks does the real thing.
    t("build_model is told the resolution",
      "img_size" in _insp.signature(build_model).parameters)
    t("build_model verifies with a forward pass by default",
      _insp.signature(build_model).parameters["verify"].default is True)
    t("Trainer passes input_resolution to build_model",
      "img_size=cfg[\"input_resolution\"]" in _insp.getsource(Trainer.run))
    patch = {"dinov2_s": 14, "dinov2_b": 14, "clip_b16": 16, "vit_s": 16,
             "deit3_s": 16, "maxvit_t": 32, "swin_t": 32, "swin_s": 32}
    bad_res = {a: ZOO[a]["res"] for a, p in patch.items()
               if a in ZOO and ZOO[a]["res"] % p}
    t(f"every patch-based arch has a divisible resolution {bad_res or ''}", not bad_res)

    # --- Bug 16: mask propagation, pinned ---------------------------------
    # The original replay read `box` and `angle`; the dataset records
    # `crop_box` and `degrees`. Both lookups quietly found nothing, so the crop
    # and the rotation were skipped on all 4,180 derivatives and the files were
    # written anyway. These assert that each operation actually MOVES pixels.
    try:
        from PIL import Image as _I
        src = _I.new("L", (100, 200), 0)
        src.paste(255, (0, 0, 50, 100))                 # bright top-left quadrant
        a = np.asarray(apply_trace(src, [{"name": "horizontal_flip"}], (100, 200)))
        t("apply_trace: flip actually flips", a[0:50, 0:25].mean() < a[0:50, 75:100].mean())

        crop = [{"name": "random_resized_crop_letterbox",
                 "crop_box": [0, 0, 50, 100], "output_size": 64}]
        c = np.asarray(apply_trace(src, crop, (64, 64)))
        t("apply_trace: crop_box is read (not 'box')", c.shape == (64, 64) and c.max() > 0)
        t("apply_trace: letterbox pads rather than stretching",
          bool((c[:, 0] == 0).all() and (c[:, -1] == 0).all()))

        rot = np.asarray(apply_trace(src, [{"name": "rotation", "degrees": 90.0}], (100, 200)))
        t("apply_trace: degrees is read (not 'angle')",
          not np.array_equal(rot, np.asarray(src)))

        t("apply_trace: photometric ops are no-ops",
          np.array_equal(np.asarray(apply_trace(src, [{"name": "gamma", "value": 2.0}], (100, 200))),
                         np.asarray(src)))
        raised = False
        try:
            apply_trace(src, [{"name": "some_new_geometric_op"}], (100, 200))
        except ValueError:
            raised = True
        t("apply_trace: unknown operation RAISES, never skipped", raised)

        # alignment_score must prefer the true mask over a shifted one
        g_ = np.full((80, 80), 200.0, np.float32); g_[20:60, 20:60] = 40.0
        m_ = np.zeros((80, 80), np.uint8); m_[20:60, 20:60] = 1
        t("alignment_score: correct beats shifted",
          alignment_score(g_, m_) > alignment_score(g_, np.roll(m_, 20, axis=1)))
    except ImportError:
        t("apply_trace checks (PIL unavailable -- SKIPPED)", True)

    t("ensure_annotations does not trust the version file",
      "annotation_version" not in _insp.getsource(ensure_annotations).split("_print")[0]
      or "not trusted" in _insp.getsource(ensure_annotations))

    # --- Post-Stage-A ablation/XAI contracts -----------------------------
    try:
        validate_config(dict(RECIPE))
        cfg_ok = True
    except Exception:
        cfg_ok = False
    t("base recipe passes the OFAT config gate", cfg_ok)
    try:
        validate_config(dict(RECIPE, preprocessing="misspelled")); rejected = False
    except ValueError:
        rejected = True
    t("unsupported OFAT values fail instead of becoming no-ops", rejected)
    t("dual-GPU checkpoints save the unwrapped module",
      "core_model.state_dict" in _insp.getsource(Trainer.save_ckpt))
    t("frozen arm exposes only the classifier",
      "get_classifier" in _insp.getsource(Trainer.run)
      and "requires_grad = False" in _insp.getsource(Trainer.run))

    try:
        import torch as _torch
        z = _torch.tensor([2.0, -1.0])
        cp = [float(ClassProbabilityTarget(k, "coral")(z)) for k in range(3)]
        t("CAM target understands all three CORAL classes",
          len(cp) == 3 and cp[0] > 0 and cp[2] > 0)
    except Exception:
        t("CAM target understands all three CORAL classes", False)

    try:
        from PIL import Image as _Image
        td = Path(tempfile.mkdtemp()); (td / "images").mkdir()
        img = _Image.new("RGB", (80, 100), (120, 130, 140))
        img.save(td / "images" / "x.png")
        clean = td / "masks"; clean.mkdir(); mask = np.zeros((100, 80), np.uint8)
        mask[20:80, 25:55] = MASK_TREAD; _Image.fromarray(mask).save(clean / "id.png")
        frame = pd.DataFrame([{"relative_path": "images/x.png", "image_id": "id",
                               "image_kind": "clean_original", "proxy_label": CLASSES[0]}])
        ds = TyreDataset(frame, td, lambda im: np.asarray(im), roi_mode="tyre_crop",
                         annotation_roots={"clean_masks": clean, "propagated_masks": clean})
        cropped, _, _ = ds[0]
        t("tyre_crop changes the actual pixels given to the model",
          cropped.shape[0] < 100 and cropped.shape[1] < 80)
        t("tyre_crop bbox preserves the legacy crop coordinates",
          tuple(cropped.shape[:2]) == (66, 36))
        t("tyre_crop bbox avoids full per-pixel coordinate arrays",
          "getbbox" in _insp.getsource(TyreDataset.__getitem__)
          and "mask_path" in _insp.getsource(TyreDataset.__getitem__))
        roi_cfg = dict(RECIPE, roi_mode="tyre_crop", sampler_name="uniform",
                       batch_size=1, clean_mask_root=str(clean),
                       propagated_mask_root=str(clean))
        tr_test, va_test = build_loaders(td, frame, frame, roi_cfg)
        t("tyre_crop loader disables workers and pinned-memory caching",
          tr_test.num_workers == 0 and not tr_test.pin_memory
          and va_test.num_workers == 0 and not va_test.pin_memory)
        xb_test, yb_test, _ = next(iter(tr_test))
        t("tyre_crop memory-safe loader yields a real training batch",
          tuple(xb_test.shape) == (1, 3, RECIPE["input_resolution"],
                                   RECIPE["input_resolution"])
          and tuple(yb_test.shape) == (1,))
        _shutdown_loader(tr_test); _shutdown_loader(va_test)
        clahe = build_transforms(32, False, "clahe")(_Image.new("RGB", (40, 50), (80, 90, 100)))
        t("CLAHE arm is implemented, not a raw-image alias", tuple(clahe.shape) == (3, 32, 32))
    except Exception as e:
        t(f"ROI/CLAHE smoke test ({type(e).__name__}: {e})", False)

    failed_gate, failed_choice = cam_method_gate([
        {"method": "gradcam", "sanity_delta": 0.012974,
         "insertion_auc": 0.899683, "deletion_auc": 0.397754},
        {"method": "hirescam", "sanity_delta": 0.013138,
         "insertion_auc": 0.899689, "deletion_auc": 0.397655},
    ], revision="2026-08-30-r3")
    t("failed CAM gate excludes without raising",
      failed_choice is None and not failed_gate.selected.any()
      and failed_gate.gate_status.eq("failed").all())
    passed_gate, passed_choice = cam_method_gate([
        {"method": "gradcam", "sanity_delta": 0.08,
         "insertion_auc": 0.70, "deletion_auc": 0.40},
        {"method": "hirescam", "sanity_delta": 0.09,
         "insertion_auc": 0.85, "deletion_auc": 0.35},
    ])
    t("valid CAM gate still selects best faithfulness",
      passed_choice == "hirescam" and int(passed_gate.selected.sum()) == 1)
    maps_a = np.zeros((2, 8, 8), np.float32); maps_a[:, 2:4, 2:4] = 1
    maps_b = maps_a.copy(); maps_b[1] = 0; maps_b[1, 5:7, 5:7] = 1
    t("randomisation sanity averages both maps with scale-free decorrelation",
      saliency_change_score(maps_a, maps_a) < 1e-7
      and saliency_change_score(maps_a, maps_b) > 0.05)

    print("=== selftest", "PASSED" if ok else "FAILED", "===")
    return ok


# --------------------------------------------------------------------------
# 13. Annotation masks -- the XAI measuring instrument
# --------------------------------------------------------------------------
#
# ⚠ Bug 16 -- why this module rebuilds the masks instead of trusting them.
#
# Kaggle attaches ONE VERSION of a dataset to a notebook. Re-uploading does not
# move existing notebooks onto the new version; they keep reading the old one,
# silently, with nothing on screen to say so. So "which propagated masks am I
# actually looking at" is a question the notebook cannot answer and the user
# cannot easily control.
#
# It is also a question we never needed to ask. Everything required to BUILD
# the propagated masks is present in every version of the dataset:
#
#   annotations/clean/masks/        418 hand-drawn masks -- never were broken
#   FINAL/manifests/dataset_manifest.csv
#                                   augmentation_trace_json: the exact ops,
#                                   in order, for all 4,180 derivatives
#
# Replaying that takes about a minute. So the notebooks stop depending on the
# 4,180 propagated PNGs entirely: measure what is there, and if it does not
# track its images, rebuild it into the session's scratch directory and use
# that. Self-healing, version-proof, and the propagation logic lives in one
# place instead of in a script the notebooks cannot reach.

# Single indexed layer, so a later class ERASES the earlier one underneath.
# `m == 1` is NOT "the tyre"; it is "tyre minus whatever is painted on top",
# which on a head-on tyre photo is nearly empty. Always use these accessors.
MASK_BG, MASK_TYRE, MASK_TREAD, MASK_MARKING, MASK_DAMAGE = 0, 1, 2, 3, 4

# Every operation the augmentation policy can emit must be in exactly one set.
# An unrecognised name RAISES -- silently skipping one is precisely how the
# original propagation wrote 4,180 well-formed, correctly sized, misplaced
# masks without a single warning.
GEOMETRIC_OPS = {"random_resized_crop_letterbox", "horizontal_flip",
                 "vertical_flip", "rotation"}
PHOTOMETRIC_OPS = {"brightness_contrast", "gamma", "saturation", "clahe",
                   "gaussian_noise", "gaussian_blur", "box_blur", "unsharp_mask",
                   "jpeg_recompression", "coarse_dropout"}


def _letterbox_mask(im, out: int):
    """Aspect-preserving resize onto a square canvas, centred, padded with 0.

    `round`, not `int`: checked against the real images -- on 400 unrotated
    derivatives the bar widths implied by `round` matched the measured
    constant-column runs 215 times against 103 for `int`.
    """
    from PIL import Image
    w, h = im.size
    s = out / max(w, h)
    w2, h2 = max(1, round(w * s)), max(1, round(h * s))
    im = im.resize((w2, h2), Image.NEAREST)
    canvas = Image.new("L", (out, out), 0)
    canvas.paste(im, ((out - w2) // 2, (out - h2) // 2))
    return canvas


def apply_trace(mask, ops: list, target_size):
    """Replay the geometric operations of one derivative onto its source mask.

    Nearest-neighbour throughout: bilinear invents class values at boundaries.
    Exact key names, no substring matching -- the trace records `crop_box` and
    `degrees`, and guessing `box` and `angle` is what produced masks that were
    wrong on every derivative.
    """
    from PIL import Image
    m = mask
    for op in ops:
        name = op.get("name") or op.get("op") or ""
        if name in PHOTOMETRIC_OPS:
            continue                       # does not move pixels
        if name not in GEOMETRIC_OPS:
            raise ValueError(
                f"operation {name!r} is in neither GEOMETRIC_OPS nor "
                f"PHOTOMETRIC_OPS. Classify it before trusting any mask.")
        if name == "random_resized_crop_letterbox":
            m = m.crop(tuple(int(v) for v in op["crop_box"]))
            m = _letterbox_mask(m, int(op["output_size"]))
        elif name == "horizontal_flip":
            m = m.transpose(Image.FLIP_LEFT_RIGHT)
        elif name == "vertical_flip":
            m = m.transpose(Image.FLIP_TOP_BOTTOM)
        elif name == "rotation":
            # PIL rotates counter-clockwise for positive angles. Established by
            # measurement: on the largest-|angle| decile, rotate(+degrees)
            # scored 33.96 on the alignment metric against 28.36 for negative.
            ang = float(op["degrees"])
            if ang:
                m = m.rotate(ang, resample=Image.NEAREST, expand=False, fillcolor=0)
    if m.size != tuple(target_size):
        m = m.resize(tuple(target_size), Image.NEAREST)
    return m


def alignment_score(grey: np.ndarray, mask: np.ndarray) -> float:
    """Mean luminance outside the mask minus mean luminance inside it.

    A tyre is much darker than road, wall and sky, so a correctly placed mask
    puts the dark pixels inside and the bright ones outside. Misplace it and
    the populations mix and the score collapses. Needs no ground truth beyond
    the image itself, which is why it can catch a replay bug.
    """
    t = mask > 0
    f = t.mean()
    if f < 0.02 or f > 0.995:
        return float("nan")
    return float(grey[~t].mean() - grey[t].mean())


def measure_masks(data_root, mask_dir, manifest=None, n: int = 120,
                  seed: int = 0) -> dict:
    """Score real masks against three deliberately wrong versions of themselves.

    Same image, same photometry, only the placement differs:
      shift    moved 6% of the frame sideways
      mirror   flipped left-right
      swap     a different image's mask

    Correct masks beat all three by a wide margin. The broken propagation
    scored 15.7 against a swap control of 9.8 -- barely better than a mask
    belonging to a different photograph, which is what a broken replay is.
    """
    from PIL import Image
    root = Path(data_root)
    mask_dir = Path(mask_dir)
    df = manifest if manifest is not None else read_manifest(root / "manifests" / "dataset_manifest.csv")
    aug = df[df.image_kind == "synthetic_derivative"]
    rows = list(aug.itertuples())
    random.Random(seed).shuffle(rows)

    cor, shf, mir, swp = [], [], [], []
    prev = None
    for r in rows:
        p = mask_dir / f"{r.image_id}.png"
        ip = root / r.relative_path
        if not (p.exists() and ip.exists()):
            continue
        g = np.asarray(Image.open(ip).convert("L"), dtype=np.float32)
        k = np.asarray(Image.open(p))
        if g.shape != k.shape:
            continue
        d = int(0.06 * k.shape[1])
        cor.append(alignment_score(g, k))
        shf.append(alignment_score(g, np.roll(k, d, axis=1)))
        mir.append(alignment_score(g, k[:, ::-1]))
        if prev is not None and prev.shape == k.shape:
            swp.append(alignment_score(g, prev))
        prev = k
        if len(cor) >= n:
            break

    f = lambda x: float(np.nanmean(x)) if len(x) else float("nan")
    out = {"n": len(cor), "correct": f(cor), "shifted": f(shf),
           "mirrored": f(mir), "swapped": f(swp)}
    ctrls = [out["shifted"], out["mirrored"], out["swapped"]]
    ctrls = [c for c in ctrls if not np.isnan(c)]
    out["worst_control"] = max(ctrls) if ctrls else float("nan")
    out["margin"] = out["correct"] - out["worst_control"]
    out["ok"] = bool(out["n"] >= 20 and out["margin"] > 5.0)
    return out


def propagate_masks(ann_root, data_root, out_dir, verbose: bool = True) -> int:
    """Rebuild all propagated masks from the clean ones and the recorded traces.

    ~60 s for 4,180. The source of truth is the 418 hand-drawn masks plus
    `augmentation_trace_json`, both of which are in every version of the
    dataset, so this never depends on which copy of the derivatives is present.
    """
    from PIL import Image
    ann, root, out = Path(ann_root), Path(data_root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = read_manifest(root / "manifests" / "dataset_manifest.csv")
    aug = df[df.image_kind == "synthetic_derivative"]
    cache: dict = {}
    n_ok = n_miss = 0
    t0 = now()
    for i, r in enumerate(aug.itertuples()):
        sm = ann / "clean" / "masks" / f"{r.source_image_id}.png"
        if not sm.exists():
            n_miss += 1
            continue
        if r.source_image_id not in cache:
            cache[r.source_image_id] = Image.open(sm).convert("L")
        trace = json.loads(r.augmentation_trace_json)
        ops = trace.get("operations", trace.get("ops", [])) if isinstance(trace, dict) else trace
        if not ops:
            raise ValueError(f"{r.image_id}: empty augmentation trace -- cannot replay")
        apply_trace(cache[r.source_image_id], ops,
                    (int(r.width), int(r.height))).save(out / f"{r.image_id}.png")
        n_ok += 1
        if verbose and (i + 1) % 1000 == 0:
            print(f"    {i+1}/{len(aug)}")
    if verbose:
        _print("ANN", f"rebuilt {n_ok} propagated mask(s) in {human_time(now()-t0)}"
                      + (f"  ({n_miss} missing source)" if n_miss else ""))
    return n_ok


def ensure_annotations(data_root, ann_root=None, work_dir=None,
                       verbose: bool = True) -> dict:
    """Return annotation directories that are known-good, rebuilding if needed.

    THE POINT: a notebook should not be able to silently consume misplaced
    masks because Kaggle handed it an older dataset version. So:

      1. Measure the propagated masks that are present.
      2. If they track their images, use them.
      3. If they do not, rebuild them from the clean masks and the traces into
         the session scratch directory, measure again, and use those.
      4. Only fail if the REBUILT masks are also bad -- which would mean the
         hand-drawn masks or the traces are wrong, and that is a real problem
         rather than a stale upload.

    Returns {"clean_masks", "propagated_masks", "rebuilt", "before", "after"}.
    """
    root = Path(data_root)
    ann = Path(ann_root) if ann_root else find_annotations_root(root)
    if ann is None:
        raise FileNotFoundError("annotations/ not found beside FINAL/")
    clean = ann / "clean" / "masks"
    prop = ann / "propagated" / "masks"

    ver = read_json(ann / "ANNOTATION_VERSION.json", {})
    if verbose:
        _print("ANN", f"root {ann}  (file says version "
                      f"{ver.get('annotation_version','unknown')!r} -- not trusted, measuring)")

    before = measure_masks(root, prop) if prop.is_dir() else {"ok": False, "n": 0, "margin": float("nan")}
    if verbose:
        _print("ANN", f"as supplied: correct {before.get('correct', float('nan')):.1f}  "
                      f"worst control {before.get('worst_control', float('nan')):.1f}  "
                      f"margin {before.get('margin', float('nan')):+.1f}  "
                      f"-> {'OK' if before['ok'] else 'MISALIGNED'}")
    if before["ok"]:
        return {"clean_masks": clean, "propagated_masks": prop,
                "rebuilt": False, "before": before, "after": before}

    work = Path(work_dir) if work_dir else (staging_root() / "annotations")
    rebuilt_dir = work / "propagated" / "masks"
    if verbose:
        _print("ANN", "rebuilding from the 418 hand-drawn masks + the recorded "
                      "transform traces (both are in every version of the dataset)")
    propagate_masks(ann, root, rebuilt_dir, verbose=verbose)
    after = measure_masks(root, rebuilt_dir)
    if verbose:
        _print("ANN", f"rebuilt:     correct {after['correct']:.1f}  "
                      f"worst control {after['worst_control']:.1f}  "
                      f"margin {after['margin']:+.1f}  "
                      f"-> {'OK' if after['ok'] else 'STILL BAD'}")
    if not after["ok"]:
        raise RuntimeError(
            "Rebuilt masks still do not track their images (margin "
            f"{after['margin']:+.1f}, want > +5).\n"
            "That is not a stale upload -- either the 418 hand-drawn masks in "
            "annotations/clean/masks/ are wrong, or augmentation_trace_json "
            "does not describe what was actually done to the images.")
    _print("ANN", f"using rebuilt masks at {rebuilt_dir}")
    return {"clean_masks": clean, "propagated_masks": rebuilt_dir,
            "rebuilt": True, "before": before, "after": after}


def region_tyre(m):      return m > MASK_BG
def region_tread(m):     return (m == MASK_TREAD) | (m == MASK_MARKING)
def region_marking(m):   return m == MASK_MARKING
def region_damage(m):    return m == MASK_DAMAGE
def region_background(m): return m == MASK_BG


REGIONS = {"tyre": region_tyre, "tread": region_tread, "marking": region_marking,
           "damage": region_damage, "background": region_background}


def mask_path(ann_root, image_id: str, kind: str = "clean_original") -> Path:
    """Resolve one mask without decoding it."""
    if isinstance(ann_root, dict):
        return Path(ann_root["clean_masks" if kind == "clean_original"
                             else "propagated_masks"]) / f"{image_id}.png"
    sub = "clean" if kind == "clean_original" else "propagated"
    return Path(ann_root) / sub / "masks" / f"{image_id}.png"


def load_mask(ann_root, image_id: str, kind: str = "clean_original"):
    """Load one mask into owned memory and close the image immediately.

    `ann_root` may be the annotations directory, OR the dict returned by
    `ensure_annotations()` -- pass the dict and you automatically read the
    rebuilt masks when the supplied ones were misaligned, which is the only
    way a notebook can be sure which masks it is measuring.
    """
    from PIL import Image
    p = mask_path(ann_root, image_id, kind)
    if not p.exists():
        return None
    with Image.open(p) as im:
        return np.array(im, copy=True)


def evidence_metrics(sal: np.ndarray, mask: np.ndarray) -> dict:
    """TER / BAR / SAR / DmgAR from one saliency map and one annotation mask.

    On THIS dataset tread and tyre are nearly the same region (median area ratio
    0.990; 114/418 images have no visible shoulder), so TER measures attention
    on the TYRE versus the BACKGROUND -- not tread versus shoulder. Word claims
    accordingly. See 14_XAI_PROTOCOL.
    """
    import cv2
    if sal.shape != mask.shape:
        sal = cv2.resize(sal.astype(np.float32), (mask.shape[1], mask.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
    sal = np.clip(sal, 0, None)
    tot = sal.sum()
    if tot <= 0:
        return {k: NA for k in ("ter", "ter_norm", "bar", "sar", "dmgar", "edi",
                                "tread_area_frac", "peak_in_tread")}
    p = sal / tot
    out = {}
    for key, fn in (("ter", region_tread), ("bar", region_background),
                    ("sar", region_marking), ("dmgar", region_damage)):
        out[key] = float(p[fn(mask)].sum())
    area = float(region_tread(mask).mean())
    out["tread_area_frac"] = area
    # Area-normalised is THE number. Raw TER is inflated whenever the tyre fills
    # the frame -- and frame occupancy is itself a class cue here (low 72%,
    # mid 62%, high 61%), so raw TER partly measures the shortcut we are hunting.
    out["ter_norm"] = float(out["ter"] / area) if area > 1e-9 else NA
    q = p[p > 0]
    out["edi"] = float(-(q * np.log(q)).sum() / np.log(p.size))
    yx = np.unravel_index(int(np.argmax(p)), p.shape)
    out["peak_in_tread"] = bool(region_tread(mask)[yx])
    return out


# --------------------------------------------------------------------------
# 14. Attribution -- architecture-appropriate, faithfulness-selected
# --------------------------------------------------------------------------

CAM_TARGETS = {
    "resnet18": "layer4", "resnet50": "layer4", "resnext50": "layer4",
    "densenet121": "features", "vgg16bn": "features",
    "convnextv2_t": "stages", "convnextv2_s": "stages", "effnetv2s": "conv_head",
    "regnety016": "s4", "mobilenetv4": "blocks", "coatnet0": "stages",
    "maxvit_t": "stages", "swin_t": "layers", "swin_s": "layers",
    "vit_s": "blocks", "deit3_s": "blocks", "dinov2_s": "blocks",
    "dinov2_b": "blocks", "clip_b16": "blocks",
}
IS_TRANSFORMER = {"vit_s", "deit3_s", "dinov2_s", "dinov2_b", "clip_b16"}
IS_WINDOWED = {"swin_t", "swin_s"}


class ClassProbabilityTarget:
    """A CAM target that understands both CE and two-threshold CORAL heads."""
    def __init__(self, category: int, head_type: str = "coral"):
        self.category = int(category)
        self.head_type = head_type

    def __call__(self, output):
        import torch
        if self.head_type == "coral":
            cum = torch.sigmoid(output)
            if self.category == 0:
                return 1 - cum[0]
            if self.category == 1:
                return cum[0] - cum[1]
            return cum[1]
        return torch.softmax(output, dim=-1)[self.category]


def _resolve_layer(model, path: str):
    mod = model
    for part in path.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


def cam_target_layers(model, arch: str):
    """The last spatial feature stage. Verified non-degenerate in NB00."""
    name = CAM_TARGETS.get(arch)
    if name is None:
        return None
    try:
        mod = _resolve_layer(model, name)
        return [mod[-1]] if hasattr(mod, "__getitem__") and len(mod) else [mod]
    except Exception:
        return None


def reshape_transform_for(arch: str):
    """ViTs emit tokens, not a feature map. Grad-CAM needs it reshaped -- and
    the exact transform must be REPORTED, because 'Grad-CAM on a ViT' names
    several different algorithms in the literature (14_XAI_PROTOCOL §1)."""
    if arch in IS_WINDOWED:
        def _windowed(tensor, height=None, width=None):
            # timm Swin blocks expose channels-last [B,H,W,C]. CAM expects
            # [B,C,H,W]. Leave already-channels-first tensors untouched.
            if tensor.ndim == 4 and tensor.shape[-1] > tensor.shape[1]:
                return tensor.permute(0, 3, 1, 2)
            return tensor
        return _windowed
    if arch not in IS_TRANSFORMER:
        return None

    def _t(tensor, height=None, width=None):
        import torch
        t = tensor[:, 1:, :] if tensor.shape[1] % 2 == 1 else tensor
        n = t.shape[1]
        h = w = int(round(n ** 0.5))
        if h * w != n:
            return tensor
        r = t.reshape(t.size(0), h, w, t.size(2))
        return r.permute(0, 3, 1, 2)
    return _t


def make_cam(model, arch: str, method: str = "gradcam"):
    """pytorch-grad-cam wrapper. Returns (cam_object, label) or (None, reason)."""
    try:
        from pytorch_grad_cam import (GradCAM, HiResCAM, LayerCAM, XGradCAM,
                                      EigenCAM, ScoreCAM)
    except ImportError:
        return None, "pytorch-grad-cam not installed"
    cls = {"gradcam": GradCAM, "hirescam": HiResCAM, "layercam": LayerCAM,
           "xgradcam": XGradCAM, "eigencam": EigenCAM, "scorecam": ScoreCAM}.get(method)
    if cls is None:
        return None, f"unknown method {method}"
    layers = cam_target_layers(model, arch)
    if not layers:
        return None, f"no CAM target layer registered for {arch}"
    rt = reshape_transform_for(arch)
    try:
        cam = cls(model=model, target_layers=layers, reshape_transform=rt)
        reshape_tag = (", reshape=channels_last" if arch in IS_WINDOWED else
                       ", reshape=tokens_to_square" if rt else "")
        tag = f"{method}({CAM_TARGETS[arch]}" + reshape_tag + ")"
        return cam, tag
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def cam_method_gate(rows, sanity_threshold: float = 0.05,
                    revision: str | None = None):
    """Apply the locked XAI method gate without turning a negative result into
    a notebook failure.

    Returns ``(table, chosen_method_or_None)``. ``None`` means the architecture
    has no attribution method trustworthy enough for TER ranking; callers must
    record and exclude it, never relax the threshold after seeing the result.
    """
    d = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    required = {"method", "sanity_delta", "insertion_auc", "deletion_auc"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"CAM gate rows missing columns: {sorted(missing)}")
    d["faithfulness"] = d.insertion_auc - d.deletion_auc
    d["passes_sanity"] = d.sanity_delta > float(sanity_threshold)
    d["passes_faithfulness"] = d.faithfulness.notna()
    if revision is not None:
        d["xai_revision"] = revision
    d["selected"] = False
    d["gate_status"] = np.where(
        d.passes_sanity & d.passes_faithfulness, "passed", "failed")
    valid = d[d.passes_sanity & d.passes_faithfulness]
    if not len(valid):
        return d, None
    chosen = str(valid.sort_values("faithfulness", ascending=False).iloc[0].method)
    d["selected"] = d.method.eq(chosen)
    return d, chosen


def saliency_change_score(before, after) -> float:
    """Mean decorrelation after weight randomisation, averaged over images.

    A sparse CAM can move completely while retaining a tiny pixelwise MAE
    because most pixels are zero. Correlation is scale-independent: identical
    maps score 0, decorrelated maps score about 1. Both members of a batch are
    measured; the old implementation accidentally kept only ``[0]``.
    """
    a, b = np.asarray(before, dtype=np.float32), np.asarray(after, dtype=np.float32)
    if a.ndim == 2: a = a[None]
    if b.ndim == 2: b = b[None]
    if a.shape != b.shape or not len(a):
        raise ValueError(f"saliency shapes must match and be non-empty: {a.shape} vs {b.shape}")
    scores = []
    for x, y in zip(a, b):
        x = (x - x.min()) / (np.ptp(x) + 1e-9)
        y = (y - y.min()) / (np.ptp(y) + 1e-9)
        xf, yf = x.ravel(), y.ravel()
        if xf.std() < 1e-9 or yf.std() < 1e-9:
            scores.append(float(np.abs(xf - yf).mean()))
            continue
        corr = float(np.corrcoef(xf, yf)[0, 1])
        scores.append(float(np.clip(1.0 - corr, 0.0, 2.0)))
    return float(np.mean(scores))


def randomisation_sanity(model, arch, batch, method="gradcam", targets=None) -> float:
    """Randomise the last block's weights; the saliency map MUST change.

    A method whose output barely moves is not explaining the model -- it is an
    edge detector. This has failed for published methods before, so it is
    checked once per architecture rather than assumed.
    """
    import copy
    import torch
    cam, _ = make_cam(model, arch, method)
    if cam is None:
        return float("nan")
    a = cam(input_tensor=batch, targets=targets)
    m2 = copy.deepcopy(model)
    layers = cam_target_layers(m2, arch)
    if layers:
        for p in layers[-1].parameters():
            torch.nn.init.normal_(p, std=0.1)
    cam2, _ = make_cam(m2, arch, method)
    b = cam2(input_tensor=batch, targets=targets)
    return saliency_change_score(a, b)


def insertion_deletion(model, x, sal, target, steps=32, mode="deletion",
                       head_type="coral") -> float:
    """Faithfulness. Deletion: confidence should FALL fast. Insertion: RISE fast."""
    import torch
    import torch.nn.functional as F
    dev = x.device
    flat = sal.ravel()
    order = np.argsort(-flat)
    n = len(order)
    base = torch.zeros_like(x) if mode == "insertion" else x.clone()
    scores = []
    with torch.no_grad():
        for k in range(steps + 1):
            cur = base.clone()
            idx = order[: int(n * k / steps)]
            if len(idx):
                ys, xs = np.unravel_index(idx, sal.shape)
                if mode == "insertion":
                    cur[0, :, ys, xs] = x[0, :, ys, xs]
                else:
                    cur[0, :, ys, xs] = 0
            logits = model(cur.to(dev)).float()
            p = (CoralHead.probs(logits)[0, target] if head_type == "coral"
                 else F.softmax(logits, 1)[0, target])
            scores.append(float(p))
    return float(np.trapz(scores, dx=1.0 / steps))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--isolated-train":
        raise SystemExit(_isolated_train_child(sys.argv[2], sys.argv[3]))
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        raise SystemExit(0 if selftest() else 1)
