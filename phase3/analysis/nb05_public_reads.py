"""Shared bounded public-read retry; no credential or token-file lookup."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT/'training'))
from evidence_hub import delay_for,wait

def retry(fn):
 for attempt in range(6):
  try:return fn()
  except Exception as exc:
   delay=delay_for(exc,attempt)
   if delay is None or attempt==5:raise
   print('Pinned public read: respecting server/backoff delay',delay,flush=True)
   wait(delay)
