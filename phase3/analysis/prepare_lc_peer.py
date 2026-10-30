"""Pin the already independently verified v1 review comparison table."""
import hashlib,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];old=ROOT/'tread_two/nb09_completion/5aa7e27dd416'
m=json.loads((old/'REMOTE_MANIFEST.json').read_text());meta=m['files']['cohort_metrics.csv']
target=ROOT/'tread_two/lc_review/assets/v1_cohort_metrics.csv'
assert hashlib.sha256(target.read_bytes()).hexdigest()==meta['sha256']
(target.parent/'PEER.json').write_text(json.dumps(dict(revision='93d7ea34c5539769d122330141f96b1a23083b7a',contract=m['contract'],source_file='cohort_metrics.csv',sha256=meta['sha256'],bytes=meta['bytes'],scope='Previously verified v1 descriptive endpoint/probe metrics; no new fit and no single-factor attribution'),indent=2)+'\n',encoding='utf-8')
print('Pinned v1 peer table SHA:',meta['sha256'])
