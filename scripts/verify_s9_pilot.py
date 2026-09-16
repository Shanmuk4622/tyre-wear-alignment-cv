"""Offline contract tests. Synthetic annotations are tests, never human evidence."""
import ast
import base64
import copy
import json
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tyrelib'))
import s9_pilot as p

def main():
    packages=sorted((ROOT/'outputs/s9_pilot').glob('*/MANIFEST.json'),key=lambda f:f.stat().st_mtime)
    package=packages[-1].parent;manifest=json.loads(packages[-1].read_text())
    assert len(manifest['images'])==12 and len({r['session'] for r in manifest['images']})==12
    assert manifest['source_sha256']==p.digest((ROOT/'tyrelib/s9_pilot.py').read_bytes())
    assert (package/'PILOT_12_IMAGES.zip').stat().st_size<=p.MAX_BYTES
    page=(package/'ANNOTATE.html').read_text(encoding='utf-8')
    embedded=json.loads(re.search(r'const DATA=(.*);\nconst \$',page).group(1))
    assert all(p.digest(base64.b64decode(e['image'].split(',')[1]))==e['image_sha256'] for e in embedded['images'])
    qa=ROOT/'outputs/s9_pilot_qa';qa.mkdir(exist_ok=True)
    annotations=[]
    for r in manifest['images']:
        annotations.append(dict(pilot_id=r['pilot_id'],image_sha256=r['image_sha256'],
            points={n:dict(state='visible',x=200 if i%2==0 else 900,y=r['guide_y'][i//2]) for i,n in enumerate(p.POINTS)},
            visual_review='unassessable',note='SYNTHETIC SOFTWARE TEST ONLY',independent_record='unknown'))
    value=dict(version=p.VERSION,package_id=manifest['package_id'],annotations=annotations)
    checked=p.validate_annotations(value,manifest);assert len(checked)==12 and all(r['complete'] for r in checked)
    assert not any(r['healthy_reference_eligible'] for r in checked)
    def rejects(edit):
        bad=copy.deepcopy(value);edit(bad)
        try:p.validate_annotations(bad,manifest)
        except (ValueError,TypeError):return
        raise AssertionError('Invalid input accepted')
    rejects(lambda v:v.update(package_id='bad'))
    rejects(lambda v:v['annotations'].append(v['annotations'][0]))
    rejects(lambda v:v['annotations'][0].update(image_sha256='bad'))
    rejects(lambda v:v['annotations'][0]['points']['left_upper'].update(x=float('nan')))
    rejects(lambda v:v['annotations'][0]['points']['left_upper'].update(y=4))
    rejects(lambda v:v['annotations'][0]['points']['left_upper'].update(x=1000))
    rejects(lambda v:v['annotations'][0]['points']['left_upper'].update(state='occluded'))
    rejects(lambda v:v['annotations'][0].update(visual_review='healthy'))
    partial=copy.deepcopy(value);del partial['annotations'][0]['points']['left_upper']
    assert sum(r['complete'] for r in p.validate_annotations(partial,manifest))==11
    issue=copy.deepcopy(value);issue['annotations'][0].update(visual_review='visible_issue',note='')
    assert not p.validate_annotations(issue,manifest)[0]['complete']
    claimed=copy.deepcopy(value);claimed['annotations'][0].update(healthy_reference_eligible=True,independent_record='available')
    assert p.validate_annotations(claimed,manifest)[0]['healthy_reference_eligible'] is False
    fixture=qa/'TEST_ONLY_annotations.json';p.write_json(fixture,value)
    reviewed=p.review(fixture,package,qa)
    result=json.loads((reviewed/'REVIEW.json').read_text());assert result['status']=='needs_human_review'
    assert result['training_approved'] is False
    with patch('huggingface_hub.HfApi') as api:
        api.return_value.upload_folder.return_value.oid='TEST_NO_REMOTE_WRITE'
        p.publish(reviewed,'TEST_TOKEN','TEST_ONLY')
        args=api.return_value.upload_folder.call_args.kwargs
        assert 'ANNOTATE.html' not in args['allow_patterns']
        assert api.return_value.upload_folder.call_count==1
    template=(ROOT/'tyrelib/pilot_template.html').read_text(encoding='utf-8')
    js=template.split('<script>')[1].split('</script>')[0].replace('__PILOT_DATA__',json.dumps(embedded))
    jsfile=qa/'syntax.js';jsfile.write_text(js,encoding='utf-8')
    subprocess.run(['node','--check',str(jsfile)],check=True,capture_output=True)
    for name in ['NB21_S9_Annotation_Pilot.ipynb','NB22_S9_Pilot_Review.ipynb']:
        nb=json.loads((ROOT/'notebooks'/name).read_text());assert nb['metadata']['kaggle']['isGpuEnabled'] is False
        code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        for c in nb['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']));assert not c['outputs']
        assert base64.b64encode((ROOT/'tyrelib/s9_pilot.py').read_bytes()).decode() in code
        assert base64.b64encode((ROOT/'tyrelib/pilot_template.html').read_bytes()).decode() in code
    summary=dict(status='passed',package_id=manifest['package_id'],images=12,
        original_image_hashes_verified=True,zip_bytes=(package/'PILOT_12_IMAGES.zip').stat().st_size,
        notebook_syntax_and_embedded_source=True,invalid_coordinate_identity_visibility_tests=True,
        partial_labels_preserved=True,healthy_claim_never_auto_approved=True,
        browser_check='separately exercised skip states, counter, navigation, JSON download and import on localhost',
        kaggle_execution='not yet run',hf_writes=False,training=False)
    p.write_json(qa/'VALIDATION.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
