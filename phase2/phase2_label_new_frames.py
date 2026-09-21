"""Launch existing LabelMe on new Phase 2 frames only; never old images."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', choices=['video1','video2','video3'])
    parser.add_argument('--check', action='store_true', help='Validate paths/config only; do not open a window')
    args = parser.parse_args()
    video = args.video
    if not video:
        if args.check:
            video = 'video1'
        else:
            print('Label NEW frames only. Existing photos/labels are not opened.\n1: video1 (27)\n2: video2 (93)\n3: video3 (32)')
            choice = input('Choose 1, 2 or 3: ').strip()
            if choice not in ('1','2','3'):
                raise SystemExit('No valid selection. No files changed.')
            video = 'video'+choice
    images = ROOT/'data'/'new_frames'/video
    output = ROOT/'annotation_work'/video
    if not images.is_dir() or not list(images.glob('phase2_*.png')):
        raise SystemExit('New frame directory is missing; run phase2_extract_frames.py first.')
    import labelme.config
    config = labelme.config.load_config(ROOT/'phase2_labelme.yaml', {})
    if config['labels'] != ['tyre','tread','ignore'] or config['with_image_data'] or config['keep_prev']:
        raise RuntimeError('Unexpected annotation configuration')
    if args.check:
        print(f'PASS: {len(list(images.glob("phase2_*.png")))} NEW images; explicit local config; output {output}')
        return
    output.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-B', '-m', 'labelme', str(images), '--output', str(output),
           '--config', str(ROOT/'phase2_labelme.yaml'), '--labels', str(ROOT/'phase2_labels.txt'),
           '--validate-label', 'exact']
    print('Save with Ctrl+S. Finish each reviewed image before moving on. Reopen this same video to resume.')
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))


if __name__ == '__main__': main()
