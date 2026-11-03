"""Secret-free subprocess entry point; engineering fixture or downloaded-state seam."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
assert not os.environ.get('HF_TOKEN'), 'The numeric worker must not receive HF_TOKEN'
from common import atomic
from engineering_geo import continue_remote, run_fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('fixture', 'continue'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--weights')
    parser.add_argument('--data-root', default='')
    parser.add_argument('--restored')
    args = parser.parse_args()
    output = Path(args.output)
    try:
        if args.stage == 'fixture':
            run_fixture(output, args.weights, args.data_root)
        else:
            continue_remote(output, args.restored)
    except BaseException as exc:
        atomic(output/'WORKER_STOP.json', dict(kind=type(exc).__name__, message=str(exc),
                    measured_fit_updates=0, scope='engineering_fixture'))
        raise


if __name__ == '__main__':
    main()
