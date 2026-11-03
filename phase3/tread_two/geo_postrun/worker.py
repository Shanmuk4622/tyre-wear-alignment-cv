"""Secret-free numeric analysis worker; no Hub, images forwards or optimizer."""
import argparse,os,sys
from pathlib import Path
from common import read,atomic

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['review','passport']);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--contract',required=True);p.add_argument('--dataset',default='');args=p.parse_args()
    assert not os.environ.get('HF_TOKEN'),'Numeric worker must never receive the token'
    import torch
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    if args.stage=='review':
        from pipeline import review
        review(Path(args.input),read(Path(args.input)/'REMOTE_MANIFEST.json'),Path(args.output),args.contract,args.dataset)
    else:
        from passport import run
        run(Path(args.input),Path(args.output),args.contract)

if __name__=='__main__':main()
