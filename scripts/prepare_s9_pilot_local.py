"""Local read-only data preparation convenience. Does not publish anything."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tyrelib'))
import s9_pilot as p
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',default='D:/Dataset Download/Tire Dataset Prepared/FINAL')
    args=parser.parse_args()
    print(p.prepare(args.data_root,ROOT/'outputs/s9_pilot',ROOT/'tyrelib/pilot_template.html'))
