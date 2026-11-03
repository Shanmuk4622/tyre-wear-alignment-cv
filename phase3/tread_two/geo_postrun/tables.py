"""Atomic derived CSVs with stage receipts; no archive or hidden database."""
import os
from pathlib import Path
import pandas as pd
from common import atomic,file_meta,read

class Tables:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'TABLES.json';self.inventory=read(self.path) if self.path.exists() else {}
    def write(self,name,value):
        frame=value if isinstance(value,pd.DataFrame) else pd.DataFrame(value)
        path=self.root/(name+'.csv');tmp=path.with_name(path.name+'.tmp')
        frame.to_csv(tmp,index=False,float_format='%.17g',lineterminator='\n')
        os.replace(tmp,path);self.inventory[name]=dict(file=name+'.csv',rows=len(frame),columns=list(frame.columns),**file_meta(path));atomic(self.path,self.inventory)
        return frame
    def completed(self,name):
        return name in self.inventory and file_meta(self.root/self.inventory[name]['file'])=={k:self.inventory[name][k] for k in ('bytes','sha256')}
