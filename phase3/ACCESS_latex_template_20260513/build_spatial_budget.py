"""Create the matched-budget figure and LaTeX table from the sealed CSV."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
receipt = json.loads((ROOT/'audit/SPATIAL_BUDGET.json').read_text())
source = ROOT/'evidence/spatial_budget_summary.csv'
assert hashlib.sha256(source.read_bytes()).hexdigest()==receipt['outputs']['evidence/spatial_budget_summary.csv']
data = pd.read_csv(source)
main = data[(data.fit=='main') & (data.view==0)]
methods = {'DCT':'DCT','token_PCA':'Token PCA','Jacobian_PCA':'Jacobian PCA',
           'decoded_PCA':'Decoded PCA','moments':'Spatial moments'}
colors = {'DCT':'#555555','token_PCA':'#0072B2','Jacobian_PCA':'#D55E00',
          'decoded_PCA':'#009E73','moments':'#CC79A7'}
styles = {'DCT':('s',':'),'token_PCA':('o','-'),'Jacobian_PCA':('^','--'),
          'decoded_PCA':('v','-.'),'moments':('D','none')}

def score(table, method, k):
    return float(table.loc[(table.method==method)&(table.k==k),'retained'].item())*100

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(2,2,figsize=(7.12,4.6),sharex=True,sharey=True)
budgets = [1,2,4,5,8,16,32]
xs = np.arange(len(budgets))
for i,branch in enumerate(('local','masked')):
    for j,role in enumerate(('validation','test')):
        ax = axes[i,j]
        table = main[(main.branch==branch)&(main.role==role)]
        rand = table[table.method.str.startswith('random')].groupby('k').retained.agg(['min','median','max'])*100
        for k in (1,32):
            rand.loc[k] = score(table,'DCT',k)
        rand = rand.reindex(budgets)
        ax.fill_between(xs,rand['min'],rand['max'],color='#989898',alpha=.20,label='Random range (16)')
        ax.plot(xs,rand['median'],color='#888888',lw=.9,label='Random median')
        for method,label in methods.items():
            marker,ls=styles[method]
            available=[5] if method=='moments' else budgets
            values=[score(table,'DCT' if k in (1,32) else method,k) for k in available]
            ax.plot([budgets.index(k) for k in available],values,marker=marker,ls=ls,
                    color=colors[method],lw=1.15,ms=3.5,label=label)
        ax.set_title(f'({chr(97+i*2+j)}) {branch.capitalize()} / '+('validation' if role=='validation' else 'historical test'),fontsize=9)
        ax.set(xticks=xs,xticklabels=budgets,
               ylim=(min(45,5*np.floor(20*main[main.role.isin(['validation','test'])].retained.min())),103))
        ax.grid(axis='y',alpha=.2)
        if i==1:ax.set_xlabel('Spatial modes including constant')
        if j==0:ax.set_ylabel('Retained advantage (%)')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=4,loc='lower center',frameon=False,fontsize=7)
fig.tight_layout(rect=[0,.09,1,1],pad=.7)
for ext in ('pdf','png'):
    fig.savefig(ROOT/f'figures/spatial_budget.{ext}',dpi=240,bbox_inches='tight',pad_inches=.06)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(7.12, 2.62), sharey=True)
intermediate = [2, 4, 5, 8, 16]
positions = np.arange(len(intermediate))
for ax, branch in zip(axes, ('local', 'masked')):
    ax.axhline(0, color='#777777', lw=.8, ls=':', zorder=0)
    for role, role_label, line in [('validation', 'Validation', '-'), ('test', 'Historical test', '--')]:
        table = main[(main.branch == branch) & (main.role == role)]
        token = np.array([score(table, 'token_PCA', k) for k in intermediate])
        for method in ('Jacobian_PCA', 'decoded_PCA'):
            retained = np.array([score(table, method, k) for k in intermediate])
            ax.plot(positions, retained - token, color=colors[method], ls=line,
                    marker=styles[method][0], ms=3.5, lw=1.1,
                    label=f'{methods[method]} / {role_label}')
    ax.set(title=f'{branch.capitalize()} branch', xticks=positions,
           xticklabels=intermediate, xlabel='Spatial modes including constant')
    ax.grid(axis='y', alpha=.18)
axes[0].set_ylabel('Difference from token PCA (pp)')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=2, loc='lower center', frameon=False, fontsize=7.5)
fig.tight_layout(rect=[0, .21, 1, 1], pad=.65)
for ext in ('pdf', 'png'):
    fig.savefig(ROOT/f'figures/spatial_budget_differences.{ext}', dpi=240,
                bbox_inches='tight', pad_inches=.06)
plt.close(fig)

five=main[main.k==5]
scopes=[('local','validation'),('local','test'),('masked','validation'),('masked','test')]
lines=[r'% Generated from the sealed spatial-budget CSV.',r'\newcommand{\budgetrows}{%']
for method,label in methods.items():
    vals=[score(five[(five.branch==b)&(five.role==r)],method,5) for b,r in scopes]
    lines.append(label+' & '+' & '.join(f'{v:.2f}' for v in vals)+r' \\')
random=[]
for b,r in scopes:
    sub=five[(five.branch==b)&(five.role==r)&five.method.str.startswith('random')]
    assert len(sub)==16
    random.append(f'{100*sub.retained.median():.2f}')
lines.append('Random median (16) & '+' & '.join(random)+r' \\')
lines.append('}')
(ROOT/'generated/spatial_budget_rows.tex').write_text('\n'.join(lines)+'\n')
print('Spatial budget figure and table built from sealed results.')
