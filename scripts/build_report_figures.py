"""Offline report figures; read archived results without changing evaluations."""
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aca.storage import read_json

repo=Path(__file__).resolve().parents[1]
out=repo/'deliverables/figures';out.mkdir(parents=True,exist_ok=True)
for old,new in [('confirmatory_accuracy_cost','original_confirmatory'),('exploratory_accuracy_cost','original_exploratory')]:
    for ext in ['png','pdf']:
        shutil.copyfile(repo/f'results/review_figures/{old}.{ext}',out/f'{new}.{ext}')
d=read_json(repo/'results/qwen15_exploratory_mixture.json')['diagnostics']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(7.5,4.2),layout='constrained')
x=np.arange(2);width=.33
for off,values,label,color in [(-.5,[d['small_correctness_probe_auroc'],d['ic_correctness_auroc']],'Small correctness; all 512 questions','#176b91'),
                              (.5,[d['low_probe_rescue_auroc_among_small_errors'],d['low_ic_rescue_auroc_among_small_errors']],'Low confidence predicts rescue; 376 small errors','#b95732')]:
    rect=ax.bar(x+off*width,values,width,label=label,color=color)
    ax.bar_label(rect,fmt='%.3f',padding=4)
ax.axhline(.5,color='#777777',ls='--',lw=1,label='Chance ranking')
ax.set(xticks=x,xticklabels=['Cheap correctness probe','Internal Confidence'],ylabel='AUROC',ylim=(0,1),title='Original test: two distinct prediction targets')
ax.legend(loc='upper right',frameon=False,fontsize=9)
for ext in ['png','pdf']:fig.savefig(out/f'correctness_vs_rescue.{ext}',dpi=300)
plt.close(fig)

original=read_json(repo/'results/qwen15_confirmatory.json')
s=next(r['mean_seconds'] for r in original['results'] if r['id']=='small-only')
t=next(r['mean_seconds'] for r in original['results'] if r['id']=='strong-only')
rho=s/t;lift=t/(t-s)
fig,ax=plt.subplots(figsize=(7.5,4.2),layout='constrained')
ratio=np.linspace(0,.8,201)
ax.plot(ratio,1/(1-ratio),color='#176b91',lw=2.2,label='Required lift before router/IC cost')
ax.scatter([rho],[lift],color='#b95732',zorder=3)
ax.annotate(f'Original mean-cost illustration\ns/t = {rho:.3f}; lift > {lift:.3f}',(rho,lift),xytext=(.12,2.7),arrowprops=dict(arrowstyle='-',color='#555555'))
ax.set(xlabel='Small / strong QA service cost (s/t)',ylabel='Selected / global mean correctness benefit',title='Constant-cost cascade break-even condition',ylim=(.9,5.3))
ax.legend(loc='upper left',frameon=False,fontsize=9)
for ext in ['png','pdf']:fig.savefig(out/f'cascade_break_even.{ext}',dpi=300)
plt.close(fig)
from aca.benefit_analysis import plot as plot_benefit
plot_benefit(read_json(repo/'results/benefit_development.json'),out)
print('Wrote five figure pairs; original numerical results unchanged')
