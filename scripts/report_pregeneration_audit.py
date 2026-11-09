"""Read-only diagnostic curves. Never select or refit a routing method."""
from pathlib import Path
import numpy as np
from aca.storage import read_json,atomic_json,fingerprint
from aca.pregeneration import (PROTOCOL,load_development,scalar_replay,random_match,
    summarize,interval,prefill_replay,match_family)

repo=Path(__file__).resolve().parents[1]
out=repo/'outputs/pregeneration-audit'
report=read_json(out/'results.json')
assert report['protocol_hash']==fingerprint(PROTOCOL)
cal,d,original=load_development(repo/'outputs/kaggle-qwen15-v6/outputs/qwen15screen',repo/'results/qwen15_methods.json')
indices=np.random.default_rng(PROTOCOL['seed']).integers(0,128,size=(10000,128))
# These are ALL cuts from the pre-analysis grid, not new assessment-selected rules.
grid={};fixed={}
for part,x in [('calibration',cal),('assessment',d)]:
    grid[part]={}
    for signal in PROTOCOL['signals']:
        values=[]
        for threshold in PROTOCOL['thresholds']:
            if not 0<=threshold<=1:continue
            r=scalar_replay(x,signal,threshold,'ic-first')
            values.append(dict(threshold=threshold,**summarize(r,x),random=random_match(r,x)))
        grid[part][signal]=values
        r=scalar_replay(x,signal,.5,'ic-first');late=scalar_replay(x,signal,.5,'answer-first')
        fixed.setdefault(signal,{})[part]=dict(**summarize(r,x),random=random_match(r,x),
            answer_first_seconds=float(late['seconds'].mean()),saving_seconds=float((late['seconds']-r['seconds']).mean()),
            saving_fraction=float((late['seconds']-r['seconds']).mean()/late['seconds'].mean()),
            **interval(random_match(r,x,indices),1))
local_changes={}
for family,record in report['families'].items():
    if not family.startswith('prefill-selective'):continue
    rho=float(family.split('rho')[1]);local_changes[family]=[]
    for point in record['points']:
        p=point['policy'];r=prefill_replay(d,**p,rho=rho)
        base=prefill_replay(d,p['threshold'],p['ic_threshold'],0.,rho)
        changed=r['strong']!=base['strong'];delta=r['correct']-base['correct']
        local_changes[family].append(dict(penalty=point['penalty'],changes=int(changed.sum()),
            beneficial=int(np.sum(changed&(delta>0))),harmful=int(np.sum(changed&(delta<0))),
            neutral=int(np.sum(changed&(delta==0))),accuracy_change=float(delta.mean()),
            incremental_scenario_seconds=float((r['seconds']-base['seconds']).mean())))
decisions=read_json(out/'assessment_decisions.json');matched=[]
for gamma in PROTOCOL['matched_budgets']:
    values={}
    for family in report['families']:
        routes=[{k:np.array(v) for k,v in r.items() if k!='penalty'} for r in decisions[family]]
        values[family]=match_family(routes,d,gamma)
    matched.append(dict(budget_fraction=gamma,methods=values))
# Paired AUROC contrasts describe this sample, not a routing-benefit test.
from sklearn.metrics import roc_auc_score
auc_delta={};s=d['small_correct']
for other in ['top_right','unweighted']:
    values=[]
    for ix in indices:
        if len(np.unique(s[ix]))==2:
            values.append(roc_auc_score(s[ix],d['weighted'][ix])-roc_auc_score(s[ix],d[other][ix]))
    auc_delta[other]=dict(weighted_minus_other=report['diagnostics']['weighted']['correctness_auroc']-report['diagnostics'][other]['correctness_auroc'],
        pointwise_95=np.quantile(values,[.025,.975]).tolist())
extra=dict(scope='Unselected fixed-grid and cost-sensitivity diagnostics only; no new policy selection',
    grid=grid,illustrative_unselected_threshold_05=fixed,selective_changes=local_changes,
    matched_family_summaries=matched,correctness_auroc_contrasts=auc_delta,
    caveat='Threshold .5 is an illustrative member of the fixed grid, not a calibration-selected policy. Prefill fractions are assumptions, not measured component costs.')
atomic_json(out/'diagnostics.json',extra)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
colors={'weighted':'#176b91','top_right':'#b95732','unweighted':'#398455'}
labels={'weighted':'Weighted IC','top_right':'Top-right P(Yes)','unweighted':'Unweighted final-window mean'}
for ax,part in zip(axes[:2],['calibration','assessment']):
    x=cal if part=='calibration' else d
    p=np.linspace(0,1,101)
    ax.plot(x['small_seconds'].mean()+p*(x['strong_seconds']-x['small_seconds']).mean(),
        100*(x['small_correct'].mean()+p*(x['strong_correct']-x['small_correct']).mean()),color='black',lw=1.8,label='Random model preselection')
    for signal in PROTOCOL['signals']:
        rows=grid[part][signal]
        ax.plot([r['mean_seconds'] for r in rows],[100*r['accuracy'] for r in rows],color=colors[signal],lw=1.3,label=labels[signal],alpha=.85)
    ax.scatter([x['small_seconds'].mean(),x['strong_seconds'].mean()],
        [100*x['small_correct'].mean(),100*x['strong_correct'].mean()],color='black',marker='*',s=75,zorder=6,label='Calibration-selected IC rules')
    ax.set(title=f'{part.title()} n=128',xlabel='Warm component seconds / query',ylabel='Exact match (%)')
    if part=='assessment':
        for key,marker,color,label in [('original-adaptive','D','#7c559d','Original adaptive'),('original-cheap','x','#878d32','Original cheap')]:
            rows=report['families'][key]['points']
            ax.scatter([r['assessment']['mean_seconds'] for r in rows],[100*r['assessment']['accuracy'] for r in rows],marker=marker,s=32,color=color,label=label)
        oracle=report['oracle_preselection']
        ax.plot(oracle['mean_seconds'],100*np.array(oracle['accuracy']),color='#999999',ls='--',lw=1,label='Free label oracle')
        r=fixed['weighted']['assessment']
        ax.plot([r['mean_seconds'],r['answer_first_seconds']],[100*r['accuracy']]*2,color=colors['weighted'],lw=2.5)
        ax.annotate('Same cut: 0.283 s saved',xy=(r['answer_first_seconds'],100*r['accuracy']),
            xytext=(1.,49),fontsize=8,arrowprops=dict(arrowstyle='-',color='#666666'))
    ax.legend(loc='lower right',fontsize=6.7,frameon=False)
ax=axes[2]
for kind,color,marker,label in [('cheap','#398455','s','Cheap prefill'),('selective','#176b91','o','Prefill + selective IC')]:
    for rho in PROTOCOL['prefill_fractions']:
        points=report['families'][f'prefill-{kind}-rho{rho:g}']['points']
        ax.scatter([rho]*len(points),[100*p['random']['gain'] for p in points],color=color,marker=marker,s=33,alpha=.75,
            label=label if rho==0 else None)
ax.axhline(0,color='#888888',lw=1);ax.axhline(3,color='#777777',lw=1,ls='--',label='Indicative +3 pp target')
ax.set(xlabel='ASSUMED prefill fraction of small QA (ρ)',ylabel='Gain over random / clipped endpoint (pp)',
    title='Unmeasured prefill sensitivity',xticks=[0,.25,.5,1],ylim=(-2,3.3))
ax.legend(loc='upper right',fontsize=7,frameon=False)
fig.suptitle('Development-only architecture audit: curves are unselected fixed cuts; stars are selected endpoints',fontsize=12)
figdir=repo/'deliverables/figures';figdir.mkdir(exist_ok=True,parents=True)
for ext in ['png','pdf']:fig.savefig(figdir/f'pregeneration_audit.{ext}',dpi=300)
plt.close(fig)
print('Saved fixed-grid diagnostics and figure; no policy or threshold changed')
