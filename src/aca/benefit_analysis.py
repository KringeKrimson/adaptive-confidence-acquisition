"""Read-only diagnostics and publication figures for the bounded extension.

Never fits a predictor or changes a routing threshold. Does not open test data.
"""
from pathlib import Path
import argparse
import numpy as np
from .storage import read_json,atomic_json,fingerprint
from .benefit import PROTOCOL,read_development


def analyze(root,analysis,out,original_methods):
    root,analysis,out=map(Path,[root,analysis,out]);out.mkdir(parents=True,exist_ok=True)
    report=read_json(analysis/'development.json');methods=read_json(analysis/'methods.json')
    assert fingerprint(methods)==report['methods_hash']
    assert fingerprint(PROTOCOL)==report['protocol_hash']==methods['protocol_hash']
    rows=read_development(root,original_methods)
    assessment=[r for r in rows if r['partition']=='assessment']
    assert [r['query']['id'] for r in assessment]==methods['assessment_ids_excluded_from_selection']
    from sklearn.metrics import mean_squared_error,roc_auc_score
    from scipy.stats import spearmanr
    s,t,cs,ct=[np.asarray([r[k] for r in assessment],float) for k in ['small_correct','strong_correct','small_seconds','strong_seconds']]
    delta=t-s;diagnostics={}
    auc=lambda y,p:float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
    for kind,values in report['assessment_predictions'].items():
        pred=np.asarray(values);seconds=np.asarray(report['measured_router_seconds'][kind])
        diagnostics[kind]=dict(benefit_mse=float(mean_squared_error(delta,pred[:,0])),
            benefit_spearman=float(spearmanr(delta,pred[:,0]).statistic),
            positive_benefit_auroc=auc(delta>0,pred[:,0]),harm_auroc=auc(delta<0,-pred[:,0]),
            rescue_auroc_among_small_errors=auc(t[s==0],pred[s==0,0]),
            mean_router_seconds=float(seconds.mean()),p50_router_seconds=float(np.median(seconds)),
            p95_router_seconds=float(np.quantile(seconds,.95)),
            small_cost_rmse=float(np.sqrt(mean_squared_error(cs,pred[:,1]))),
            strong_cost_rmse=float(np.sqrt(mean_squared_error(ct,pred[:,2]))))
    partition_summary={}
    for partition in ['train','calibration','assessment']:
        selected=[r for r in rows if r['partition']==partition]
        ss,tt=np.array([[r['small_correct'],r['strong_correct']] for r in selected]).T
        partition_summary[partition]=dict(n=len(selected),small_accuracy=float(ss.mean()),strong_accuracy=float(tt.mean()),
            rescues=int(np.sum((ss==0)&(tt==1))),harms=int(np.sum((ss==1)&(tt==0))),
            small_service_seconds=float(sum(r['small_seconds'] for r in selected)),
            strong_service_seconds=float(sum(r['strong_seconds'] for r in selected)))
    details=[];lines=['| Method | λ | Cal. gain, pp | Assessment EM (%) | s/query | Strong (%) | Gain vs random, pp | Familywise 95%, pp | Gate pass |',
                      '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for i,p in enumerate(report['points']):
        a=p['assessment'];m=a['random_preselection'];lo,hi=np.array(a['familywise_95'])*100
        details.append(dict(id=f'{p["kind"]}:{p["penalty"]}',**p))
        lines.append(f'| {p["kind"]} | {p["penalty"]:g} | {100*p["calibration"]["random_preselection"]["gain"]:+.2f} | {100*a["accuracy"]:.2f} | {a["mean_seconds"]:.4f} | {100*a["strong_rate"]:.2f} | {100*m["gain"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {p["passed"]} |')
    (out/'development_table.md').write_text('\n'.join(lines)+'\n')
    result=dict(scope='Descriptive development diagnostics; no new confirmatory inference or tuning',
        status=report['status'],protocol_hash=report['protocol_hash'],methods_hash=report['methods_hash'],
        partition_summary=partition_summary,assessment_diagnostics=diagnostics,
        policies_not_modified=True,original_final_test_predictions_read=0,fresh_test_predictions_read=0,
        finite_frontier=finite_frontier(report),
        new_training_inference=dict(expected_missing_strong_answers=1024,
            measured_QA_service_seconds=partition_summary['train']['strong_service_seconds'],
            loads_and_warmups='Retained separately in development/overhead; not amortized into per-query service cost'),
        statistical_note='Primary gains and intervals copied from the frozen evaluator; Bonferroni16 paired bootstrap. Diagnostics are descriptive point estimates.')
    atomic_json(out/'diagnostics.json',result)
    plot(report,out)
    return result


def finite_frontier(report):
    """Describe fixed points relative to other points and the random-model line."""
    curve=report['random_preselection_curve'];low,high=curve[0],curve[-1]
    candidates=[dict(id=f'{p["kind"]}:{p["penalty"]}',accuracy=p['assessment']['accuracy'],mean_seconds=p['assessment']['mean_seconds']) for p in report['points']]
    candidates += [dict(id=f'{p["method"]}:{p["penalty"]}',accuracy=p['accuracy'],mean_seconds=p['mean_seconds']) for p in report['baselines']]
    remaining=[];line_dominated=[]
    for p in candidates:
        probability=np.clip((p['mean_seconds']-low['mean_seconds'])/(high['mean_seconds']-low['mean_seconds']),0,1)
        cost=(1-probability)*low['mean_seconds']+probability*high['mean_seconds']
        accuracy=(1-probability)*low['accuracy']+probability*high['accuracy']
        dominated=(cost<=p['mean_seconds']+1e-12 and accuracy>=p['accuracy']-1e-12 and
                   (cost<p['mean_seconds']-1e-12 or accuracy>p['accuracy']+1e-12))
        if dominated:line_dominated.append(p['id']);continue
        if any(q['mean_seconds']<=p['mean_seconds']+1e-12 and q['accuracy']>=p['accuracy']-1e-12 and
               (q['mean_seconds']<p['mean_seconds']-1e-12 or q['accuracy']>p['accuracy']+1e-12) for q in candidates):continue
        remaining.append(p)
    return dict(fixed_points_not_dominated_by_line_or_other_points=remaining,
        fixed_points_dominated_by_random_preselection=line_dominated,
        interpretation='Descriptive assessment frontier candidates, not new selected operating points. If candidates lie above the random line, portions of that line can cease to be efficient. No new policy mixtures are evaluated.')


def plot(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained',gridspec_kw={'width_ratios':[1.05,1]})
    ax=axes[0];curve=report['random_preselection_curve']
    ax.plot([r['mean_seconds'] for r in curve],[100*r['accuracy'] for r in curve],color='black',lw=1.8,label='Random model preselection')
    styles={'preselection':('#176b91','o'),'cascade':('#b95732','^')}
    for kind,(color,marker) in styles.items():
        pts=[r for r in report['points'] if r['kind']==kind]
        ax.scatter([r['assessment']['mean_seconds'] for r in pts],[100*r['assessment']['accuracy'] for r in pts],color=color,marker=marker,s=35,label='Benefit '+kind,zorder=4)
    baselines=report['baselines']
    for method,color,marker,label in [('same-grid-correctness','#398455','s','New correctness control'),
        ('original-adaptive','#7c559d','D','Original adaptive IC'),('original-cheap-only','#889947','x','Original cheap'),
        ('original-always-ic','#8c6b29','v','Original always IC'),('original-random-ic','#999999','+','Original random IC')]:
        pts=[r for r in baselines if r['method']==method]
        ax.scatter([r['mean_seconds'] for r in pts],[100*r['accuracy'] for r in pts],color=color,marker=marker,s=27,label=label,alpha=.75)
    ax.set(xlabel='Mean warm service seconds / query',ylabel='Exact match (%)',title='Assessment n=128: all fixed operating points')
    ax.legend(fontsize=8,loc='lower right',frameon=False)
    ax=axes[1]
    for i,p in enumerate(report['points']):
        a=p['assessment'];gain=a['random_preselection']['gain']*100;lo,hi=np.asarray(a['familywise_95'])*100
        color=styles[p['kind']][0] if a['random_preselection']['exact_cost_match'] else '#aaaaaa'
        ax.plot([lo,hi],[i,i],color=color,lw=1.2);ax.plot(gain,i,styles[p['kind']][1],color=color,ms=4)
    ax.axvline(0,color='#555555',lw=1);ax.axvline(3,color='#398455',lw=1,ls='--')
    ax.set(yticks=np.arange(16),yticklabels=[f'{p["kind"]}, λ={p["penalty"]:g}' for p in report['points']],
        xlabel='Benefit method minus cost-matched random EM (pp)',title='Paired familywise 95% intervals; 16 comparisons')
    ax.invert_yaxis();ax.tick_params(axis='y',labelsize=8)
    fig.suptitle('DEVELOPMENT ONLY — bounded benefit-aware extension',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(Path(out)/f'benefit_development.{ext}',dpi=300)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--analysis',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--original-methods',default='results/qwen15_methods.json')
    args=parser.parse_args();print(analyze(args.root,args.analysis,args.out,original_methods=args.original_methods))
