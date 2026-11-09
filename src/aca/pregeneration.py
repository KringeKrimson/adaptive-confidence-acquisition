"""Development-only architecture audit. No inference and no test-cache access."""
from pathlib import Path
import argparse
import numpy as np
from .storage import read_json, atomic_json, fingerprint, load_records
from .ic import aggregate_ic
from .routing import Policy, replay

PROTOCOL = dict(
    name='development_only_pregeneration_architecture_audit', version=1, seed=4905, bootstrap_repeats=10000,
    scope='Exploratory reuse of original128 calibration/128 assessment. No original or reserved test data, new model calls, predictor fitting, model or IC hyperparameter changes.',
    original_methods_hash='59ef6e267603260190ade5468e7e54d248c25ba11ababec32cdb276d220fa8ed',
    signals=['weighted','top_right','unweighted'],
    signal_definitions='Weighted=unchanged alpha1 IC; top_right=p_yes[-1,-1]; unweighted=mean of cached final6tokens by29states only. No full-question matrix is cached. All share the same measured full IC forward cost; no claimed GPU savings from simpler aggregation.',
    penalties=[0.,.025,.05,.1,.2,.4,.8,1.6], thresholds=[-.01,*np.linspace(0,1,21).tolist(),1.01],
    widths=[0.,.025,.05,.1,.2,.4,1.1],
    calibration='Maximize EM-lambda*mean_total_seconds/calibration_mean_strong_seconds. Ties least cost, then fewer IC acquisitions, then deterministic grid order. Retain all8 points per family. Original frozen rules replayed unchanged. Add same-grid answer-first weighted IC and cheap controls to separate architecture from grid/tuning changes.',
    constants='IC-first outside[0,1] thresholds compile to direct model choice and omit the unused estimator. Interior thresholds always pay IC even if their observed choices happen to be constant. Original archived policies retain original charges.',
    measured_costs='IC-first: I+where(strong,T,S), I=measured IC time for all three signals. Same-rule answer-first: S+I+strong*T. Cheap answer-first: S+P+strong*T. Per-query original CPU probe times reused. Generic threshold/dispatch arithmetic not separately timed. These are warm additive replays, not deployment latency.',
    prefill='QA-prefill time F and retained-KV continuation costs were not recorded. Only explicit idealized sensitivity scenarios F=rho*S,rho in[0,.25,.5,1] are allowed, with zero extra dispatch/continuation overhead. Cost=S+P+acquire*I+strong*(T-S+F). This pays F on strong branches and includes it in S on small branches; no measured cost or reliable empirical bound is claimed. Tune on calibration separately in each labeled scenario. Weighted selective IC only, plus cheap-only. No IC/QA prompt cache sharing.',
    prefill_fractions=[0.,.25,.5,1.],
    matched_budgets=[.25,.5,.75],
    matching='Random preselection p=clip((policy_cost-meanS)/(meanT-meanS),0,1). Above endpoint costs explicitly unmatched. Fixed-budget comparisons use B=meanS+gamma*(meanT-meanS); within each family mix closest lower/upper fixed policies by COST ONLY, with direct small/strong endpoints. Recompute brackets and weights per bootstrap; never select by assessment accuracy.',
    uncertainty='10000 paired question bootstrap seed4905. Fixed measured-family gains: pointwise95 plus Bonferroni40 across5families*8points. Fixed-budget weighted-minus-random/top/mean: pointwise95 plus Bonferroni9 across3budgets*3contrasts. Separate families; no omnibus claim. AUROCs and prefill scenarios descriptive only.',
    oracle='Free label-aware fractional oracle chooses smallest incremental cost rescues, avoids harms. Diagnostic upper bound, never a fitted policy or assessment selection rule.',
    stopping='No final test in this task. A fresh confirmatory proposal requires credible development gain over random AND simpler confidence at matched cost, with practical3pp as an indicative target. If only unknown-prefill scenarios are promising, recommend a bounded development timing measurement, not a final-test run.')


def signals(matrix):
    m=np.asarray(matrix,float)
    if m.shape!=(6,29):raise ValueError('Expected cached6x29 final-window matrix')
    return dict(weighted=aggregate_ic(m,1.),top_right=float(m[-1,-1]),unweighted=float(m.mean()))


def scalar_route(score,threshold):
    return np.asarray(score)<threshold


def scalar_replay(d,signal,threshold,architecture):
    strong=scalar_route(d[signal],threshold)
    # Outside the support is a globally constant rule, not a data-dependent bypass.
    acquire=np.full(len(strong),0.<=threshold<=1.)
    confidence=acquire*d['ic_seconds']
    if architecture=='ic-first':cost=confidence+np.where(strong,d['strong_seconds'],d['small_seconds'])
    elif architecture=='answer-first':cost=d['small_seconds']+confidence+strong*d['strong_seconds']
    else:raise ValueError(architecture)
    return dict(correct=np.where(strong,d['strong_correct'],d['small_correct']),seconds=cost,
                strong=strong.astype(float),acquire=acquire.astype(float))


def prefill_replay(d,threshold,ic_threshold,width,rho):
    if not 0<=rho<=1:raise ValueError('Sensitivity fraction outside[0,1]')
    acquire=np.abs(d['cheap']-threshold)<width
    strong=np.where(acquire,d['weighted']<ic_threshold,d['cheap']<threshold)
    cost=d['small_seconds']+d['probe_seconds']+acquire*d['ic_seconds']+strong*(d['strong_seconds']-d['small_seconds']+rho*d['small_seconds'])
    return dict(correct=np.where(strong,d['strong_correct'],d['small_correct']),seconds=cost,
                strong=strong.astype(float),acquire=acquire.astype(float))


def summarize(r,d):
    a=r['strong'];s=d['small_correct'];t=d['strong_correct']
    return dict(accuracy=float(r['correct'].mean()),mean_seconds=float(r['seconds'].mean()),
        ic_rate=float(r['acquire'].mean()),strong_rate=float(a.mean()),
        rescued=float(np.sum(a*(s==0)*(t==1))),harmed=float(np.sum(a*(s==1)*(t==0))),
        avoided_harms=float(np.sum((1-a)*(s==1)*(t==0))),missed_rescues=float(np.sum((1-a)*(s==0)*(t==1))))


def random_match(r,d,indices=None):
    mean=(lambda x:np.asarray(x)[indices].mean(axis=1)) if indices is not None else (lambda x:float(np.mean(x)))
    lo,hi,target=mean(d['small_seconds']),mean(d['strong_seconds']),mean(r['seconds'])
    if np.any(hi<=lo):raise ValueError('Nonpositive model cost gap')
    p=np.clip((target-lo)/(hi-lo),0,1);accuracy=(1-p)*mean(d['small_correct'])+p*mean(d['strong_correct'])
    if indices is not None:return mean(r['correct'])-accuracy
    cost=(1-p)*lo+p*hi
    return dict(probability=float(p),accuracy=float(accuracy),mean_seconds=float(cost),
        exact_cost_match=bool(np.isclose(cost,target,rtol=0,atol=1e-10)),gain=float(mean(r['correct'])-accuracy))


def interval(values,family):
    return dict(pointwise_95=np.quantile(values,[.025,.975]).tolist(),
        familywise_95=np.quantile(values,[.05/(2*family),1-.05/(2*family)]).tolist())


def select(policies,cal,run):
    if set(cal['partition'])!={'calibration'}:raise ValueError('Calibration only')
    evaluated=[(p,run(p,cal)) for p in policies];scale=cal['strong_seconds'].mean();selected=[]
    for penalty in PROTOCOL['penalties']:
        p,r=max(evaluated,key=lambda pr:(pr[1]['correct'].mean()-penalty*pr[1]['seconds'].mean()/scale,
            -pr[1]['seconds'].mean(),-pr[1]['acquire'].mean()))
        selected.append(dict(penalty=penalty,policy=p,calibration=summarize(r,cal),calibration_random=random_match(r,cal)))
    return selected


def model_endpoint(d,strong):
    n=len(d['ids'])
    return dict(correct=d['strong_correct' if strong else 'small_correct'].copy(),
        seconds=d['strong_seconds' if strong else 'small_seconds'].copy(),strong=np.full(n,float(strong)),acquire=np.zeros(n))


def match_family(routes,d,gamma,indices=None):
    """Nearest cost bracket only. Assessment labels never choose a bracket."""
    routes=[model_endpoint(d,False),*routes,model_endpoint(d,True)]
    if indices is None:
        indices=np.arange(len(d['ids']))[None,:];single=True
    else:single=False
    costs=np.array([r['seconds'][indices].mean(axis=1) for r in routes]).T
    target=d['small_seconds'][indices].mean(axis=1)+gamma*(d['strong_seconds'][indices].mean(axis=1)-d['small_seconds'][indices].mean(axis=1))
    lower=np.where(costs<=target[:,None]+1e-12,costs,-np.inf).argmax(axis=1)
    upper=np.where(costs>=target[:,None]-1e-12,costs,np.inf).argmin(axis=1)
    batch=np.arange(len(indices));cl=costs[batch,lower];cu=costs[batch,upper]
    w=np.divide(target-cl,cu-cl,out=np.zeros_like(target),where=cu-cl>1e-12)
    result={}
    for k in routes[0]:
        values=np.array([r[k][indices].mean(axis=1) for r in routes]).T
        result[k]=(1-w)*values[batch,lower]+w*values[batch,upper]
    result['lower_index']=lower;result['upper_index']=upper;result['upper_weight']=w
    return {k:float(v[0]) for k,v in result.items()} if single else result


def load_development(root,methods):
    root=Path(root);rows=read_json(root/'viability_rows.json');original=read_json(methods)
    assert fingerprint(original)==PROTOCOL['original_methods_hash']
    allowed=set(original['selection_ids'])|set(original['assessment_ids_excluded_from_selection'])
    assert len(rows)==256 and {r['id'] for r in rows}==allowed
    q=[q for q in read_json(root/'queries.json')['queries'] if q['id'] in allowed]
    assert all(qi['split']=='validation' for qi in q)
    records={r['id']:r for r in load_records(root,'ic',q)}
    for row in rows:
        r=records[row['id']];assert r['output_tokens']==0 and r['forward_passes']==r['model_calls']==1
        assert r['last_k']==6 and r['alpha']==1 and r['yes_no_ids']==[9454,2753]
        assert r['config_hash']==fingerprint(original['inference_manifest'])
        value=signals(r['p_yes']);assert np.isclose(value['weighted'],r['ic'],rtol=0,atol=1e-14)
        assert r['ic']==row['ic'] and r['elapsed_seconds']==row['ic_seconds']
        row.update(value)
    partitions=[]
    for part,ids in [('calibration',original['selection_ids']),('assessment',original['assessment_ids_excluded_from_selection'])]:
        ordered=[next(r for r in rows if r['id']==i) for i in ids];assert all(r['partition']==part for r in ordered)
        d={k:np.array([r[k] for r in ordered]) for k in ordered[0]}
        d['ids']=d.pop('id');partitions.append(d)
    return *partitions,original


def oracle(d,answer_first=False):
    s,t,cs,ct=[d[k] for k in ['small_correct','strong_correct','small_seconds','strong_seconds']]
    increment=ct if answer_first else ct-cs
    if (increment<=0).any():raise ValueError('Oracle diagnostic assumes positive increments; audit before generalizing')
    rescue=np.flatnonzero((s==0)&(t==1));order=rescue[np.argsort(increment[rescue])]
    costs=np.r_[cs.mean(),cs.mean()+np.cumsum(increment[order])/len(s)]
    accuracy=np.r_[s.mean(),s.mean()+np.arange(1,len(order)+1)/len(s)]
    return dict(mean_seconds=costs.tolist(),accuracy=accuracy.tolist(),
        complementarity=dict(both_correct=int(np.sum((s==1)&(t==1))),both_wrong=int(np.sum((s==0)&(t==0))),
            rescues=len(rescue),harms=int(np.sum((s==1)&(t==0)))),free_label_oracle=True)


def evaluate(root,out,methods):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cal,assess,original=load_development(root,methods)
    atomic_json(out/'protocol.json',PROTOCOL)
    families={};frozen={};runners={}
    thresholds=[dict(threshold=t) for t in PROTOCOL['thresholds']]
    for name in PROTOCOL['signals']:
        key='ic-first-'+name
        runners[key]=lambda p,d,n=name:scalar_replay(d,n,p['threshold'],'ic-first')
    runners['answer-first-weighted']=lambda p,d:scalar_replay(d,'weighted',p['threshold'],'answer-first')
    runners['answer-first-cheap-retuned']=lambda p,d:replay(Policy('cheap-only',cheap_threshold=p['threshold']),d)
    # Existing routing.replay expects the original 'ic' field, still retained in rows.
    for key,run in runners.items():frozen[key]=select(thresholds,cal,run)
    # Freeze every sensitivity policy using calibration before evaluating its assessment outcome.
    adaptive=[dict(threshold=t,ic_threshold=u,width=w) for t in PROTOCOL['thresholds']
        for u in PROTOCOL['thresholds'] for w in PROTOCOL['widths']]
    for rho in PROTOCOL['prefill_fractions']:
        for name,choices in [('cheap', [dict(threshold=t,ic_threshold=.5,width=0.) for t in PROTOCOL['thresholds']]),('selective',adaptive)]:
            key=f'prefill-{name}-rho{rho:g}'
            runners[key]=lambda p,d,r=rho:prefill_replay(d,**p,rho=r)
            frozen[key]=select(choices,cal,runners[key])
    selected=dict(protocol_hash=fingerprint(PROTOCOL),families=frozen)
    if (out/'calibration_selected.json').exists():
        assert read_json(out/'calibration_selected.json')==selected,'Calibration selection changed'
    else:atomic_json(out/'calibration_selected.json',selected)
    rng=np.random.default_rng(PROTOCOL['seed']);indices=rng.integers(0,len(assess['ids']),size=(PROTOCOL['bootstrap_repeats'],len(assess['ids'])))
    decisions={}
    for key,points in frozen.items():
        measured=not key.startswith('prefill-');results=[];routes=[]
        for p in points:
            r=runners[key](p['policy'],assess);routes.append(r)
            result=dict(**p,assessment=summarize(r,assess),random=random_match(r,assess))
            if measured:result.update(interval(random_match(r,assess,indices),40))
            results.append(result)
        families[key]=dict(cost_status='measured component replay' if measured else 'unmeasured idealized prefill sensitivity; NOT measured costs',points=results)
        decisions[key]=[dict(penalty=p['penalty'],**{k:v.tolist() for k,v in r.items()}) for p,r in zip(points,routes)]
    for key,kind in [('original-adaptive','adaptive'),('original-cheap','cheap-only')]:
        families[key]=dict(cost_status='original measured component replay',points=[]);decisions[key]=[]
        for p in original['selected']:
            r=replay(Policy(**p[kind]),assess)
            families[key]['points'].append(dict(penalty=p['penalty'],policy=p[kind],assessment=summarize(r,assess),random=random_match(r,assess)))
            decisions[key].append(dict(penalty=p['penalty'],**{k:v.tolist() for k,v in r.items()}))
    matched=[]
    newkeys=['ic-first-'+k for k in PROTOCOL['signals']]
    for gamma in PROTOCOL['matched_budgets']:
        values={};boots={}
        for key in newkeys:
            routes=[{k:np.array(v) for k,v in r.items() if k!='penalty'} for r in decisions[key]]
            values[key]=match_family(routes,assess,gamma);boots[key]=match_family(routes,assess,gamma,indices)
        random_correct=(1-gamma)*assess['small_correct']+gamma*assess['strong_correct']
        comparison={}
        for key in ['random','ic-first-top_right','ic-first-unweighted']:
            comparator=random_correct[indices].mean(axis=1) if key=='random' else boots[key]['correct']
            point=float(random_correct.mean()) if key=='random' else values[key]['correct']
            comparison[key]=dict(gain=values['ic-first-weighted']['correct']-point,**interval(boots['ic-first-weighted']['correct']-comparator,9))
        matched.append(dict(budget_fraction=gamma,methods=values,weighted_contrasts=comparison,
            random_accuracy=float(random_correct.mean()),mean_seconds=float((assess['small_seconds']+gamma*(assess['strong_seconds']-assess['small_seconds'])).mean())))
    from sklearn.metrics import roc_auc_score
    diagnostics={}
    s,t=assess['small_correct'],assess['strong_correct'];errors=s==0
    for signal in ['cheap',*PROTOCOL['signals']]:
        score=assess[signal]
        diagnostics[signal]=dict(correctness_auroc=float(roc_auc_score(s,score)),
            rescue_auroc_among_errors=float(roc_auc_score(t[errors],-score[errors])),
            positive_benefit_auroc=float(roc_auc_score((t-s)>0,-score)))
    result=dict(scope=PROTOCOL['scope'],protocol_hash=fingerprint(PROTOCOL),methods_hash=fingerprint(read_json(out/'calibration_selected.json')),
        n_calibration=len(cal['ids']),n_assessment=len(assess['ids']),families=families,matched_budgets=matched,diagnostics=diagnostics,
        components={p:{k:float(d[k].mean()) for k in ['small_seconds','strong_seconds','ic_seconds','probe_seconds']} for p,d in [('calibration',cal),('assessment',assess)]},
        oracle_preselection=oracle(assess),oracle_answer_first=oracle(assess,True),
        no_new_gpu_inference=True,original_test_records_read=0,reserved_test_records_read=0)
    atomic_json(out/'assessment_data.json',{k:v.tolist() for k,v in assess.items()})
    atomic_json(out/'assessment_decisions.json',decisions);atomic_json(out/'results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--methods',default='results/qwen15_methods.json');args=parser.parse_args()
    r=evaluate(args.root,args.out,args.methods)
    print('Completed development-only pre-generation audit:',r['protocol_hash'])
