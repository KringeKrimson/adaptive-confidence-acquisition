"""Final-test evaluation. No fitting, threshold search or assessment optimization."""
from pathlib import Path
import time
import numpy as np
from .storage import atomic_json, bind_manifest, fingerprint, load_records, read_json
from .routing import Policy, actions, replay, summarize, random_replay

PROTOCOL = dict(
    version=1, n_test=512, methods_hash='59ef6e267603260190ade5468e7e54d248c25ba11ababec32cdb276d220fa8ed',
    test_queries_hash='239a66e11323f98ca57c60ad5c055a5b05843613ef1c02beef36ceb34ec80c86',
    seed=4901, bootstrap_repeats=10000, family_size=8, alpha=.05, practical_gain=.03,
    policies='Exactly the eight committed operating points, portable probe and 20 random seed/threshold pairs. No fitting or policy selection on test.',
    primary='Adaptive accuracy minus cheap-only randomized-mixture expected accuracy at matched expected measured service cost, at each of the eight fixed points.',
    cost_matching='Use only the two unique committed cheap-only policies, ordered by threshold. Uniform query-independent mixture probability from measured test mean costs only: clip((adaptive-low)/(high-low),0,1). No test labels select endpoints or weights. Exact expected cost inside endpoint range; above it compare to cheaper high endpoint and mark not an exact match. No extrapolation or new thresholds. Retrospective timing-oracle benchmark, not an online cost scheduler.',
    uncertainty='Paired query bootstrap, seed4901, 10000 resamples. Recompute cost-matching probability in every resample. Pointwise95 and Bonferroni familywise95 across all eight fixed points, including duplicate and zero-acquisition points. Random control secondary intervals condition on the saved 20-seed expected schedules.',
    decision='Support for a meaningful improvement requires an exact-cost-matched point with nonzero IC, point gain >=0.03, and familywise95 lower bound >0. Also report whether the lower bound exceeds0.03; an estimate above0.03 alone does not establish that effect size. If every eligible familywise upper bound <0.03, rule out a >=3-point gain for these tested policies; otherwise an unsupported result can be inconclusive. Report all eight; do not promote a test-selected operating point.',
    effect_threshold='Three percentage points chosen before test inference, consistent with the project minimum routing-gain scale; this cutoff and exact cost matching were not specified in the earlier frozen-policy artifact. This is a transparent pre-test analysis addendum, not a claim they were previously frozen.',
    secondary='Paired accuracy differences against raw same-penalty cheap-only, always-IC, random-IC, small-only and strong-only. Mean/p50/p95 additive latency (weighted branch-mixture quantiles for randomized controls), acquisition/escalation rates, IC-induced routing changes and beneficial/harmful changes against the same adaptive rule with acquisition disabled.',
    frontier='Descriptive nondominated observed mean accuracy/cost points among all fixed policies and baselines, including matched cheap mixtures. No frontier-based retuning or new deployment operating point.',
    costs='Warm batch-one component times measured on T4 GPU0: small QA + required CPU probe + acquired IC + escalated strong QA. Always-IC has no probe charge; strong-only no small QA charge. Model loads, warmups, projection verification and offline cache-collection expenditure separately recorded. Replayed cascade latency is not measured live two-GPU queue/communication latency.',
    decoding='Unchanged pinned Qwen1.5B/Qwen14B, NF4 double quantization with FP32 compute/modules, eager attention, greedy first nonempty answer line/EOS, max64, strict alias exact match. New1536d ordinary-QA prefill features, fixed probe, Qwen IC last6/alpha1/no examples/all29 states.',
    distinction='The frozen test policies and primary matched-budget comparisons are confirmatory under this pre-test addendum. Test frontier identification and diagnostic breakdowns are descriptive. All development results are separate; no result-conditioned protocol changes.')


def verify_methods(methods):
    from .frozen import implementation_hash
    from .collect import source_hash
    if fingerprint(methods)!=PROTOCOL['methods_hash']:
        raise ValueError('Committed frozen policies/probe changed')
    if methods['implementation_hash']!=implementation_hash():
        raise ValueError('Frozen routing/selection implementation changed')
    if source_hash()!=methods['inference_manifest']['source_hash']:
        raise ValueError('Frozen inference implementation changed')
    assert len(methods['selected'])==8


def portable_probe(saved):
    """Restore sklearn's scoring path without fitting any parameter."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    scaler=StandardScaler();scaler.mean_=np.array(saved['mean']);scaler.scale_=np.array(saved['scale'])
    scaler.var_=scaler.scale_**2;scaler.n_features_in_=len(scaler.mean_);scaler.n_samples_seen_=len(saved['training_ids'])
    classifier=LogisticRegression(C=saved['C'],max_iter=2000,random_state=4901)
    classifier.coef_=np.array(saved['coefficients']);classifier.intercept_=np.array(saved['intercept'])
    classifier.classes_=np.array([0,1]);classifier.n_features_in_=len(scaler.mean_);classifier.n_iter_=np.array([0])
    return make_pipeline(scaler,classifier)


def prepare_test(root, methods_path, original_manifest=None):
    from .data import prepare,validate_splits
    root=Path(root);methods=read_json(methods_path);verify_methods(methods)
    inputs=root.parent/'confirmatory_inputs'
    if original_manifest is None:
        pilot=inputs/'pilot'
        prepare(pilot,test=32,revision='0f7faf33a3908546c6fd5b73a660e0f8ff173c2f')
        original=prepare(inputs/'original_manifest',train=1024,validation=256,test=512,seed=4901,
            revision='0f7faf33a3908546c6fd5b73a660e0f8ff173c2f',exclude_manifest=pilot/'queries.json')
    else:
        original=read_json(original_manifest)
    queries=[q for q in original['queries'] if q['split']=='test']
    if len(queries)!=512 or fingerprint(queries)!=PROTOCOL['test_queries_hash']:
        raise ValueError('Reserved final-test cohort changed')
    validate_splits(queries)
    dev=set(methods['probe']['training_ids'])|set(methods['selection_ids'])|set(methods['assessment_ids_excluded_from_selection'])
    assert not {q['id'] for q in queries}&dev
    assert all(q['source_split']=='validation' for q in queries)
    bind_manifest(root,'queries.json',dict(revision=original['revision'],sampling='Previously reserved512 official-validation queries; original seed4901 cohort, pilot excluded',queries=queries))
    bind_manifest(root,'frozen_methods.json',methods)
    bind_manifest(root,'confirmatory_protocol.json',PROTOCOL)
    bind_manifest(root,'model_revisions.json',methods['inference_manifest']['model_revisions'])
    print('CONFIRMATORY FREEZE:',fingerprint(PROTOCOL),'methods:',fingerprint(methods),'test:',fingerprint(queries),flush=True)
    return queries


def matched_budget(adaptive, low, high):
    """Label-blind cost matching. Accuracy is not an input to the mixture weight."""
    target=float(np.mean(adaptive['seconds']));lo=float(np.mean(low['seconds']));hi=float(np.mean(high['seconds']))
    if hi<lo-1e-12: raise ValueError('Cheap endpoint costs are reversed')
    weight=float(np.clip((target-lo)/(hi-lo),0,1)) if hi>lo+1e-12 else 0.
    result={k:(1-weight)*low[k]+weight*high[k] for k in low}
    exact=bool(np.isclose(np.mean(result['seconds']),target,atol=1e-10,rtol=1e-10))
    return result,dict(high_probability=weight,exact_cost_match=exact,unused_budget_seconds=target-float(np.mean(result['seconds'])))


def intervals(samples):
    pointwise=np.quantile(samples,[PROTOCOL['alpha']/2,1-PROTOCOL['alpha']/2]).tolist()
    tail=PROTOCOL['alpha']/(2*PROTOCOL['family_size'])
    return dict(pointwise_95=pointwise,familywise_95=np.quantile(samples,[tail,1-tail]).tolist())


def matched_bootstrap(adaptive,low,high,indices):
    mean=lambda x:np.mean(x[indices],axis=1)
    numerator=mean(adaptive['seconds'])-mean(low['seconds']);denominator=mean(high['seconds'])-mean(low['seconds'])
    weight=np.clip(np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>1e-12),0,1)
    delta=mean(adaptive['correct'])-((1-weight)*mean(low['correct'])+weight*mean(high['correct']))
    return intervals(delta)


def weighted_quantiles(values,weights):
    values,weights=np.asarray(values,dtype=float),np.asarray(weights,dtype=float)
    order=np.argsort(values);cumulative=np.cumsum(weights[order]);cumulative/=cumulative[-1]
    return [float(values[order[min(np.searchsorted(cumulative,q),len(order)-1)]]) for q in [.5,.95]]


def extended_summary(r,d,latency=None):
    quantiles=weighted_quantiles(*latency) if latency is not None else np.quantile(r['seconds'],[.5,.95]).tolist()
    return dict(**summarize(r),p50_seconds=quantiles[0],p95_seconds=quantiles[1],
        latency_quantiles='Weighted randomized-branch replay' if latency is not None else 'Deterministic per-query additive replay',
        mean_ic_seconds=float(np.mean(r['acquire']*d['ic_seconds'])),
        mean_escalation_seconds=float(np.mean(r['strong']*d['strong_seconds'])))


def observed_frontier(results):
    return [r['id'] for r in results if not any(
        other['mean_seconds']<=r['mean_seconds']+1e-12 and other['accuracy']>=r['accuracy']-1e-12 and
        (other['mean_seconds']<r['mean_seconds']-1e-12 or other['accuracy']>r['accuracy']+1e-12)
        for other in results)]


def evaluate(root):
    root=Path(root);methods=read_json(root/'frozen_methods.json');verify_methods(methods)
    if read_json(root/'confirmatory_protocol.json')!=PROTOCOL:
        raise ValueError('Pre-test analysis protocol changed')
    queries=read_json(root/'queries.json')['queries']
    if len(queries)!=512 or fingerprint(queries)!=PROTOCOL['test_queries_hash']:
        raise ValueError('Unexpected final-test cohort')
    manifest=read_json(root/'inference_manifest.json');expected=methods['inference_manifest']
    for k,v in expected.items():
        if k not in {'dataset_hash','runtime'} and manifest.get(k)!=v:
            raise ValueError(f'Frozen inference setting changed: {k}')
    if manifest['runtime']['gpu']!='Tesla T4':
        raise ValueError('T4 hardware required for cost consistency')
    for k in ['torch','transformers','accelerate','bitsandbytes','tokenizers']:
        if manifest['runtime']['packages'][k]!=expected['runtime']['packages'][k]:
            raise ValueError(f'Inference runtime changed: {k}')
    stages={s:load_records(root,s,queries) for s in ['small','strong','ic']}
    for rows in stages.values():
        if any(r['config_hash']!=fingerprint(manifest) for r in rows): raise ValueError('Mixed caches')
    small,strong,ic=[stages[s] for s in ['small','strong','ic']]
    features=np.array([r['features'] for r in small],dtype=np.float64)
    if features.shape!=(512,1536) or not np.isfinite(features).all(): raise ValueError('Invalid probe features')
    score_path=root/'test_scores.json'
    if score_path.exists():
        saved=read_json(score_path)
        assert saved['probe_hash']==methods['probe_hash'] and saved['config_hash']==fingerprint(manifest)
        assert saved['ids']==[q['id'] for q in queries]
        cheap=np.array(saved['cheap']);probe_seconds=np.array(saved['probe_seconds'])
    else:
        probe=portable_probe(methods['probe']);probe.predict_proba(features[:1]);cheap=[];probe_seconds=[]
        for x in features:
            start=time.perf_counter();cheap.append(float(probe.predict_proba(x[None,:])[0,1]));probe_seconds.append(time.perf_counter()-start)
        atomic_json(score_path,dict(probe_hash=methods['probe_hash'],config_hash=fingerprint(manifest),
            ids=[q['id'] for q in queries],cheap=cheap,probe_seconds=probe_seconds))
        cheap=np.array(cheap);probe_seconds=np.array(probe_seconds)
    d=dict(ids=np.array([q['id'] for q in queries]),cheap=cheap,ic=np.array([r['ic'] for r in ic]),
        small_correct=np.array([r['correct'] for r in small]),strong_correct=np.array([r['correct'] for r in strong]),
        small_seconds=np.array([r['elapsed_seconds'] for r in small]),strong_seconds=np.array([r['elapsed_seconds'] for r in strong]),
        ic_seconds=np.array([r['elapsed_seconds'] for r in ic]),probe_seconds=probe_seconds)
    atomic_json(root/'per_query.json',[dict(query=q,small_prediction=s['prediction'],strong_prediction=t['prediction'],
        query_hash=fingerprint(q),**{k:float(v[i]) for k,v in d.items() if k!='ids'})
        for i,(q,s,t) in enumerate(zip(queries,small,strong))])
    if any(not np.isfinite(d[k]).all() or (d[k]<=0).any() for k in ['small_seconds','strong_seconds','ic_seconds','probe_seconds']):
        raise ValueError('Invalid measured costs')
    rng=np.random.default_rng(PROTOCOL['seed']);indices=rng.integers(0,512,size=(PROTOCOL['bootstrap_repeats'],512))
    unique={fingerprint(p['cheap-only']):p['cheap-only'] for p in methods['selected']}
    endpoints=sorted(unique.values(),key=lambda p:p['cheap_threshold'])
    if len(endpoints)!=2: raise ValueError('Precommitted cheap endpoints changed')
    low,high=[replay(Policy(**p),d) for p in endpoints]
    results=[];comparisons=[];base={k:replay(Policy(k),d) for k in ['small-only','strong-only']}
    for k,r in base.items():
        results.append(dict(id=k,method=k,penalty=None,**extended_summary(r,d)))
    for i,point in enumerate(methods['selected']):
        actual={k:replay(Policy(**point[k]),d) for k in ['cheap-only','always-ic','adaptive']}
        adaptive=actual['adaptive'];budget=float(np.dot(adaptive['acquire'],d['ic_seconds']))
        random_runs=[random_replay(d,p['cheap_threshold'],p['ic_threshold'],budget,p['seed']) for p in point['random']]
        assert all(np.isclose(np.dot(r['acquire'],d['ic_seconds']),budget) for r in random_runs)
        random_mean={k:np.mean([r[k] for r in random_runs],axis=0) for k in random_runs[0]}
        mixture,match=matched_budget(adaptive,low,high)
        actual.update({'random-ic-budget-matched':random_mean,'cheap-cost-matched':mixture})
        random_times=[];random_weights=[]
        for p,rr in zip(point['random'],random_runs):
            base_times=replay(Policy('cheap-only',p['cheap_threshold']),d)['seconds']
            ic_times=d['small_seconds']+d['probe_seconds']+d['ic_seconds']+(d['ic']<p['ic_threshold'])*d['strong_seconds']
            random_times.extend([*base_times,*ic_times]);random_weights.extend([*(1-rr['acquire']),*rr['acquire']])
        mix_latency=(np.r_[low['seconds'],high['seconds']],np.r_[np.full(512,1-match['high_probability']),np.full(512,match['high_probability'])])
        for kind,r in actual.items():
            latency=mix_latency if kind=='cheap-cost-matched' else (random_times,random_weights) if kind=='random-ic-budget-matched' else None
            results.append(dict(id=f'p{i}:{kind}',method=kind,penalty=point['penalty'],**extended_summary(r,d,latency)))
        primary=dict(gain=float(np.mean(adaptive['correct']-mixture['correct'])),**match,**matched_bootstrap(adaptive,low,high,indices))
        primary.update(statistically_positive=bool(primary['familywise_95'][0]>0),
            practically_meaningful_estimate=bool(primary['gain']>=PROTOCOL['practical_gain']),
            lower_bound_exceeds_practical_threshold=bool(primary['familywise_95'][0]>=PROTOCOL['practical_gain']))
        initial=d['cheap']<point['adaptive']['cheap_threshold'];route=adaptive['strong'].astype(bool)
        delta=(route.astype(int)-initial.astype(int))*(d['strong_correct']-d['small_correct'])
        assert not np.any((route!=initial)&~adaptive['acquire'].astype(bool))
        changes=dict(acquired=int(adaptive['acquire'].sum()),changed=int(np.sum(route!=initial)),
            beneficial=int(np.sum(delta>0)),harmful=int(np.sum(delta<0)),unchanged_correctness=int(np.sum((route!=initial)&(delta==0))),
            added_escalations=int(np.sum(route&~initial)),avoided_escalations=int(np.sum(initial&~route)),
            beneficial_added=int(np.sum((delta>0)&route)),beneficial_avoided=int(np.sum((delta>0)&~route)),
            harmful_added=int(np.sum((delta<0)&route)),harmful_avoided=int(np.sum((delta<0)&~route)))
        secondary={}
        for kind,r in {**base,**{k:v for k,v in actual.items() if k not in ['adaptive','cheap-cost-matched']}}.items():
            diff=adaptive['correct']-r['correct'];secondary[kind]=dict(gain=float(diff.mean()),**intervals(diff[indices].mean(axis=1)))
        comparisons.append(dict(point=i,penalty=point['penalty'],primary=primary,routing_changes=changes,
            ic_budget_seconds=budget,random_accuracy_seed_std=float(np.std([r['correct'].mean() for r in random_runs])),secondary=secondary))
        serial=lambda r,j:{k:float(v[j]) for k,v in r.items()}
        atomic_json(root/'decisions'/f'point-{i}.json',dict(penalty=point['penalty'],policies={k:point[k] for k in ['cheap-only','always-ic','adaptive']},
            mixture_endpoints=endpoints,mixture=match,random_policies=point['random'],rows=[dict(id=q['id'],
                initial_strong=bool(initial[j]),ic_changed_routing=bool(route[j]!=initial[j]),ic_correctness_delta=int(delta[j]),
                methods={kind:serial(r,j) for kind,r in {**base,**actual}.items()},
                random_seeds=[serial(r,j) for r in random_runs]) for j,q in enumerate(queries)]))
    eligible=[c for c in comparisons if c['primary']['exact_cost_match'] and c['routing_changes']['acquired']>0]
    meaningful=any(c['primary']['gain']>=PROTOCOL['practical_gain'] and c['primary']['familywise_95'][0]>0 for c in eligible)
    ruled_out=bool(eligible) and all(c['primary']['familywise_95'][1]<PROTOCOL['practical_gain'] for c in eligible)
    report=dict(scope='Confirmatory evaluation of pre-test frozen policies and analysis addendum; no tuning on final test',
        n_test=512,methods_hash=fingerprint(methods),protocol_hash=fingerprint(PROTOCOL),inference_config_hash=fingerprint(manifest),
        results=results,comparisons=comparisons,observed_frontier=observed_frontier(results),
        meaningful_support=bool(meaningful),three_point_gain_ruled_out_for_tested_eligible_points=bool(ruled_out),
        cost_definition=PROTOCOL['costs'],random_control=methods['random_control'],
        mean_component_seconds={k:float(d[k].mean()) for k in ['small_seconds','strong_seconds','ic_seconds','probe_seconds']},
        offline_collection={s:dict(model_calls=sum(r['model_calls'] for r in rows),seconds=sum(r['elapsed_seconds'] for r in rows),
            output_tokens=sum(r['output_tokens'] for r in rows),token_limit_hits=sum(r.get('hit_token_limit',False) for r in rows)) for s,rows in stages.items()},
        overhead=[read_json(p) for p in sorted((root/'overhead').glob('*.json'))],
        runtime=manifest['runtime'],limitations=['Single-pass warm component timings; additive offline replay rather than live two-GPU latency',
            'Retrospective cost matching and random IC schedule use measured test timings, not test labels; no claim of an online budget controller',
            'Bootstrap treats distinct questions as independent; secondary random intervals conditional on saved seed schedules',
            'Findings apply to this quantized model pair, zero-shot prompt, decoder and eight frozen operating points'])
    atomic_json(root/'confirmatory_results.json',report)
    plot(root,report)
    return report


def plot(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    names=['small-only','strong-only','cheap-only','always-ic','adaptive','random-ic-budget-matched','cheap-cost-matched']
    for name in names:
        points={(r['mean_seconds'],100*r['accuracy']) for r in report['results'] if r['method']==name}
        axes[0].scatter([x[0] for x in points],[x[1] for x in points],label=name,s=38)
    axes[0].set(xlabel='Mean warm service seconds/query',ylabel='Exact match (%)',title='All frozen operating points; final test n=512')
    axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
    for c in report['comparisons']:
        p=c['primary'];gain=100*p['gain'];lo,hi=np.array(p['familywise_95'])*100
        axes[1].vlines(c['point'],lo,hi,color='#a92c35');axes[1].scatter(c['point'],gain,color='#a92c35',s=25)
    axes[1].axhline(0,color='#666',lw=1);axes[1].axhline(3,color='#53874e',ls='--',label='Pre-test practical threshold: 3 points')
    axes[1].set(xlabel='Fixed operating point index',ylabel='Adaptive minus matched-budget cheap (points)',title='Paired bootstrap, familywise 95% intervals')
    axes[1].set_xticks(range(8));axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(Path(root)/'confirmatory_accuracy_cost.png',dpi=180);plt.close(fig)


def run(root,methods_path):
    from .collect import collect
    if read_json(Path(methods_path).parent/'confirmatory_protocol.json')!=PROTOCOL:
        raise ValueError('Committed pre-test protocol differs from implementation')
    queries=prepare_test(root,methods_path);methods=read_json(methods_path);m=methods['inference_manifest']
    collect(root,stages=['small','ic','strong'],splits=['test'],small_model=m['model_roles']['small'],strong_model=m['model_roles']['strong'],
        quantization=m['quantization'],max_new_tokens=m['max_new_tokens'],answer_stop=m['answer_stop'])
    report=evaluate(root)
    print('CONFIRMATORY COMPLETE:',dict(n_test=len(queries),meaningful_support=report['meaningful_support'],
        mean_component_seconds=report['mean_component_seconds'],frontier=report['observed_frontier']),flush=True)
    return report
