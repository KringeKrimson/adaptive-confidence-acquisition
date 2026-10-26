"""One bounded benefit-aware extension. Original inference/policies are untouched."""
from pathlib import Path
import time
import numpy as np
from .storage import atomic_json, read_json, fingerprint, load_records, bind_manifest
from .routing import Policy, replay, random_replay

PROTOCOL = dict(
    version=1, name='final_benefit_aware_extension', seed=4904,
    model_roles=dict(small='Qwen/Qwen2.5-1.5B-Instruct', strong='Qwen/Qwen2.5-14B-Instruct'),
    frozen_original_methods='59ef6e267603260190ade5468e7e54d248c25ba11ababec32cdb276d220fa8ed',
    data='Reuse original Qwen1024 train/128 calibration/128 assessment. Reserve fresh512 official-validation questions, seed4904, excluding every earlier project manifest by ID and normalized question. Original512 test is excluded and never used for extension fitting/selection/evaluation.',
    target='Multioutput linear ridge predicts [strong_correct-small_correct, small_QA_seconds, strong_QA_seconds]. No correctness-only target and no new neural feature encoder.',
    query_features='Train-only word TFIDF, ngrams1-2, min_df2, max_features8000, sublinear_tf=True, L2 normalization; inference on raw question only before either model runs.',
    hidden_features='Original1536d ordinary-QA prompt hidden state captured at prefill, with train-only StandardScaler. Evaluate the decision after complete small QA and charge full QA/extraction cost. No generated-answer hidden state.',
    alphas=[1.,10.,100.,1000.], penalties=[0.,.025,.05,.1,.2,.4,.8,1.6],
    selection='Select one alpha per feature family by calibration benefit MSE, tie larger alpha. For each fixed penalty, score predicted_benefit-penalty*predicted_incremental_cost/training_mean_strong_seconds. Incremental cost is predicted strong-small for preselection and predicted strong for cascade. Choose a threshold from [-2,-1.95,...,2] plus -3,3 by calibration actual accuracy-penalty*actual_total_cost/calibration_mean_strong_seconds, then least cost, then larger threshold. Never optimize on assessment.',
    numerical='Clip benefit predictions to [-1,1] and cost predictions to >=.001 seconds; no log-cost retransformation. Fit all three output columns with the same regularization selected using benefit MSE.',
    measurements='CPU perf_counter around per-query TFIDF+ridge or array conversion+scaler+ridge, first warmup excluded, all required inference overhead charged. Cached small/strong/IC times retain their original pinned NF4 FP32 settings. Missing strong training QA only is collected; loads/warmups separately recorded.',
    development_gate='Any of16 fixed new points must have an exact random-preselection cost match, nontrivial choice (>=8 queries each branch), calibration gain>=.03, assessment gain>=.03 and Bonferroni16 familywise95 paired lower>0. Otherwise STOP: no fresh-test inference, no further predictors/architectures/model-pair search. Previously inspected assessment is development evidence, not a new confirmatory test.',
    primary='On fresh test, compare each of16 fixed benefit-aware points to cost-only query-independent randomized small/strong preselection at equal expected measured total cost. Outside endpoint cost range use clipped endpoint and report unmatched cost explicitly; only exact matches support primary gain claims.',
    bootstrap_repeats=10000, family_size=16, practical_gain=.03,
    uncertainty='Paired query bootstrap seed4904,10000 resamples. Recompute cost-only random probability each resample; pointwise95 and Bonferroni16 familywise95. Integrate preselection coin analytically. For random-IC secondary comparisons condition on saved20seed expected schedules.',
    final_decision='Report all16 points; meaningful support requires exact match, estimated gain>=.03 and familywise lower>0. Also report lower>3points and whether all eligible upper bounds exclude3points. Never select/retune a point using final test.',
    baselines='Original frozen small-only,strong-only,eight cheap-only/always-IC/adaptive points and their20seed random-IC controls; randomized model preselection, matched-rate random cascading, and eight newly calibrated correctness-cascade controls using the original probe with threshold grid [-.01,0,.0125,...,1,1.01] and the same utility/tie rule. Original archived thresholds remain unchanged; this finer-grid control is explicitly separate.',
    test_freeze='If development PASS, commit portable predictors, all16 thresholds, hashes and this protocol before fresh-test inference; if FAIL keep reserved test untouched. No refitting on calibration/assessment after selection.',
    scope='A controlled comparison of direct incremental-benefit prediction and first-model overhead, not a claim that benefit-aware routing or linear quality estimation is novel.')


def predict_one(model, item):
    start=time.perf_counter()
    value=model.predict([item] if isinstance(item,str) else np.asarray(item,dtype=float)[None,:])[0]
    seconds=time.perf_counter()-start
    return np.r_[np.clip(value[0],-1,1),np.maximum(value[1:],.001)],seconds


def fit_candidates(train,calibration,kind):
    if any(r['partition']!='train' for r in train) or any(r['partition']!='calibration' for r in calibration):
        raise ValueError('Fit on training and select on calibration only')
    if {r['query']['id'] for r in train}&{r['query']['id'] for r in calibration}:
        raise ValueError('Training/calibration overlap')
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    x=lambda rows:[r['query']['question'] if kind=='preselection' else r['features'] for r in rows]
    y=lambda rows:np.array([[r['strong_correct']-r['small_correct'],r['small_seconds'],r['strong_seconds']] for r in rows])
    target=y(train);losses=[];candidates=[]
    for alpha in PROTOCOL['alphas']:
        transformer=(TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=8000,sublinear_tf=True)
                     if kind=='preselection' else StandardScaler())
        model=make_pipeline(transformer,Ridge(alpha=alpha,solver='lsqr',tol=1e-6))
        model.fit(x(train),target)
        prediction=np.clip(model.predict(x(calibration))[:,0],-1,1)
        loss=float(np.mean((prediction-y(calibration)[:,0])**2))
        candidates.append((loss,-alpha,model));losses.append(dict(alpha=alpha,benefit_mse=loss))
    loss,negative_alpha,model=min(candidates,key=lambda item:item[:2])
    return model,dict(alpha=-negative_alpha,calibration_benefit_mse=loss,candidates=losses)


def portable(model,kind):
    feature,ridge=model.steps[0][1],model.steps[1][1]
    common=dict(kind=kind,alpha=ridge.alpha,coef=ridge.coef_.tolist(),intercept=ridge.intercept_.tolist())
    if kind=='preselection':
        common.update(vocabulary={k:int(v) for k,v in feature.vocabulary_.items()},idf=feature.idf_.tolist())
    else:
        common.update(mean=feature.mean_.tolist(),scale=feature.scale_.tolist())
    return common


def restore(saved):
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    if saved['kind']=='preselection':
        feature=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=8000,sublinear_tf=True,vocabulary=saved['vocabulary'])
        feature.idf_=np.array(saved['idf'])
    else:
        feature=StandardScaler();feature.mean_=np.array(saved['mean']);feature.scale_=np.array(saved['scale'])
        feature.var_=feature.scale_**2;feature.n_features_in_=len(feature.mean_);feature.n_samples_seen_=1024
    ridge=Ridge(alpha=saved['alpha'],solver='lsqr',tol=1e-6)
    ridge.coef_=np.array(saved['coef']);ridge.intercept_=np.array(saved['intercept']);ridge.n_features_in_=ridge.coef_.shape[1]
    return make_pipeline(feature,ridge)


def route_predictions(kind,penalty,threshold,prediction,reference):
    incremental=prediction[:,2]-(prediction[:,1] if kind=='preselection' else 0)
    return prediction[:,0]-penalty*incremental/reference > threshold


def route_result(kind,strong,rows,overhead):
    s,t,cs,ct=[np.array([r[k] for r in rows]) for k in ['small_correct','strong_correct','small_seconds','strong_seconds']]
    cost=(np.where(strong,ct,cs) if kind=='preselection' else cs+strong*ct)+overhead
    return dict(correct=np.where(strong,t,s).astype(float),seconds=cost,strong=strong.astype(float))


def random_match(policy,rows,indices=None):
    s,t,cs,ct=[np.array([r[k] for r in rows]) for k in ['small_correct','strong_correct','small_seconds','strong_seconds']]
    mean=(lambda v:float(v.mean())) if indices is None else (lambda v:v[indices].mean(axis=1))
    lo,hi,target=mean(cs),mean(ct),mean(policy['seconds'])
    if np.any(hi<=lo):raise ValueError('Strong-only expected cost must exceed small-only')
    probability=np.clip((target-lo)/(hi-lo),0,1)
    accuracy=(1-probability)*mean(s)+probability*mean(t)
    cost=(1-probability)*lo+probability*hi
    if indices is not None:return mean(policy['correct'])-accuracy
    return dict(strong_probability=float(probability),accuracy=float(accuracy),mean_seconds=float(cost),
        exact_cost_match=bool(np.isclose(target,cost,atol=1e-10,rtol=0)),gain=float(mean(policy['correct'])-accuracy))


def interval(samples):
    tail=.05/(2*PROTOCOL['family_size'])
    return dict(pointwise_95=np.quantile(samples,[.025,.975]).tolist(),
                familywise_95=np.quantile(samples,[tail,1-tail]).tolist())


def score_rows(model,rows,kind):
    items=[r['query']['question'] if kind=='preselection' else r['features'] for r in rows]
    predict_one(model,items[0]);values=[];seconds=[]
    for x in items:
        p,c=predict_one(model,x);values.append(p);seconds.append(c)
    return np.array(values),np.array(seconds)


def read_development(root,original_methods):
    root=Path(root);queries=read_json(root/'queries.json')['queries']
    old=read_json(original_methods)
    assert fingerprint(old)==PROTOCOL['frozen_original_methods']
    partitions={qid:part for part,values in [('train',old['probe']['training_ids']),
        ('calibration',old['selection_ids']),('assessment',old['assessment_ids_excluded_from_selection'])] for qid in values}
    stages={s:load_records(root,s,queries) for s in ['small','strong']}
    manifest=read_json(root/'inference_manifest.json')
    for key in ['model_revisions','quantization','source_hash','max_new_tokens','answer_stop','feature_definition']:
        assert manifest[key]==old['inference_manifest'][key],key
    rows=[]
    for q,s,t in zip(queries,stages['small'],stages['strong']):
        assert q['id'] in partitions and s['config_hash']==t['config_hash']==fingerprint(manifest)
        rows.append(dict(query=q,partition=partitions[q['id']],features=s['features'],
            small_correct=s['correct'],strong_correct=t['correct'],small_seconds=s['elapsed_seconds'],strong_seconds=t['elapsed_seconds']))
    assert [sum(r['partition']==p for r in rows) for p in ['train','calibration','assessment']]==[1024,128,128]
    return rows


def baseline_data(root,rows,original):
    from .confirmatory import portable_probe
    probe=portable_probe(original['probe']);probe.predict_proba(np.asarray(rows[0]['features'])[None,:])
    scores=[];seconds=[]
    for row in rows:
        start=time.perf_counter();scores.append(float(probe.predict_proba(np.asarray(row['features'])[None,:])[0,1]))
        seconds.append(time.perf_counter()-start)
    ic=load_records(root,'ic',[r['query'] for r in rows])
    return dict(ids=np.array([r['query']['id'] for r in rows]),cheap=np.array(scores),probe_seconds=np.array(seconds),
        ic=np.array([r['ic'] for r in ic]),ic_seconds=np.array([r['elapsed_seconds'] for r in ic]),
        **{k:np.array([r[k] for r in rows]) for k in ['small_correct','strong_correct','small_seconds','strong_seconds']})


def baseline_comparisons(data,original,retuned):
    summary=lambda r:dict(accuracy=float(r['correct'].mean()),mean_seconds=float(r['seconds'].mean()),
        strong_rate=float(r['strong'].mean()),ic_rate=float(r.get('acquire',np.zeros(len(r['correct']))).mean()))
    result=[dict(method=kind,penalty=None,**summary(replay(Policy(kind),data))) for kind in ['small-only','strong-only']]
    for point in original['selected']:
        adaptive=replay(Policy(**point['adaptive']),data);budget=float(adaptive['acquire']@data['ic_seconds'])
        for kind in ['cheap-only','adaptive','always-ic']:
            result.append(dict(method='original-'+kind,penalty=point['penalty'],**summary(replay(Policy(**point[kind]),data))))
        runs=[random_replay(data,p['cheap_threshold'],p['ic_threshold'],budget,p['seed']) for p in point['random']]
        average={k:np.mean([r[k] for r in runs],axis=0) for k in runs[0]}
        result.append(dict(method='original-random-ic',penalty=point['penalty'],**summary(average)))
    for point in retuned:
        r=replay(Policy('cheap-only',cheap_threshold=point['threshold']),data)
        result.append(dict(method='same-grid-correctness',penalty=point['penalty'],threshold=point['threshold'],**summary(r)))
    return result


def fit_development(root,out,original_methods):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    bind_manifest(out,'protocol.json',PROTOCOL)
    rows=read_development(root,original_methods)
    train,cal,assess=[[r for r in rows if r['partition']==part] for part in ['train','calibration','assessment']]
    original=read_json(original_methods)
    if (out/'development.json').exists():
        report=read_json(out/'development.json')
        assert report['protocol_hash']==fingerprint(PROTOCOL)
        assert report['methods_hash']==fingerprint(read_json(out/'methods.json'))
        return report
    if (out/'methods.json').exists():
        frozen=read_json(out/'methods.json');assert frozen['protocol_hash']==fingerprint(PROTOCOL)
        reference=frozen['training_reference_seconds'];selection=frozen['policies'];models=frozen['models']
        retuned=frozen['same_grid_correctness'];calibration=read_json(out/'calibration_scores.json')
    else:
        reference=float(np.mean([r['strong_seconds'] for r in train]));selection=[];models={};fit_info={};calibration={}
        cald=baseline_data(root,cal,original);retuned=[]
        for penalty in PROTOCOL['penalties']:
            choices=[]
            for threshold in np.r_[-.01,np.linspace(0,1,81),1.01]:
                r=replay(Policy('cheap-only',cheap_threshold=float(threshold)),cald)
                value=r['correct'].mean()-penalty*r['seconds'].mean()/cald['strong_seconds'].mean()
                choices.append((value,-r['seconds'].mean(),float(threshold)))
            retuned.append(dict(penalty=penalty,threshold=max(choices)[2]))
        for kind in ['preselection','cascade']:
            model,info=fit_candidates(train,cal,kind);models[kind]=portable(model,kind);fit_info[kind]=info
            prediction,overhead=score_rows(model,cal,kind)
            restored,other=score_rows(restore(models[kind]),cal,kind)
            assert np.allclose(prediction,restored,atol=1e-10,rtol=0)
            calibration[kind]=dict(prediction=prediction.tolist(),seconds=overhead.tolist())
            for penalty in PROTOCOL['penalties']:
                candidates=[]
                for threshold in np.r_[-3.,np.linspace(-2,2,81),3.]:
                    route=route_predictions(kind,penalty,threshold,prediction,reference)
                    result=route_result(kind,route,cal,overhead)
                    utility=result['correct'].mean()-penalty*result['seconds'].mean()/np.mean([r['strong_seconds'] for r in cal])
                    candidates.append((utility,-result['seconds'].mean(),float(threshold),result))
                _,_,threshold,result=max(candidates,key=lambda r:r[:3])
                selection.append(dict(kind=kind,penalty=penalty,threshold=threshold,
                    calibration=dict(accuracy=float(result['correct'].mean()),mean_seconds=float(result['seconds'].mean()),
                        random_preselection=random_match(result,cal))))
        frozen=dict(protocol_hash=fingerprint(PROTOCOL),training_ids=[r['query']['id'] for r in train],
            selection_ids=[r['query']['id'] for r in cal],assessment_ids_excluded_from_selection=[r['query']['id'] for r in assess],
            models=models,fit_info=fit_info,training_reference_seconds=reference,policies=selection,
            original_methods_hash=PROTOCOL['frozen_original_methods'],same_grid_correctness=retuned)
        atomic_json(out/'calibration_scores.json',calibration)
    # This artifact exists and is printed BEFORE assessment predictions/evaluation.
    bind_manifest(out,'methods.json',frozen)
    print('BENEFIT METHODS FROZEN:',fingerprint(frozen),'protocol:',fingerprint(PROTOCOL),flush=True)
    if (out/'assessment_scores.json').exists():
        scored=read_json(out/'assessment_scores.json')
        predictions={k:np.array(v['prediction']) for k,v in scored.items()}
        overheads={k:np.array(v['seconds']) for k,v in scored.items()}
    else:
        predictions={};overheads={}
        for kind in models:
            predictions[kind],overheads[kind]=score_rows(restore(models[kind]),assess,kind)
        atomic_json(out/'assessment_scores.json',{k:dict(prediction=predictions[k].tolist(),seconds=overheads[k].tolist()) for k in models})
    indices=np.random.default_rng(PROTOCOL['seed']).integers(0,len(assess),size=(PROTOCOL['bootstrap_repeats'],len(assess)))
    points=[];decisions=[]
    for policy in selection:
        kind=policy['kind'];route=route_predictions(kind,policy['penalty'],policy['threshold'],predictions[kind],reference)
        result=route_result(kind,route,assess,overheads[kind]);match=random_match(result,assess)
        ci=interval(random_match(result,assess,indices))
        eligible=match['exact_cost_match'] and 8<=int(route.sum())<=len(route)-8 and policy['calibration']['random_preselection']['gain']>=.03
        passed=bool(eligible and match['gain']>=.03 and ci['familywise_95'][0]>0)
        points.append(dict(**policy,assessment=dict(accuracy=float(result['correct'].mean()),mean_seconds=float(result['seconds'].mean()),
            strong_rate=float(route.mean()),mean_router_seconds=float(overheads[kind].mean()),random_preselection=match,**ci),
            eligible=bool(eligible),passed=passed))
        decisions.append(dict(kind=kind,penalty=policy['penalty'],rows=[dict(id=r['query']['id'],strong=bool(route[j]),
            correct=float(result['correct'][j]),seconds=float(result['seconds'][j])) for j,r in enumerate(assess)]))
    assessmentd=baseline_data(root,assess,original)
    baseline=baseline_comparisons(assessmentd,original,retuned)
    random_curve=[dict(strong_probability=float(p),accuracy=float(((1-p)*assessmentd['small_correct']+p*assessmentd['strong_correct']).mean()),
        mean_seconds=float(((1-p)*assessmentd['small_seconds']+p*assessmentd['strong_seconds']).mean())) for p in np.linspace(0,1,101)]
    random_cascade=[]
    for point in points:
        rate=point['assessment']['strong_rate']
        random_cascade.append(dict(kind=point['kind'],penalty=point['penalty'],strong_probability=rate,
            accuracy=float(((1-rate)*assessmentd['small_correct']+rate*assessmentd['strong_correct']).mean()),
            mean_seconds=float((assessmentd['small_seconds']+rate*assessmentd['strong_seconds']).mean())))
    report=dict(scope='Development screen of a precommitted bounded extension; assessment is previously inspected development, not fresh confirmation',
        protocol_hash=fingerprint(PROTOCOL),methods_hash=fingerprint(frozen),status='PASS' if any(p['passed'] for p in points) else 'STOP',
        n_train=len(train),n_calibration=len(cal),n_assessment=len(assess),points=points,
        fresh_test_inference_authorized=bool(any(p['passed'] for p in points)),test_predictions_read=0,
        measured_router_seconds={kind:overheads[kind].tolist() for kind in models},
        assessment_predictions={kind:predictions[kind].tolist() for kind in models},baselines=baseline,
        random_preselection_curve=random_curve,matched_rate_random_cascades=random_cascade,
        overhead=[read_json(p) for p in sorted((Path(root)/'overhead').glob('*.json'))])
    atomic_json(out/'development.json',report);atomic_json(out/'assessment_decisions.json',decisions)
    atomic_json(out/'calibration_scores.json',calibration)
    print('BENEFIT DEVELOPMENT:',report['status'],flush=True)
    return report
