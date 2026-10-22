import csv
import importlib.metadata
import platform
import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from .data import validate_splits
from .routing import Policy, random_budget_probabilities, random_replay, replay, summarize, tune_policies
from .storage import atomic_json, fingerprint, load_records, read_json


def fit_probe(x, labels, splits, c=1.0):
    if set(splits) != {'train'}:
        raise ValueError('Probe training may use training data only')
    if len(np.unique(labels)) != 2:
        raise ValueError('Probe needs correct and incorrect training answers; increase training sample')
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=4901)).fit(x, labels)


def confidence_metrics(y, p):
    return dict(auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p, labels=[0,1])))


def bootstrap_delta(a, b, seed=4901, repeats=1000):
    rng = np.random.default_rng(seed)
    delta = a-b
    values = [float(np.mean(delta[rng.integers(0, len(delta), len(delta))])) for _ in range(repeats)]
    return [float(x) for x in np.quantile(values, [.025, .975])]


def analyze(root, penalties=(0., .025, .05, .1, .2, .4, .8, 1.6), random_seeds=20):
    root = Path(root)
    if random_seeds < 1:
        raise ValueError('At least one random seed is required')
    queries = read_json(root/'queries.json')['queries']
    validate_splits(queries)
    split = np.array([q['split'] for q in queries])
    if any(not np.any(split == part) for part in ['train','validation','test']):
        raise ValueError('Analysis requires separate nonempty train, validation, test sets')
    small, strong, ic = [load_records(root, s, queries) for s in ['small','strong','ic']]
    manifest_hash = fingerprint(read_json(root/'inference_manifest.json'))
    if any(r['config_hash'] != manifest_hash for stage in [small,strong,ic] for r in stage):
        raise ValueError('Mixed inference configurations in cache')
    x = np.asarray([r['features'] for r in small], dtype=np.float64)
    y = np.array([r['correct'] for r in small])
    train, val, test = [split == s for s in ['train','validation','test']]
    start = time.perf_counter()
    candidates = []
    for c in [.01, .1, 1.0, 10.0]:
        probe = fit_probe(x[train], y[train], split[train], c)
        loss = log_loss(y[val], probe.predict_proba(x[val])[:,1], labels=[0,1])
        candidates.append((float(loss), c, probe))
    _, c, probe = min(candidates, key=lambda p: (p[0], p[1]))
    training_seconds = time.perf_counter()-start
    probabilities, scoring_times = [], []
    # Batch one CPU timing, matching the decision service. Includes feature scaling.
    probe.predict_proba(x[:1])
    for feature in x:
        start = time.perf_counter()
        probabilities.append(float(probe.predict_proba(feature[None,:])[0,1]))
        scoring_times.append(time.perf_counter()-start)
    d = dict(ids=np.array([q['id'] for q in queries]), splits=split, cheap=np.array(probabilities),
             ic=np.array([r['ic'] for r in ic]), small_correct=y,
             strong_correct=np.array([r['correct'] for r in strong]),
             small_seconds=np.array([r['elapsed_seconds'] for r in small]),
             strong_seconds=np.array([r['elapsed_seconds'] for r in strong]),
             ic_seconds=np.array([r['elapsed_seconds'] for r in ic]), probe_seconds=np.array(scoring_times))
    v, t = [{k:a[mask] for k,a in d.items()} for mask in [val,test]]
    selected = tune_policies(v, penalties)
    atomic_json(root/'selected_policies.json', dict(selection_split='validation', policies=selected,
        probe_C=c, validation_losses=[dict(loss=l,C=c0) for l,c0,_ in candidates],
        train_ids=d['ids'][train].tolist(), validation_ids=v['ids'].tolist()))
    # Portable probe weights; avoid loading pickle from experiment archives.
    scaler, classifier = probe.steps[0][1], probe.steps[1][1]
    atomic_json(root/'probe.json', dict(C=c, mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
        coefficients=classifier.coef_.tolist(), intercept=classifier.intercept_.tolist(),
        training_seconds=training_seconds, training_query_hashes=[fingerprint(q) for q in queries if q['split']=='train']))
    atomic_json(root/'probe_scores.json',dict(
        probe_hash=fingerprint(read_json(root/'probe.json')),
        runtime=dict(machine=platform.machine(),processor=platform.processor(),
                     python=platform.python_version(),
                     sklearn=importlib.metadata.version('scikit-learn')),
        rows=[dict(id=q['id'],split=q['split'],query_hash=fingerprint(q),
                   probability=float(p),elapsed_seconds=float(sec))
              for q,p,sec in zip(queries,probabilities,scoring_times)]))
    results, details = [], []
    for kind in ['small-only','strong-only']:
        results.append(dict(method=kind, penalty=None, **summarize(replay(Policy(kind),t))))
    for point in selected:
        replays = {}
        for kind in ['cheap-only','always-ic','adaptive']:
            policy = Policy(**point[kind])
            r = replay(policy,t)
            replays[kind] = r
            results.append(dict(method=kind, penalty=point['penalty'], **summarize(r)))
        # Match realized IC acquisition budget, not total cascade budget.
        # Random routing thresholds are selected on validation for the same control.
        av = replay(Policy(**point['adaptive']),v)
        at = replays['adaptive']
        bv = float(np.dot(av['acquire'],v['ic_seconds']))
        bt = float(np.dot(at['acquire'],t['ic_seconds']))
        random_runs, random_policies = [], []
        for seed in range(random_seeds):
            pv = random_budget_probabilities(v['ids'],v['ic_seconds'],bv,seed)
            scored = []
            for cheap_t in [-.01,*np.linspace(0,1,11),1.01]:
                for ic_t in [-.01,*np.linspace(0,1,11),1.01]:
                    r = random_replay(v,cheap_t,ic_t,bv,seed,probabilities=pv)
                    m = summarize(r)
                    scored.append((m['accuracy']-point['penalty']*m['mean_seconds']/point['reference_seconds'],
                                   -m['mean_seconds'],float(cheap_t),float(ic_t)))
            _,_,cheap_t,ic_t = max(scored)
            random_policies.append(dict(seed=seed,cheap_threshold=cheap_t,ic_threshold=ic_t))
            rr = random_replay(t,cheap_t,ic_t,bt,seed)
            random_runs.append(rr)
            if not np.isclose(np.dot(rr['acquire'],t['ic_seconds']),bt):
                raise AssertionError('Random IC budget mismatch')
        random_mean = {k:np.mean([r[k] for r in random_runs],axis=0) for k in random_runs[0]}
        rm = summarize(random_mean)
        results.append(dict(method='random-ic-budget-matched',penalty=point['penalty'],**rm))
        details.append(dict(penalty=point['penalty'],ic_budget_seconds=bt, random_seeds=random_seeds,
            random_policies=random_policies,validation_ic_budget_seconds=bv,
            random_accuracy_seed_std=float(np.std([np.mean(r['correct']) for r in random_runs])),
            adaptive_minus_random_accuracy=float(np.mean(at['correct']-random_mean['correct'])),
            paired_bootstrap_95=bootstrap_delta(at['correct'],random_mean['correct']),
            rows=[dict(id=qid,cheap=float(p),ic=float(i),adaptive_strong=bool(s),acquire=bool(a),
                       adaptive_seconds=float(sec),random_expected_correct=float(rc))
                  for qid,p,i,s,a,sec,rc in zip(t['ids'],t['cheap'],t['ic'],at['strong'],at['acquire'],
                                               at['seconds'],random_mean['correct'])]))
    report = dict(results=results, comparisons=details,
        confidence={part:dict(cheap=confidence_metrics(y[mask],d['cheap'][mask]),
                              ic=confidence_metrics(y[mask],d['ic'][mask]))
                    for part,mask in [('validation',val),('test',test)]},
        cost_definition='Warm batch-one additive replay service seconds; includes QA, probe, acquired IC and routed strong QA. Model loads, warmups, training and offline cache collection are separate.',
        random_control='Offline timing-oracle random prefix with fractional boundary query: equal expected measured IC seconds, not equal total routing cost.',
        n_train=int(train.sum()),n_validation=int(val.sum()),n_test=int(test.sum()))
    atomic_json(root/'analysis.json',report)
    with open(root/'accuracy_cost.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    plot_results(root,results)
    return report


def plot_results(root, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,4.5))
    for method in dict.fromkeys(r['method'] for r in results):
        rows = sorted([r for r in results if r['method']==method],key=lambda r:r['mean_seconds'])
        ax.plot([r['mean_seconds'] for r in rows],[r['accuracy'] for r in rows],marker='o',label=method)
    ax.set(xlabel='Mean replay service seconds / query',ylabel='Test exact-match accuracy',
           title='Validation-selected routing policies')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(root)/'accuracy_cost.png',dpi=180)
    plt.close(fig)
