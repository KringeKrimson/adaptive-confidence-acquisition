"""Post-test, CPU-only model preselection benchmark; never edits frozen outputs."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from .storage import atomic_json, fingerprint, read_json


def preselect(small, strong, probability):
    """One Bernoulli choice BEFORE generation; the unchosen model costs nothing."""
    if not 0 <= probability <= 1:
        raise ValueError('Probability must be in [0,1]')
    return {k: (1-probability)*small[k]+probability*strong[k]
            for k in ['correct', 'seconds']}


def budget_probability(target, small_cost, strong_cost):
    """Cost-only interpolation, clipped to endpoints; never extrapolate."""
    if np.any(np.asarray(strong_cost) <= small_cost):
        raise ValueError('Expected strong cost must exceed small cost')
    return np.clip((target-small_cost)/(strong_cost-small_cost), 0, 1)


def comparison(small, strong, policy, boot_small, boot_strong, boot_policy):
    probability = float(budget_probability(policy['seconds'].mean(),
                                           small['seconds'].mean(), strong['seconds'].mean()))
    mixture = preselect(small, strong, probability)
    gain = float(np.mean(mixture['correct']-policy['correct']))
    saving = float(np.mean(policy['seconds']-mixture['seconds']))
    p = budget_probability(boot_policy['seconds'], boot_small['seconds'], boot_strong['seconds'])
    boot_accuracy = (1-p)*boot_small['correct']+p*boot_strong['correct']
    boot_cost = (1-p)*boot_small['seconds']+p*boot_strong['seconds']
    gain_ci = np.quantile(boot_accuracy-boot_policy['correct'], [.025, .975]).tolist()
    saving_ci = np.quantile(boot_policy['seconds']-boot_cost, [.025, .975]).tolist()
    return dict(strong_probability=probability, mixture_accuracy=float(mixture['correct'].mean()),
        mixture_seconds=float(mixture['seconds'].mean()), accuracy_gain=gain,
        cost_saving_seconds=saving, accuracy_gain_pointwise_95=gain_ci,
        cost_saving_pointwise_95=saving_ci, exact_cost_match=bool(abs(saving)<1e-10),
        observed_dominated=bool(gain>=-1e-12 and saving>=-1e-12 and
                                (gain>1e-12 or saving>1e-12))), mixture


def snapshot(cache, repo):
    paths = [*cache.rglob('*'), *repo.glob('results/*confirmatory*.json'),
             repo/'results/qwen15_methods.json']
    paths += [repo/'src/aca'/name for name in
              ['confirmatory.py', 'frozen.py', 'routing.py', 'models.py', 'ic.py', 'collect.py', 'metrics.py']]
    key = lambda p: 'cache/'+str(p.relative_to(cache)) if cache in p.parents else 'repository/'+str(p.relative_to(repo))
    return {key(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths)) if p.is_file()}


def analyze(cache, out, repeats=10000, seed=4903):
    cache, out = Path(cache).resolve(), Path(out).resolve()
    if out == cache or cache in out.parents:
        raise ValueError('Exploratory output must be outside the frozen cache')
    repo = Path(__file__).resolve().parents[2]
    before = snapshot(cache, repo)
    original = read_json(cache/'confirmatory_results.json')
    if original != read_json(repo/'results/qwen15_confirmatory.json'):
        raise ValueError('Cache and committed confirmatory report differ')
    rows = read_json(cache/'per_query.json')
    queries = read_json(cache/'queries.json')['queries']
    protocol = read_json(cache/'confirmatory_protocol.json')
    assert fingerprint(queries) == protocol['test_queries_hash']
    assert fingerprint(read_json(cache/'frozen_methods.json')) == protocol['methods_hash']
    assert [r['query'] for r in rows] == queries and len(rows) == 512
    d = {k: np.array([r[k] for r in rows]) for k in
         ['small_correct', 'strong_correct', 'small_seconds', 'strong_seconds', 'cheap', 'ic']}
    small = dict(correct=d['small_correct'], seconds=d['small_seconds'])
    strong = dict(correct=d['strong_correct'], seconds=d['strong_seconds'])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(repeats, len(rows)))
    boot = lambda r: {k: r[k][indices].mean(axis=1) for k in ['correct', 'seconds']}
    bs, bt = boot(small), boot(strong)
    saved = [read_json(cache/'decisions'/f'point-{i}.json') for i in range(8)]
    ids = [q['id'] for q in queries]
    for point in saved:
        assert [r['id'] for r in point['rows']] == ids
    low_point = min(range(8), key=lambda i: saved[i]['policies']['cheap-only']['cheap_threshold'])
    high_point = max(range(8), key=lambda i: saved[i]['policies']['cheap-only']['cheap_threshold'])
    def cached(i, method):
        return {k: np.array([r['methods'][method][k] for r in saved[i]['rows']])
                for k in ['correct', 'seconds']}
    bl, bh = boot(cached(low_point, 'cheap-only')), boot(cached(high_point, 'cheap-only'))
    bas = [boot(cached(i, 'adaptive')) for i in range(8)]
    comparisons, per_query = [], []
    for point in original['results']:
        name = point['id']
        if name in ['small-only', 'strong-only']:
            r = small if name == 'small-only' else strong
            br = bs if name == 'small-only' else bt
        else:
            prefix, method = name.split(':');i = int(prefix[1:])
            r = cached(i, method);br = boot(r)
            if method == 'cheap-cost-matched':
                # Recompute this original comparator's COST-only mixture in each
                # resample too; its two frozen thresholds are never changed.
                w = budget_probability(bas[i]['seconds'], bl['seconds'], bh['seconds'])
                br = {k: (1-w)*bl[k]+w*bh[k] for k in ['correct', 'seconds']}
        assert np.isclose(r['correct'].mean(), point['accuracy'], atol=1e-12, rtol=0)
        assert np.isclose(r['seconds'].mean(), point['mean_seconds'], atol=1e-12, rtol=0)
        result, mix = comparison(small, strong, r, bs, bt, br)
        comparisons.append(dict(id=name, method=point['method'], penalty=point['penalty'],
            policy_accuracy=point['accuracy'], policy_seconds=point['mean_seconds'], **result))
        per_query.append(dict(policy_id=name, strong_probability=result['strong_probability'],
            rows=[dict(id=qid, expected_accuracy=float(mix['correct'][j]),
                       expected_seconds=float(mix['seconds'][j])) for j, qid in enumerate(ids)]))
    curve = []
    for p in np.linspace(0, 1, 101):
        r = preselect(small, strong, p)
        curve.append(dict(strong_probability=float(p), accuracy=float(r['correct'].mean()),
            mean_seconds=float(r['seconds'].mean()),
            accuracy_pointwise_95=np.quantile((1-p)*bs['correct']+p*bt['correct'], [.025,.975]).tolist(),
            seconds_pointwise_95=np.quantile((1-p)*bs['seconds']+p*bt['seconds'], [.025,.975]).tolist(),
            finite_cohort_coin_accuracy_sd=float(np.sqrt(np.sum(p*(1-p)*(strong['correct']-small['correct'])**2))/len(rows))))
    from sklearn.metrics import roc_auc_score
    s, t = d['small_correct'], d['strong_correct']; errors = s == 0
    diagnostics = dict(small_correctness_probe_auroc=float(roc_auc_score(s, d['cheap'])),
        low_probe_rescue_auroc_among_small_errors=float(roc_auc_score(t[errors], -d['cheap'][errors])),
        ic_correctness_auroc=float(roc_auc_score(s, d['ic'])),
        low_ic_rescue_auroc_among_small_errors=float(roc_auc_score(t[errors], -d['ic'][errors])),
        small_errors=int(errors.sum()), strong_rescues=int(np.sum(errors & (t == 1))),
        strong_harms=int(np.sum((s == 1) & (t == 0))),
        global_net_benefit_per_strong_choice=float(np.mean(t-s)))
    for i in [0, 5]:
        chosen = np.array([r['methods']['adaptive']['strong'] for r in saved[i]['rows']], bool)
        diagnostics[f'adaptive_point_{i}'] = dict(escalations=int(chosen.sum()),
            net_benefit_per_escalation=float(np.mean((t-s)[chosen])),
            mean_strong_seconds_when_escalated=float(d['strong_seconds'][chosen].mean()))
    frozen_intermediates = [c for c in comparisons if c['id'] in original['observed_frontier'] and
                            c['id'] not in ['small-only', 'strong-only']]
    remaining = [c['id'] for c in comparisons if c['id'] not in ['small-only','strong-only'] and not c['observed_dominated']]
    report = dict(scope='EXPLORATORY, requested after inspecting final-test results; not a frozen confirmatory hypothesis test',
        n=512, new_model_calls=0, bootstrap=dict(seed=seed, repeats=repeats, intervals='Pointwise paired-query percentile95; no multiplicity correction or new confirmatory decision',
            cost_matching='Recompute preselection probability from paired resampled costs; also recompute original cheap-control mixture weights from costs only. Saved random-IC20-seed schedules remain conditional/fixed.',
            randomness='Integrate model-choice coin analytically. Intervals quantify question-sampling uncertainty in expected outcomes, not a realized randomized run.'),
        formula=dict(accuracy_intercept=float(s.mean()), accuracy_slope=float((t-s).mean()),
            seconds_intercept=float(small['seconds'].mean()), seconds_slope=float((strong['seconds']-small['seconds']).mean()),
            probability_range=[0,1], rule='Choose strong with query-independent probability p before generation; otherwise choose small. Exactly one model call; no small call, probe or IC on the strong branch.'),
        comparisons=comparisons, curve=curve, diagnostics=diagnostics,
        frontier=dict(original_finite_point_frontier=original['observed_frontier'],
            former_intermediate_points_dominated=all(c['observed_dominated'] for c in frozen_intermediates),
            expanded_observed_frontier=('Entire preselection line segment p in [0,1], including small-only and strong-only endpoints; all previously reported cascade/control points are weakly below/right of it.'
                if not remaining else 'Preselection line does not dominate every original point; remaining candidates require an envelope comparison.'),
            non_endpoint_points_not_dominated_by_preselection=remaining,
            dominated_non_endpoint_count=sum(c['observed_dominated'] for c in comparisons if c['id'] not in ['small-only','strong-only'])),
        limitations=['Single 512-question test cohort, inspected before this analysis; no new prospective claim or deployment policy selection',
            'Warm cached component times, not a live preselection benchmark; unmeasured dispatch/RNG, residency, loading, queuing and communication overhead excluded',
            'Small-only component time retains the original feature-capture overhead; no new faster small-only inference was assumed',
            'Expected randomized accuracy/cost differ from one realized schedule; original random-IC intervals conditional on saved schedules',
            'Pointwise intervals over many comparisons do not establish simultaneous statistical dominance',
            'Observed frontier relative to compared families; does not bound all possible query-dependent routers or generalize to other models/tasks'],
        confirmatory_preservation=dict(original_report_sha256=before['cache/confirmatory_results.json'],
            methods_hash=original['methods_hash'], protocol_hash=original['protocol_hash'],
            all_original_inputs_and_frozen_files_unchanged=True),
        input_fingerprints=dict(file_count=len(before), tree_fingerprint=fingerprint(before),
            key_files={k:v for k,v in before.items() if k.startswith('repository/') or
                       k in ['cache/per_query.json','cache/queries.json','cache/inference_manifest.json']}))
    assert snapshot(cache, repo) == before, 'Original frozen inputs changed'
    atomic_json(out/'exploratory_mixture.json', report)
    atomic_json(out/'expected_per_query.json', per_query)
    plot(out, report, original)
    assert snapshot(cache, repo) == before, 'Original frozen inputs changed'
    return report


def plot(out, report, original):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios':[1.25,1]})
    names = [('small-only','Small only'), ('strong-only','Strong only'),
             ('cheap-only','Frozen cheap'), ('always-ic','Always IC'), ('adaptive','Adaptive IC'),
             ('random-ic-budget-matched','Random IC'), ('cheap-cost-matched','Frozen cheap-route mixture')]
    for method, label in names:
        points = {(r['mean_seconds'],100*r['accuracy']) for r in original['results'] if r['method']==method}
        axes[0].scatter([p[0] for p in points], [p[1] for p in points], label=label, s=35, alpha=.85)
    c = report['curve']
    axes[0].plot([r['mean_seconds'] for r in c], [100*r['accuracy'] for r in c],
                 color='black', lw=2, label='Choose ONE model before inference', zorder=1)
    old = [r for r in original['results'] if r['id'] in original['observed_frontier'] and r['penalty'] is not None]
    axes[0].scatter([r['mean_seconds'] for r in old], [100*r['accuracy'] for r in old],
                    s=130, marker='D', facecolors='none', edgecolors='gray', label='Former intermediate frontier')
    axes[0].set(xlabel='Mean warm component seconds/query', ylabel='Expected exact match (%)',
                title='Expanded comparison; all frozen points retained')
    axes[0].legend(fontsize=7, loc='lower right');axes[0].grid(alpha=.2)
    selected = [('p0:adaptive','Adaptive IC, points 0–4'), ('p0:cheap-only','Frozen cheap, points 0–4'),
                ('p5:adaptive','Adaptive IC, point 5'), ('p5:cheap-cost-matched','Cheap-route mixture, point 5'),
                ('p5:random-ic-budget-matched','Random IC, point 5')]
    for i, (name, _) in enumerate(selected):
        r = next(r for r in report['comparisons'] if r['id']==name)
        lo, hi = np.array(r['accuracy_gain_pointwise_95'])*100
        axes[1].hlines(i,lo,hi,color='#235e91',lw=2)
        axes[1].scatter(100*r['accuracy_gain'],i,color='#235e91',s=30)
    axes[1].set_yticks(range(len(selected)), [p[1] for p in selected]);axes[1].invert_yaxis()
    axes[1].axvline(0,color='gray',lw=1)
    axes[1].set(xlabel='Preselection minus policy accuracy (points)',
                title='Paired pointwise 95% intervals; exploratory')
    axes[1].grid(axis='x',alpha=.2)
    fig.suptitle('EXPLORATORY: randomized model preselection, test n=512', fontsize=13)
    fig.tight_layout();fig.savefig(Path(out)/'exploratory_accuracy_cost.png',dpi=180);plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = analyze(args.cache, args.out)
    print(result['scope'])
    print(result['frontier'])
