"""Independent arithmetic audit of the saved final-test run; no fitting/inference."""
import json
import sys
from pathlib import Path
import numpy as np
from aca.storage import read_json, fingerprint, load_records, atomic_json
from aca.metrics import exact_match, token_f1
from aca.confirmatory import PROTOCOL, portable_probe, verify_methods, evaluate

root = Path(sys.argv[1])
report = read_json(root/'confirmatory_results.json')
methods = read_json(root/'frozen_methods.json')
verify_methods(methods)
assert report['methods_hash'] == fingerprint(methods)
assert report['protocol_hash'] == fingerprint(PROTOCOL)
assert read_json(root/'confirmatory_protocol.json') == PROTOCOL
queries = read_json(root/'queries.json')['queries']
assert len(queries) == len({q['id'] for q in queries}) == 512
assert fingerprint(queries) == PROTOCOL['test_queries_hash']
manifest = read_json(root/'inference_manifest.json')
assert report['inference_config_hash'] == fingerprint(manifest)
stages = {s:load_records(root,s,queries) for s in ['small','strong','ic']}
for stage, rows in stages.items():
    assert len(list((root/stage).glob('*.json'))) == 512
    for q,r in zip(queries,rows):
        assert r['query_hash'] == fingerprint(q)
        assert r['config_hash'] == fingerprint(manifest)
        assert r['model_calls'] == 1 and r['elapsed_seconds'] > 0
        if stage != 'ic':
            assert r['correct'] == exact_match(r['prediction'],q['aliases'])
            assert abs(r['f1']-token_f1(r['prediction'],q['aliases'])) < 1e-12
        else:
            p = np.asarray(r['p_yes'])
            assert p.shape == (6,29) and r['yes_no_ids'] == [9454,2753]
            assert r['last_k'] == 6 and r['alpha'] == 1
            weights = [np.exp(-(np.arange(n)-(n-1))**2) for n in p.shape]
            w,v = [w/w.sum() for w in weights]
            assert abs(float(w@p@v)-r['ic']) < 1e-12
features = np.asarray([r['features'] for r in stages['small']])
assert features.shape == (512,1536) and np.isfinite(features).all()
scores = read_json(root/'test_scores.json')
assert np.allclose(portable_probe(methods['probe']).predict_proba(features)[:,1], scores['cheap'],atol=1e-12,rtol=0)
assert len(read_json(root/'per_query.json')) == 512
small = np.array([r['correct'] for r in stages['small']])
strong = np.array([r['correct'] for r in stages['strong']])
cheap,ic = np.array(scores['cheap']),np.array([r['ic'] for r in stages['ic']])
cs,ct,ci = [np.array([r['elapsed_seconds'] for r in stages[s]]) for s in ['small','strong','ic']]
cp = np.array(scores['probe_seconds'])
lookup = {r['id']:r for r in report['results']}
for i,point in enumerate(methods['selected']):
    saved = read_json(root/'decisions'/f'point-{i}.json')
    assert saved['policies'] == {k:point[k] for k in ['cheap-only','always-ic','adaptive']}
    for kind in ['cheap-only','always-ic','adaptive']:
        p = point[kind]
        acquisition = np.ones(512,dtype=bool) if kind=='always-ic' else abs(cheap-p['cheap_threshold'])<p['width'] if kind=='adaptive' else np.zeros(512,dtype=bool)
        route = np.where(acquisition,ic<p['ic_threshold'],cheap<p['cheap_threshold'])
        correctness = np.where(route,strong,small)
        cost = cs+(0 if kind=='always-ic' else cp)+acquisition*ci+route*ct
        expected = lookup[f'p{i}:{kind}']
        assert abs(correctness.mean()-expected['accuracy']) < 1e-12
        assert abs(cost.mean()-expected['mean_seconds']) < 1e-12
        for j,row in enumerate(saved['rows']):
            m=row['methods'][kind]
            assert m['strong']==route[j] and m['acquire']==acquisition[j]
            assert m['correct']==correctness[j] and abs(m['seconds']-cost[j])<1e-12
    a=point['adaptive'];acq=abs(cheap-a['cheap_threshold'])<a['width']
    initial=cheap<a['cheap_threshold'];route=np.where(acq,ic<a['ic_threshold'],initial)
    delta=(route.astype(int)-initial.astype(int))*(strong-small)
    changes=report['comparisons'][i]['routing_changes']
    assert changes['beneficial']==sum(delta>0) and changes['harmful']==sum(delta<0)
    assert changes['changed']==sum(route!=initial) and changes['acquired']==sum(acq)
    low,high=saved['mixture_endpoints'];lc=cs+cp+(cheap<low['cheap_threshold'])*ct;hc=cs+cp+(cheap<high['cheap_threshold'])*ct
    w=float(np.clip((lookup[f'p{i}:adaptive']['mean_seconds']-lc.mean())/(hc.mean()-lc.mean()),0,1))
    assert abs(w-saved['mixture']['high_probability'])<1e-12
    assert abs(lookup[f'p{i}:cheap-cost-matched']['mean_seconds']-((1-w)*lc+w*hc).mean())<1e-12
    for seed in range(20):
        probability=np.array([row['random_seeds'][seed]['acquire'] for row in saved['rows']])
        assert abs(float(probability@ci)-float(acq@ci))<1e-9
frontier = [r['id'] for r in report['results'] if not any(o['mean_seconds']<=r['mean_seconds']+1e-12 and o['accuracy']>=r['accuracy']-1e-12 and (o['mean_seconds']<r['mean_seconds']-1e-12 or o['accuracy']>r['accuracy']+1e-12) for o in report['results'])]
assert frontier == report['observed_frontier']
audit=dict(n_queries=512,stage_records=1536,strict_labels_recomputed=True,ic_aggregation_recomputed=True,probe_scores_recomputed=True,all_eight_policy_decisions_and_costs_recomputed=True,random_expected_ic_budgets_verified=True,frontier_verified=True,model_revisions=manifest['model_revisions'])
assert evaluate(root) == report, 'Saved report/bootstrap differs from recomputation'
audit['full_report_and_bootstrap_reproduced_locally'] = True
source = root.parents[1]/'aca-source'
repo = Path(__file__).resolve().parents[1]
if source.exists():
    checked=[]
    for path in source.rglob('*'):
        if path.is_file() and path.suffix in {'.py','.toml'} and '__pycache__' not in path.parts:
            relative=path.relative_to(source)
            local=Path('src')/Path(*relative.parts[2:]) if relative.parts[:2]==('build','lib') else relative
            assert (repo/local).read_bytes()==path.read_bytes(),str(relative)
            checked.append(str(relative))
    for name in ['qwen15_methods.json','confirmatory_protocol.json']:
        assert (source/'results'/name).read_bytes()==(repo/'results'/name).read_bytes()
    audit['source_bundle_matches_committed_files']=checked
atomic_json(root/'independent_audit.json',audit)
print(json.dumps(audit,indent=2))
