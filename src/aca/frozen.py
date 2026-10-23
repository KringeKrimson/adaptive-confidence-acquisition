"""Freeze post-screen routing methods on calibration only; no test collection."""
from pathlib import Path
import numpy as np
from .routing import Policy, replay, summarize, tune_policies, random_budget_probabilities, random_replay
from .storage import atomic_json, bind_manifest, fingerprint, read_json

PENALTIES=(0.,.025,.05,.1,.2,.4,.8,1.6)


def partition_data(rows, ids, partition):
    selected=[r for r in rows if r['id'] in set(ids)]
    if len(selected)!=len(ids) or any(r['partition']!=partition for r in selected):
        raise ValueError(f'Requires exactly the {partition} queries')
    d={k:np.array([r[k] for r in selected]) for k in selected[0] if k not in ['id','partition']}
    d.update(ids=np.array([r['id'] for r in selected]),splits=np.array(['validation']*len(selected)))
    return d


def calibration_data(rows, ids):
    return partition_data(rows,ids,'calibration')


def implementation_hash():
    return fingerprint({n:(Path(__file__).with_name(n)).read_text() for n in ['frozen.py','routing.py']})


def freeze_methods(root, random_seeds=20):
    """Do not refit the probe or select a new policy using assessment outcomes."""
    from .prospective import PROTOCOL
    from .viability import check_protocol
    root=Path(root);check_protocol(root,PROTOCOL)
    screen=read_json(root/'viability.json')
    if screen['status']!='PASS' or not screen['valid_setup']:
        raise ValueError('Advance only after the unchanged prospective screen passes')
    if random_seeds<1:
        raise ValueError('Positive random seed count required')
    probe=read_json(root/'viability_probe.json')
    if probe['selection_ids']!=screen['calibration_ids'] or probe['training_ids']!=screen['train_ids']:
        raise ValueError('Probe selection/training provenance changed')
    d=calibration_data(read_json(root/'viability_rows.json'),screen['calibration_ids'])
    selected=tune_policies(d,PENALTIES)
    grid=[-.01,*np.linspace(0,1,11).tolist(),1.01]
    for point in selected:
        adaptive=replay(Policy(**point['adaptive']),d)
        budget=float(np.dot(adaptive['acquire'],d['ic_seconds']))
        point['random']=[]
        for seed in range(random_seeds):
            probabilities=random_budget_probabilities(d['ids'],d['ic_seconds'],budget,seed)
            choices=[]
            for cheap in grid:
                for ic in grid:
                    metrics=summarize(random_replay(d,cheap,ic,budget,seed,probabilities))
                    utility=metrics['accuracy']-point['penalty']*metrics['mean_seconds']/point['reference_seconds']
                    choices.append((utility,-metrics['mean_seconds'],cheap,ic))
            _,_,cheap,ic=max(choices)
            point['random'].append(dict(seed=seed,cheap_threshold=cheap,ic_threshold=ic))
    manifest=read_json(root/'inference_manifest.json')
    frozen=dict(schema=1,scope='Calibration-only frozen adaptive/baseline methods; final test unused',
        probe=probe,probe_hash=fingerprint(probe),inference_manifest=manifest,
        viability_protocol_hash=screen['protocol_hash'],selection_ids=screen['calibration_ids'],
        assessment_ids_excluded_from_selection=screen['assessment_ids'],selected=selected,
        penalties=list(PENALTIES),thresholds=grid,widths=[0.,.025,.05,.1,.2,.4,1.1],
        selection='Accuracy minus penalty * seconds/calibration strong-only mean; then least cost. Fixed policy enumeration breaks ties. Random threshold ties use largest threshold pair, matching existing analysis.',
        random_control='Retrospective timing-oracle random prefix, fractional boundary query, equal expected measured IC seconds per operating point. Not an online scheduler and not equal total routing cost.',
        reporting='Report all eight fixed operating points, all five requested baselines, actual IC and strong-call costs, paired bootstrap95 with seed4901/10000 repeats. No selection of a winning operating point using test results.',
        assessment_use='Screen verdict only; no assessment optimization or probe refit.',
        implementation_hash=implementation_hash())
    bind_manifest(root,'methods_frozen.json',frozen)
    return frozen


def compare_development(root):
    """Assess every already-frozen point; no fitting or candidate selection here."""
    from .analysis import bootstrap_delta
    root=Path(root);frozen=read_json(root/'methods_frozen.json')
    if frozen['implementation_hash']!=implementation_hash():
        raise ValueError('Frozen method implementation changed')
    if frozen['probe_hash']!=fingerprint(read_json(root/'viability_probe.json')):
        raise ValueError('Frozen probe changed')
    d=partition_data(read_json(root/'viability_rows.json'),frozen['assessment_ids_excluded_from_selection'],'assessment')
    results=[dict(method=kind,penalty=None,**summarize(replay(Policy(kind),d)))
             for kind in ['small-only','strong-only']]
    comparisons=[]
    for point in frozen['selected']:
        actual={kind:replay(Policy(**point[kind]),d) for kind in ['cheap-only','always-ic','adaptive']}
        for kind,r in actual.items():
            results.append(dict(method=kind,penalty=point['penalty'],**summarize(r)))
        adaptive=actual['adaptive'];budget=float(np.dot(adaptive['acquire'],d['ic_seconds']))
        random_runs=[random_replay(d,p['cheap_threshold'],p['ic_threshold'],budget,p['seed'])
                     for p in point['random']]
        assert all(np.isclose(np.dot(r['acquire'],d['ic_seconds']),budget) for r in random_runs)
        random_mean={key:np.mean([r[key] for r in random_runs],axis=0) for key in random_runs[0]}
        results.append(dict(method='random-ic-budget-matched',penalty=point['penalty'],**summarize(random_mean)))
        comparisons.append(dict(penalty=point['penalty'],ic_budget_seconds=budget,
            adaptive_minus_cheap=float(np.mean(adaptive['correct']-actual['cheap-only']['correct'])),
            adaptive_minus_random=float(np.mean(adaptive['correct']-random_mean['correct'])),
            bootstrap_95_vs_cheap=bootstrap_delta(adaptive['correct'],actual['cheap-only']['correct'],repeats=10000),
            bootstrap_95_vs_random=bootstrap_delta(adaptive['correct'],random_mean['correct'],repeats=10000)))
    report=dict(scope='Development assessment only; every operating point selected on calibration and frozen beforehand. Not final-test hypothesis evidence.',
        methods_hash=fingerprint(frozen),n_assessment=len(d['ids']),test_predictions_read=0,
        results=results,comparisons=comparisons,random_control=frozen['random_control'])
    atomic_json(root/'development_comparison.json',report)
    return report
