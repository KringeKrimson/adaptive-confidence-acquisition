"""Frozen development screening rules, defined before the larger GPU run."""
from dataclasses import asdict
import hashlib
from pathlib import Path
import numpy as np
from .routing import Policy, replay, summarize
from .storage import bind_manifest, fingerprint, read_json

PROTOCOL = dict(
    version=1, seed=4901, n_train=1024, n_validation=256, n_calibration=128,
    n_assessment=128, minimum_accuracy_advantage=.05, minimum_rescue_rate=.20,
    maximum_harms_per_rescue=.50, minimum_routing_gain=.03,
    minimum_advantage_recovered=.50, maximum_strong_cost_fraction=.80,
    probe_C=[.01,.1,1.,10.], thresholds=[-.01,*[i/20 for i in range(21)],1.01],
    routing_methods=['cheap-only','always-ic'], bootstrap_repeats=10000,
    partition='Sort SHA256(4901:query_id); first 128 calibration, rest assessment',
    probe_selection='Train weights on train only; choose C by calibration log loss',
    policy_selection='On calibration, require gain >= max(0.03, 0.5 * max(strong-small,0)) and cost <= 0.8 * strong-only. Choose greatest accuracy, then least cost; fixed enumeration breaks ties.',
    decision='All three gates must pass. CI descriptive only. No criterion retuning, optional stopping, or final-test inference.',
    cost='Warm additive service seconds: small QA + required CPU probe + required IC + deferred strong QA. Load/warmup/training separate.',
    model_roles=dict(small='microsoft/Phi-3-mini-4k-instruct',strong='Qwen/Qwen2.5-14B-Instruct'),
    model_revisions={'microsoft/Phi-3-mini-4k-instruct':'f39ac1d28e925b323eae81227eaba4464caced4e',
                     'Qwen/Qwen2.5-14B-Instruct':'cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'},
    dataset_revision='0f7faf33a3908546c6fd5b73a660e0f8ff173c2f',
    quantization='4bit-fp32')


def freeze_protocol(root, protocol=PROTOCOL):
    root = Path(root)
    path = root/'viability_protocol.json'
    if not path.exists() and any(list((root/s).glob('*.json')) for s in ['small','strong','ic']):
        raise ValueError('Freeze viability criteria before collecting any predictions')
    frozen = dict(protocol=protocol,dataset_hash=fingerprint(read_json(root/'queries.json')))
    bind_manifest(root,'viability_protocol.json',frozen)
    bind_manifest(root,'model_revisions.json',protocol['model_revisions'])
    return frozen


def check_protocol(root, protocol=PROTOCOL):
    saved = read_json(Path(root)/'viability_protocol.json')
    expected = dict(protocol=protocol,dataset_hash=fingerprint(read_json(Path(root)/'queries.json')))
    if saved != expected:
        raise ValueError('Frozen viability protocol or dataset changed')
    return saved


def calibration_mask(queries, protocol=PROTOCOL):
    order = sorted(range(len(queries)),key=lambda i: hashlib.sha256(
        f'{protocol["seed"]}:{queries[i]["id"]}'.encode()).digest())
    mask = np.zeros(len(queries),bool)
    mask[order[:len(queries)//2]] = True
    return mask


def quality_cost_gate(metrics, small_accuracy, strong_accuracy, strong_seconds, protocol=PROTOCOL):
    required_gain = max(protocol['minimum_routing_gain'],protocol['minimum_advantage_recovered']*
                        max(0.,strong_accuracy-small_accuracy))
    gain = metrics['accuracy']-small_accuracy
    fraction = metrics['mean_seconds']/strong_seconds
    return dict(pass_gate=bool(gain+1e-12 >= required_gain and fraction <= protocol['maximum_strong_cost_fraction']+1e-12),
                accuracy_gain=gain,required_gain=required_gain,strong_cost_fraction=fraction)


def screen_pair(data, calibration, protocol=PROTOCOL):
    """Select one policy on calibration; assess it without consulting assessment labels."""
    calibration = np.asarray(calibration,bool)
    if not calibration.any() or calibration.all():
        raise ValueError('Calibration and assessment must both be nonempty')
    cal, assessment = [{k:np.asarray(v)[mask] for k,v in data.items()}
                       for mask in [calibration,~calibration]]
    candidates = []
    for kind in protocol['routing_methods']:
        for threshold in protocol['thresholds']:
            policy = (Policy(kind,cheap_threshold=threshold) if kind=='cheap-only'
                      else Policy(kind,ic_threshold=threshold))
            metrics = summarize(replay(policy,cal))
            gate = quality_cost_gate(metrics,float(cal['small_correct'].mean()),
                                     float(cal['strong_correct'].mean()),float(cal['strong_seconds'].mean()),protocol)
            candidates.append(dict(policy=asdict(policy),metrics=metrics,**gate))
    eligible = [c for c in candidates if c['pass_gate']]
    # If calibration fails, retain a budget-feasible diagnostic policy and still fail.
    budget_feasible = [c for c in candidates if c['strong_cost_fraction'] <= protocol['maximum_strong_cost_fraction']]
    chosen = max(eligible or budget_feasible or candidates,
                 key=lambda c:(c['metrics']['accuracy'],-c['metrics']['mean_seconds']))
    metrics = summarize(replay(Policy(**chosen['policy']),assessment))
    assessment_gate = quality_cost_gate(metrics,float(assessment['small_correct'].mean()),
        float(assessment['strong_correct'].mean()),float(assessment['strong_seconds'].mean()),protocol)
    sy, ty = np.asarray(data['small_correct']),np.asarray(data['strong_correct'])
    advantage = float(np.mean(ty-sy))
    rescues, harms = int(sum((sy==0)&(ty==1))),int(sum((sy==1)&(ty==0)))
    errors = int(sum(sy==0))
    rescue_rate = rescues/errors if errors else 0.
    correctness_pass = advantage+1e-12 >= protocol['minimum_accuracy_advantage']
    rescue_pass = (rescues>0 and rescue_rate+1e-12 >= protocol['minimum_rescue_rate']
                   and harms <= protocol['maximum_harms_per_rescue']*rescues+1e-12)
    routing_pass = bool(eligible and assessment_gate['pass_gate'])
    return dict(passed=bool(correctness_pass and rescue_pass and routing_pass),
        gates=dict(correctness=bool(correctness_pass),rescues_and_harms=bool(rescue_pass),quality_cost=routing_pass),
        accuracy_advantage=advantage,**{('phi_errors' if protocol['version']==1 else 'small_errors'):errors},rescues=rescues,harms=harms,
        rescue_rate=rescue_rate,harms_per_rescue=harms/rescues if rescues else None,
        selected_policy=chosen['policy'],calibration_eligible_count=len(eligible),
        calibration=chosen,assessment=dict(metrics=metrics,**assessment_gate),
        calibration_candidates=candidates)
