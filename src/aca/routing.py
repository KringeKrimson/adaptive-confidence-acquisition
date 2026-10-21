"""Pure cached replay. Costs are warm batch-one service seconds, not dollars."""
from dataclasses import asdict, dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class Policy:
    kind: str
    cheap_threshold: float = 0.5
    ic_threshold: float = 0.5
    width: float = 0.0


def actions(policy, cheap, ic):
    cheap, ic = np.asarray(cheap), np.asarray(ic)
    n = len(cheap)
    if policy.kind == 'small-only':
        return np.zeros(n, bool), np.zeros(n, bool)
    if policy.kind == 'strong-only':
        return np.ones(n, bool), np.zeros(n, bool)
    acquire = np.zeros(n, bool)
    strong = cheap < policy.cheap_threshold
    if policy.kind == 'always-ic':
        acquire[:] = True
        strong = ic < policy.ic_threshold
    elif policy.kind == 'adaptive':
        acquire = np.abs(cheap - policy.cheap_threshold) < policy.width
        strong = np.where(acquire, ic < policy.ic_threshold, strong)
    elif policy.kind != 'cheap-only':
        raise ValueError(f'Unknown policy: {policy.kind}')
    return strong, acquire


def replay(policy, d):
    strong, acquire = actions(policy, d['cheap'], d['ic'])
    correct = np.where(strong, d['strong_correct'], d['small_correct']).astype(float)
    if policy.kind == 'strong-only':
        cost = d['strong_seconds'].copy()
    elif policy.kind == 'small-only':
        cost = d['small_seconds'].copy()
    else:
        # always-IC needs no probe score; adaptive and cheap-only do.
        probe = 0 if policy.kind == 'always-ic' else d['probe_seconds']
        cost = d['small_seconds'] + probe + acquire * d['ic_seconds'] + strong * d['strong_seconds']
    return dict(correct=correct, seconds=cost, acquire=acquire.astype(float), strong=strong.astype(float))


def summarize(r):
    return dict(n=len(r['correct']), accuracy=float(np.mean(r['correct'])),
                mean_seconds=float(np.mean(r['seconds'])),
                ic_rate=float(np.mean(r['acquire'])), strong_rate=float(np.mean(r['strong'])))


def random_budget_probabilities(ids, ic_seconds, budget, seed):
    """Retrospective random-prefix control, exact *expected* acquisition seconds.

    At most one boundary query is fractional. Timing-oracle offline control,
    not a claim that an online scheduler knows unseen query runtimes.
    """
    costs = np.asarray(ic_seconds, float)
    if not np.isfinite(costs).all() or (costs <= 0).any() or budget < 0:
        raise ValueError('Positive finite IC costs and nonnegative budget required')
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(f'{seed}:{ids[i]}'.encode()).digest())
    probabilities = np.zeros(len(ids))
    remaining = min(float(budget), float(costs.sum()))
    for i in order:
        probabilities[i] = min(1.0, remaining/costs[i])
        remaining = max(0., remaining - probabilities[i]*costs[i])
        if remaining <= 1e-12:
            break
    return probabilities


def random_replay(d, cheap_threshold, ic_threshold, budget, seed, probabilities=None):
    p = (random_budget_probabilities(d['ids'], d['ic_seconds'], budget, seed)
         if probabilities is None else probabilities)
    base = replay(Policy('cheap-only', cheap_threshold), d)
    ic_strong = d['ic'] < ic_threshold
    acquired_correct = np.where(ic_strong, d['strong_correct'], d['small_correct'])
    acquired_seconds = d['small_seconds'] + d['probe_seconds'] + d['ic_seconds'] + ic_strong*d['strong_seconds']
    return dict(correct=(1-p)*base['correct']+p*acquired_correct,
                seconds=(1-p)*base['seconds']+p*acquired_seconds, acquire=p,
                strong=(1-p)*base['strong']+p*ic_strong)


def tune_policies(validation, penalties):
    """Only receive validation data here. Select by accuracy - lambda * relative time."""
    if set(validation['splits']) != {'validation'}:
        raise ValueError('Policy selection may use validation data only')
    thresholds = np.r_[-0.01, np.linspace(0, 1, 11), 1.01]
    candidates = {
        'cheap-only': [Policy('cheap-only', cheap_threshold=float(t)) for t in thresholds],
        'always-ic': [Policy('always-ic', ic_threshold=float(t)) for t in thresholds],
        'adaptive': [Policy('adaptive', float(t), float(u), w)
                     for t in thresholds for u in thresholds for w in [0., .025, .05, .1, .2, .4, 1.1]],
    }
    scale = float(np.mean(validation['strong_seconds']))
    if scale <= 0:
        raise ValueError('Invalid reference inference costs')
    selected = []
    for penalty in penalties:
        point = dict(penalty=float(penalty), reference_seconds=scale)
        for kind, policies in candidates.items():
            scored = [(p, summarize(replay(p, validation))) for p in policies]
            best, _ = max(scored, key=lambda ps: (
                ps[1]['accuracy'] - penalty*ps[1]['mean_seconds']/scale, -ps[1]['mean_seconds']))
            point[kind] = asdict(best)
        selected.append(point)
    return selected
