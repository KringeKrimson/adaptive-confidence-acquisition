import numpy as np
import pytest
from aca.confirmatory import matched_budget,matched_bootstrap,observed_frontier,portable_probe,weighted_quantiles


def r(seconds,correct):
    return dict(seconds=np.array(seconds,dtype=float),correct=np.array(correct,dtype=float),
        acquire=np.zeros(len(seconds)),strong=np.zeros(len(seconds)))


def test_cost_matching_uses_costs_only_and_does_not_extrapolate():
    low=r([1,1],[0,0]);high=r([3,3],[1,1]);adaptive=r([2,2],[1,0])
    matched,meta=matched_budget(adaptive,low,high)
    assert meta['exact_cost_match'] and meta['high_probability']==.5
    assert np.mean(matched['seconds'])==2 and np.mean(matched['correct'])==.5
    high['correct'][:]=0
    assert matched_budget(adaptive,low,high)[1]==meta
    above=r([4,4],[1,1]);mix,m=matched_budget(above,low,high)
    assert m['high_probability']==1 and not m['exact_cost_match'] and m['unused_budget_seconds']==1
    degenerate=matched_budget(low,low,low)
    assert degenerate[1]['exact_cost_match']


def test_paired_matching_bootstrap_recomputes_weights():
    low=r([1,1],[0,0]);high=r([3,3],[1,1]);adaptive=r([1,3],[0,1])
    # Resamples individually require weights 0 or 1; holding observed .5 is wrong.
    indices=np.array([[0,0],[1,1],[0,1]])
    ci=matched_bootstrap(adaptive,low,high,indices)
    assert ci['pointwise_95']==[0.,0.] and ci['familywise_95']==[0.,0.]


def test_frontier_preserves_ties_and_removes_dominated_points():
    rows=[dict(id='a',mean_seconds=1,accuracy=.5),dict(id='tie',mean_seconds=1,accuracy=.5),
        dict(id='dominated',mean_seconds=2,accuracy=.4),dict(id='b',mean_seconds=3,accuracy=.8)]
    assert observed_frontier(rows)==['a','tie','b']


def test_portable_probe_no_fitting_and_correct_scoring():
    saved=dict(C=.01,mean=[1.,2.],scale=[2.,1.],coefficients=[[1.,-2.]],intercept=[.3],training_ids=['train'])
    p=portable_probe(saved);x=np.array([[3.,3.],[1.,2.]])
    expected=1/(1+np.exp(-(((x-saved['mean'])/saved['scale'])@np.array(saved['coefficients'])[0]+.3)))
    assert np.allclose(p.predict_proba(x)[:,1],expected)


def test_random_latency_quantiles_use_branch_distribution():
    # A 50/50 mix has mean5 but its 95th percentile is9, not the conditional mean.
    assert weighted_quantiles([1,9],[1,1])==[1.,9.]
