import numpy as np
import pytest
from aca.exploratory_mixture import preselect, budget_probability, comparison


def test_preselection_runs_only_the_chosen_model():
    small = dict(correct=np.array([1., 0.]), seconds=np.array([2., 2.]))
    strong = dict(correct=np.array([0., 1.]), seconds=np.array([10., 10.]))
    assert np.array_equal(preselect(small, strong, 0)['seconds'], small['seconds'])
    assert np.array_equal(preselect(small, strong, 1)['seconds'], strong['seconds'])
    result = preselect(small, strong, .25)
    assert np.array_equal(result['seconds'], [4., 4.])  # Cascade would cost4.5.
    assert np.array_equal(result['correct'], [.75, .25])
    with pytest.raises(ValueError):
        preselect(small, strong, 1.01)


def test_budget_matching_clips_without_extrapolation():
    assert np.array_equal(budget_probability(np.array([0., 1., 2., 3., 4.]), 1., 3.),
                          [0., 0., .5, 1., 1.])
    with pytest.raises(ValueError):
        budget_probability(2., 1., 1.)


def test_paired_bootstrap_refits_cost_weight_and_endpoint_dominance():
    small = dict(correct=np.zeros(2), seconds=np.ones(2))
    strong = dict(correct=np.ones(2), seconds=np.full(2, 3.))
    policy = dict(correct=np.array([0., 1.]), seconds=np.array([1., 3.]))
    bs = dict(correct=np.zeros(3), seconds=np.ones(3))
    bt = dict(correct=np.ones(3), seconds=np.full(3, 3.))
    bp = dict(correct=np.array([0., 1., .5]), seconds=np.array([1., 3., 2.]))
    result, _ = comparison(small, strong, policy, bs, bt, bp)
    assert result['accuracy_gain_pointwise_95'] == [0., 0.]
    assert result['cost_saving_pointwise_95'] == [0., 0.]
    assert result['exact_cost_match'] and not result['observed_dominated']
    costly = dict(correct=np.full(2, .5), seconds=np.full(2, 4.))
    costly_boot = dict(correct=np.full(3, .5), seconds=np.full(3, 4.))
    result, _ = comparison(small, strong, costly, bs, bt, costly_boot)
    assert result['strong_probability'] == 1 and result['observed_dominated']
    assert result['cost_saving_seconds'] == 1 and not result['exact_cost_match']
