import numpy as np
import pytest
from aca.data import train_partition, validate_splits
from aca.ic import aggregate_ic, positional_weights
from aca.metrics import exact_match, normalize_answer, token_f1
from aca.storage import atomic_json, bind_manifest, read_json
from aca.routing import Policy, random_budget_probabilities, replay, tune_policies


def test_triviaqa_metrics():
    assert normalize_answer(' The U.S.A.! ') == 'usa'
    assert exact_match('The Beatles',['Beatles','the beatles']) == 1
    assert exact_match('It is Paris',['Paris']) == 0
    assert token_f1('New York',['York']) == pytest.approx(2/3)


def test_splits_group_duplicate_questions():
    assert train_partition('Who is A?',4901) == train_partition('who is a',4901)
    q = dict(id='1',question='Who is A?',aliases=['A'],split='train')
    with pytest.raises(ValueError,match='Duplicate'):
        validate_splits([q,dict(q,id='2',split='test',question='who is a')])


def test_ic_matches_authors_nested_weighting():
    p = np.arange(33,dtype=float).reshape(3,11)/33
    tw = np.exp(-(np.arange(3)-2)**2); tw /= tw.sum()
    lw = np.exp(-(np.arange(11)-10)**2); lw /= lw.sum()
    expected = sum(p[t,l]*tw[t]*lw[l] for t in range(3) for l in range(11))
    assert aggregate_ic(p) == pytest.approx(expected)
    assert aggregate_ic(np.ones((3,33))*.7) == pytest.approx(.7)
    assert positional_weights(1).tolist() == [1.]
    with pytest.raises(ValueError):
        aggregate_ic([[np.nan]])


def fixture_data():
    return dict(ids=np.array(['a','b','c']),splits=np.array(['validation']*3),
        cheap=np.array([.1,.5,.9]),ic=np.array([.9,.1,.9]),
        small_correct=np.array([1,0,1]),strong_correct=np.array([0,1,1]),
        small_seconds=np.array([1.,1.,1.]),strong_seconds=np.array([5.,5.,5.]),
        ic_seconds=np.array([2.,3.,4.]),probe_seconds=np.array([.1,.1,.1]))


def test_actual_incremental_costs():
    d=fixture_data()
    r=replay(Policy('adaptive',.5,.5,.2),d)
    assert r['acquire'].tolist() == [0,1,0]
    assert r['strong'].tolist() == [1,1,0]
    assert r['seconds'].tolist() == pytest.approx([6.1,9.1,1.1])
    assert replay(Policy('strong-only'),d)['seconds'].tolist() == [5.,5.,5.]
    assert replay(Policy('always-ic'),d)['seconds'].tolist() == [3.,9.,5.]


@pytest.mark.parametrize('budget',[0.,.1,4.4,9.])
def test_random_matches_measured_ic_budget(budget):
    d=fixture_data()
    p=random_budget_probabilities(d['ids'],d['ic_seconds'],budget,3)
    assert np.dot(p,d['ic_seconds']) == pytest.approx(budget)
    assert np.all((p>=0)&(p<=1))


def test_validation_only_policy_tuning():
    d=fixture_data()
    selected=tune_policies(d,[0.1])
    assert len(selected)==1
    d['splits'][:]='test'
    with pytest.raises(ValueError,match='validation'):
        tune_policies(d,[0.1])


def test_atomic_cache_and_configuration(tmp_path):
    atomic_json(tmp_path/'a.json',{'complete':True})
    assert read_json(tmp_path/'a.json') == {'complete':True}
    bind_manifest(tmp_path,'manifest.json',{'revision':'abc'})
    with pytest.raises(ValueError,match='configuration'):
        bind_manifest(tmp_path,'manifest.json',{'revision':'def'})
    assert len(list(tmp_path.iterdir()))==2
