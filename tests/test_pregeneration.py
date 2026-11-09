import numpy as np
import pytest
from aca.pregeneration import scalar_replay,prefill_replay,signals,match_family,select


def data():
    return dict(ids=np.array(['a','b']),partition=np.array(['calibration']*2),
        small_correct=np.array([0.,1.]),strong_correct=np.array([1.,0.]),
        small_seconds=np.array([1.,2.]),strong_seconds=np.array([5.,6.]),
        ic_seconds=np.array([.2,.3]),probe_seconds=np.array([.01,.02]),
        cheap=np.array([.4,.9]),weighted=np.array([.1,.9]))


def test_ic_first_avoids_only_escalated_small_answers():
    d=data();early=scalar_replay(d,'weighted',.5,'ic-first');late=scalar_replay(d,'weighted',.5,'answer-first')
    assert np.allclose(early['seconds'],[5.2,2.3])
    assert np.array_equal(early['correct'],late['correct'])
    assert np.allclose(late['seconds']-early['seconds'],early['strong']*d['small_seconds'])


def test_prefill_is_charged_on_strong_branch_and_reused_on_small_branch():
    d=data();r=prefill_replay(d,.5,.5,.2,.25)
    assert np.allclose(r['seconds'],[.25+.01+.2+5,2+.02])
    assert np.allclose(prefill_replay(d,.5,.5,.2,1)['seconds'],[1+.01+.2+5,2+.02])


def test_same_matrix_alternatives_do_not_get_free_forward_pass():
    m=np.full((6,29),.3);v=signals(m)
    assert all(np.isclose(x,.3) for x in v.values())
    d=data();d['top_right']=d['weighted'].copy()
    assert np.array_equal(scalar_replay(d,'top_right',.5,'ic-first')['seconds'],scalar_replay(d,'weighted',.5,'ic-first')['seconds'])
    assert scalar_replay(d,'weighted',-.01,'ic-first')['acquire'].sum()==0
    d['weighted'][:]=.9
    assert scalar_replay(d,'weighted',.5,'ic-first')['acquire'].sum()==2


def test_cost_bracket_does_not_select_best_assessment_accuracy():
    d=data()
    def r(cost,correct):
        return dict(seconds=np.full(2,cost),correct=np.full(2,correct),strong=np.zeros(2),acquire=np.zeros(2))
    # Target3.5: nearest bracket3.4/3.6 must win over a more accurate cheaper route.
    out=match_family([r(3.0,1),r(3.4,0),r(3.6,0)],d,.5)
    assert np.isclose(out['correct'],0) and np.isclose(out['seconds'],3.5)
    d['partition'][:]='assessment'
    with pytest.raises(ValueError,match='Calibration only'):select([],d,None)
