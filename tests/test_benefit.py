import numpy as np
import pytest
from aca.benefit import fit_candidates,portable,restore,predict_one,route_result,random_match,route_predictions


def examples(partition):
    return [dict(query=dict(id=f'{partition}{i}',question='known easy question' if i%2 else 'unknown hard question'),
        partition=partition,features=[i%2,float(i%2)],small_correct=i%2,strong_correct=1-i%2,
        small_seconds=.2,strong_seconds=1.) for i in range(12)]


@pytest.mark.parametrize('kind',['preselection','cascade'])
def test_portable_benefit_model_preserves_scores_and_negative_benefit(kind):
    train,cal=examples('train'),examples('calibration')
    model,info=fit_candidates(train,cal,kind);new=restore(portable(model,kind))
    x=[r['query']['question'] if kind=='preselection' else r['features'] for r in cal]
    assert np.allclose(model.predict(x),new.predict(x),atol=1e-10,rtol=0)
    assert model.predict(x)[0,0]>0 and model.predict(x)[1,0]<0
    assert info['alpha'] in [1,10,100,1000]
    value,seconds=predict_one(new,x[0]);assert value.shape==(3,) and seconds>0


def test_assessment_cannot_enter_fitting_or_selection():
    with pytest.raises(ValueError):fit_candidates(examples('train'),examples('assessment'),'cascade')
    with pytest.raises(ValueError):fit_candidates(examples('assessment'),examples('calibration'),'preselection')


def test_preselection_avoids_first_model_cost_and_cascade_break_even():
    rows=[dict(small_correct=0,strong_correct=1,small_seconds=1,strong_seconds=3),
          dict(small_correct=0,strong_correct=0,small_seconds=1,strong_seconds=3)]
    selected=np.array([True,False]);h=np.array([.1,.1])
    cascade=route_result('cascade',selected,rows,h)
    pre=route_result('preselection',selected,rows,h)
    assert np.array_equal(cascade['correct'],pre['correct'])
    assert np.isclose(cascade['seconds'].mean()-pre['seconds'].mean(),.5)
    match=random_match(cascade,rows)
    assert np.isclose(match['strong_probability'],.8) and np.isclose(match['gain'],.1)
    # E[r*delta] > E[delta]*(E[h]+E[r*T])/(E[T]-E[S]).
    required=.5*(.1+.5*3)/(3-1)
    assert np.isclose(.5-required,match['gain'])
    assert np.allclose(random_match(cascade,rows,np.array([[0,0],[1,1]])),[0,0])


def test_routing_uses_appropriate_predicted_incremental_cost():
    predicted=np.array([[.5,.4,1.]])
    assert route_predictions('preselection',.6,0,predicted,1)[0]
    assert not route_predictions('cascade',.6,0,predicted,1)[0]


def test_frontier_retains_targeting_advantage_and_clips_over_budget():
    from aca.benefit_analysis import finite_frontier
    report=dict(random_preselection_curve=[dict(accuracy=.5,mean_seconds=.5),
        dict(accuracy=1.,mean_seconds=2.)],baselines=[],points=[
        dict(kind='preselection',penalty=.1,assessment=dict(accuracy=.65,mean_seconds=.75)),
        dict(kind='cascade',penalty=.1,assessment=dict(accuracy=.99,mean_seconds=3.))])
    frontier=finite_frontier(report)
    assert [p['id'] for p in frontier['fixed_points_not_dominated_by_line_or_other_points']]==['preselection:0.1']
    assert frontier['fixed_points_dominated_by_random_preselection']==['cascade:0.1']
