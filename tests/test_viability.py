import numpy as np
import pytest
from aca.storage import atomic_json, read_json
from aca.viability import calibration_mask, check_protocol, freeze_protocol, screen_pair


def pair_data():
    sy=np.tile([0]*5+[1]*5,2)
    return dict(small_correct=sy,strong_correct=np.ones(20),cheap=np.where(sy,.9,.1),
                ic=np.where(sy,.9,.1),small_seconds=np.ones(20),strong_seconds=np.ones(20)*5,
                ic_seconds=np.ones(20)*2,probe_seconds=np.ones(20)*.01)


def test_screen_costs_and_assessment_is_not_used_for_selection():
    d=pair_data();cal=np.arange(20)<10
    first=screen_pair(d,cal)
    assert first['passed']
    assert first['assessment']['metrics']['mean_seconds']==pytest.approx(3.51)
    selected=first['selected_policy']
    d['cheap'][~cal]=.9
    second=screen_pair(d,cal)
    assert second['selected_policy']==selected
    assert second['gates']['correctness'] and second['gates']['rescues_and_harms']
    assert not second['gates']['quality_cost']
    d=pair_data();d['strong_correct']=d['small_correct'].copy()
    negative=screen_pair(d,cal)
    assert not negative['passed'] and not negative['gates']['correctness']
    assert negative['rescue_rate']==0


def test_always_ic_screen_pays_acquisition_cost():
    d=pair_data();d['cheap'][:]=.9;d['small_seconds'][:]=.1;d['ic_seconds'][:]=.01
    cal=np.arange(20)<10
    good=screen_pair(d,cal)
    assert good['selected_policy']['kind']=='always-ic' and good['passed']
    assert good['assessment']['metrics']['mean_seconds']==pytest.approx(2.61)
    d['ic_seconds'][:]=5
    assert not screen_pair(d,cal)['gates']['quality_cost']


def test_protocol_is_frozen_before_predictions(tmp_path):
    atomic_json(tmp_path/'queries.json',dict(queries=[]))
    freeze_protocol(tmp_path)
    check_protocol(tmp_path)
    saved=read_json(tmp_path/'viability_protocol.json')
    saved['protocol']['minimum_accuracy_advantage']=0
    atomic_json(tmp_path/'viability_protocol.json',saved)
    with pytest.raises(ValueError,match='changed'):
        check_protocol(tmp_path)
    other=tmp_path/'late'
    atomic_json(other/'queries.json',dict(queries=[]))
    atomic_json(other/'small'/'already.json',dict(correct=1))
    with pytest.raises(ValueError,match='before collecting'):
        freeze_protocol(other)


def test_fixed_partition_is_order_invariant():
    q=[dict(id=f'q{i}') for i in range(256)]
    assert calibration_mask(q).sum()==128
    a={x['id'] for x,selected in zip(q,calibration_mask(q)) if selected}
    rev=list(reversed(q))
    b={x['id'] for x,selected in zip(rev,calibration_mask(rev)) if selected}
    assert a==b
