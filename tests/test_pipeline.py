import numpy as np
import pytest
from aca.analysis import analyze, fit_probe, viability
from aca.storage import atomic_json, fingerprint, read_json, record_path


def test_train_only_probe():
    with pytest.raises(ValueError,match='training data'):
        fit_probe(np.ones((4,2)),[0,1,0,1],['test']*4)
    with pytest.raises(ValueError,match='correct and incorrect'):
        fit_probe(np.ones((4,2)),[1,1,1,1],['train']*4)


def test_cached_end_to_end_analysis(tmp_path):
    """Synthetic counterfactual cache: engineering validation only."""
    queries=[]
    for part in ['train','validation','test']:
        for i in range(8):
            queries.append(dict(id=f'{part}{i}',question=f'Question {part} {i}',
                                split=part,source_split='synthetic',aliases=['answer']))
    atomic_json(tmp_path/'queries.json',dict(queries=queries))
    from aca.viability import freeze_protocol
    freeze_protocol(tmp_path)
    manifest=dict(synthetic_test_only=True)
    atomic_json(tmp_path/'inference_manifest.json',manifest)
    for q in queries:
        i=int(q['id'][-1])
        common=dict(id=q['id'],split=q['split'],query_hash=fingerprint(q),config_hash=fingerprint(manifest))
        for stage in ['small','strong','ic']:
            if q['split']=='train' and stage!='small':
                continue  # Research analysis must not require unused training calls.
            r=dict(common,elapsed_seconds={'small':.1,'strong':.5,'ic':.04}[stage])
            if stage=='ic':
                r['ic']=.8 if i%2 else .2
            else:
                r.update(correct=i%2 if stage=='small' else int(i!=7),features=[float(i%2),float(i)])
            atomic_json(record_path(tmp_path,stage,q),r)
    test_record=record_path(tmp_path,'small',queries[-1])
    original=test_record.read_bytes()
    test_record.unlink()
    development=viability(tmp_path)
    assert development['test_predictions_read']==0
    assert development['n_train']==8 and development['n_validation']==8
    assert development['rescues']==4 and development['harms']==1
    test_record.write_bytes(original)
    report=analyze(tmp_path,penalties=[0.,.5],random_seeds=2)
    assert report['n_train']==8 and report['n_test']==8
    assert len(report['results'])==10
    assert (tmp_path/'accuracy_cost.png').stat().st_size>1000
    assert (tmp_path/'accuracy_cost.csv').exists()
    assert read_json(tmp_path/'selected_policies.json')['selection_split']=='validation'
    assert len(read_json(tmp_path/'probe.json')['coefficients'][0])==2
    scores=read_json(tmp_path/'probe_scores.json')
    assert len(scores['rows'])==24
    assert all(row['elapsed_seconds']>0 for row in scores['rows'])
    assert scores['probe_hash']==fingerprint(read_json(tmp_path/'probe.json'))


def test_interrupted_collection_resumes_completed_queries(tmp_path, monkeypatch):
    from aca import collect as module
    queries=[dict(id=str(i),question=f'query {i}',split='test',aliases=['answer']) for i in range(3)]
    atomic_json(tmp_path/'queries.json',dict(queries=queries))
    atomic_json(tmp_path/'model_revisions.json',{
        module.models.SMALL:'pinned-small',module.models.STRONG:'pinned-strong'})
    monkeypatch.setattr(module.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(module,'runtime',lambda:{'test':'fake'})
    monkeypatch.setattr(module.models,'load_model',lambda *args: (object(),object()))
    monkeypatch.setattr(module.models,'release_model',lambda *args:None)
    attempts=[]
    interrupted=[False]
    def generate(_m,_t,question,*args,**kwargs):
        attempts.append(question)
        if question=='query 1' and not interrupted[0]:
            interrupted[0]=True
            raise RuntimeError('simulated session interruption')
        return dict(prediction='answer',features=[1.,2.],elapsed_seconds=.1,
                    input_tokens=3,output_tokens=1,hit_token_limit=False)
    monkeypatch.setattr(module.models,'generate_answer',generate)
    with pytest.raises(RuntimeError,match='interruption'):
        module.collect(tmp_path,stages=['small'])
    saved=record_path(tmp_path,'small',queries[0])
    original=saved.read_bytes()
    module.collect(tmp_path,stages=['small'])
    assert saved.read_bytes()==original
    assert attempts.count('query 0')==1
    assert attempts.count('query 1')==2
    assert attempts.count('query 2')==1
    assert all(record_path(tmp_path,'small',q).exists() for q in queries)
    with pytest.raises(ValueError,match='Model pair changed'):
        module.collect(tmp_path,stages=['small'],strong_model=module.models.STRONG_MODELS[1])
    # A split-limited collection can later extend its cache to the test set.
    subset=tmp_path/'subset'
    partitioned=[dict(q,split=s) for q,s in zip(queries,['train','validation','test'])]
    atomic_json(subset/'queries.json',dict(queries=partitioned))
    atomic_json(subset/'model_revisions.json',read_json(tmp_path/'model_revisions.json'))
    module.collect(subset,stages=['small'],splits=['train','validation'])
    first=record_path(subset,'small',partitioned[0]).read_bytes()
    assert not record_path(subset,'small',partitioned[2]).exists()
    module.collect(subset,stages=['small'],splits=['test'])
    assert record_path(subset,'small',partitioned[0]).read_bytes()==first
    assert record_path(subset,'small',partitioned[2]).exists()
