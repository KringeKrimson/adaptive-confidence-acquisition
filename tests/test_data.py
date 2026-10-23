from types import SimpleNamespace
from aca.data import prepare, question_key, validate_splits
from aca.storage import atomic_json


def test_prepare_excludes_inspected_pilot_from_all_splits(tmp_path,monkeypatch):
    import datasets
    import huggingface_hub
    class Stream:
        def __init__(self,rows): self.rows=rows
        def shuffle(self,**kwargs): return self
        def __iter__(self): return iter(self.rows)
    def load(*args,split,**kwargs):
        prefix='t' if split=='train' else 'v'
        rows=[dict(question_id=f'{prefix}{i}',question=f'{prefix} question {i}',
                   answer=dict(value='answer',aliases=['answer'])) for i in range(100)]
        return Stream(rows)
    monkeypatch.setattr(datasets,'load_dataset',load)
    monkeypatch.setattr(huggingface_hub,'HfApi',lambda **kwargs:SimpleNamespace(
        dataset_info=lambda *a,**kw:SimpleNamespace(sha='revision')))
    pilot=tmp_path/'pilot.json'
    reserved=[dict(id='v0',question='v question 0',aliases=['answer'],split='test')]
    atomic_json(pilot,dict(revision='revision',queries=reserved))
    m=prepare(tmp_path/'research',train=3,validation=3,test=3,exclude_manifest=pilot)
    assert len(m['queries'])==9
    validate_splits(m['queries'])
    assert not {q['id'] for q in m['queries']} & {'v0'}
    assert not {question_key(q['question']) for q in m['queries']} & {'v question 0'}
    assert prepare(tmp_path/'research',train=3,validation=3,test=3,exclude_manifest=pilot)==m
