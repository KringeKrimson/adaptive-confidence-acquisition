from types import SimpleNamespace
from aca.collect import pilot_summary, resolve_revisions
from aca.models import SMALL, STRONG
from aca.storage import atomic_json, fingerprint, read_json, record_path


def test_public_model_preflight_checks_weights_anonymously(monkeypatch):
    import huggingface_hub as hub
    calls=[]
    class Api:
        def __init__(self,token): assert token is False
        def model_info(self,name):
            return SimpleNamespace(sha='revision',siblings=[SimpleNamespace(rfilename='model-1.safetensors')])
    def download(name,filename,revision,token):
        assert token is False
        calls.append((name,filename))
    def metadata(url,token):
        assert token is False
        calls.append(('weight-access',url))
    monkeypatch.setattr(hub,'HfApi',Api)
    monkeypatch.setattr(hub,'hf_hub_download',download)
    monkeypatch.setattr(hub,'get_hf_file_metadata',metadata)
    assert resolve_revisions()=={SMALL:'revision',STRONG:'revision'}
    assert sum(name=='weight-access' for name,_ in calls)==2


def test_pilot_reports_rescues_and_harms(tmp_path):
    queries=[dict(id=str(i),question=f'question {i}',split='test',aliases=['answer']) for i in range(4)]
    atomic_json(tmp_path/'queries.json',dict(queries=queries))
    for stage,labels in [('small',[1,0,0,1]),('strong',[1,0,1,0])]:
        for q,correct in zip(queries,labels):
            record=dict(id=q['id'],query_hash=fingerprint(q),model=SMALL if stage=='small' else STRONG,
                        correct=correct,prediction='answer',features=[],elapsed_seconds=.1,
                        input_tokens=10,output_tokens=2,hit_token_limit=False)
            atomic_json(record_path(tmp_path,stage,q),record)
    summary=pilot_summary(tmp_path)
    assert summary['strong']['model']==STRONG
    assert summary['paired']==dict(both_correct=1,both_wrong=1,strong_rescues=1,strong_harms=1,
                                   strong_minus_small_accuracy=0.)
    assert len(read_json(tmp_path/'predictions.json'))==8
