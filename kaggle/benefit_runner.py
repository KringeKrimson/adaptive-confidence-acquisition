"""Reuse attached v6 outputs. Collect missing strong TRAIN answers only."""
import os
from pathlib import Path
import shutil
from aca.storage import read_json,atomic_json,fingerprint,bind_manifest,record_path
from aca.benefit import PROTOCOL,fit_development


def run():
    os.environ.setdefault('HF_HOME','/kaggle/temp/huggingface')
    os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN','1')
    base=Path('/kaggle/working/outputs/benefit_extension')
    source=Path(__file__).resolve().parents[1]
    reservation=read_json(source/'results/benefit_reservation.json')
    assert reservation['protocol_hash']==fingerprint(PROTOCOL)
    fresh=read_json(source/'benefit_inputs/fresh_queries.json')
    assert fingerprint(fresh['queries'])==reservation['fresh_test_queries_hash']
    assert len(fresh['queries'])==512
    bind_manifest(base,'reservation.json',reservation)
    bind_manifest(base/'reserved_fresh_test','queries.json',fresh)
    print('FRESH TEST RESERVED; NO TEST INFERENCE:',fingerprint(fresh['queries']),flush=True)
    # Restore losslessly compressed float32 features using explicit filename order.
    compressed=source/'seed-cache/features.npz'
    if compressed.exists():
        import numpy as np
        values=np.load(compressed)['features']
        names=read_json(source/'seed-cache/feature_names.json')
        assert len(names)==len(values)==1280
        for name,value in zip(names,values):
            p=source/'seed-cache/qwen15screen/small'/name
            record=read_json(p);record['features']=value.astype(float).tolist();atomic_json(p,record)
    paths=[p for p in Path('/kaggle/input').rglob('queries.json') if p.parent.name in {'qwen15screen','development'}]
    paths += list((source/'seed-cache').rglob('queries.json'))
    candidates=[]
    for p in paths:
        q=read_json(p)['queries']
        if len(q)==1280 and sum(r['split']=='train' for r in q)==1024:
            candidates.append(p.parent)
    if not candidates:raise ValueError('Attach the saved Qwen development output version361511726, not the original final-test version')
    previous=max(candidates,key=lambda p:len(list((p/'strong').glob('*.json'))))
    root=base/'development';root.mkdir(parents=True,exist_ok=True)
    for name in ['queries.json','inference_manifest.json','model_revisions.json','small','strong','ic','overhead']:
        item=previous/name
        if not item.exists():continue
        files=list(item.rglob('*')) if item.is_dir() else [item]
        for p in files:
            if not p.is_file():continue
            target=root/p.relative_to(previous)
            if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    queries=read_json(root/'queries.json')['queries'];old=read_json(source/'results/qwen15_methods.json')
    assert {q['id'] for q in queries}==set(old['probe']['training_ids']+old['selection_ids']+old['assessment_ids_excluded_from_selection'])
    assert not {q['id'] for q in queries}&{q['id'] for q in fresh['queries']}
    manifest=read_json(root/'inference_manifest.json')
    assert manifest==old['inference_manifest']
    missing=[q for q in queries if q['split']=='train' and not record_path(root,'strong',q).exists()]
    atomic_json(base/'collection_plan.json',dict(missing_strong_training=len(missing),new_small_calls=0,new_ic_calls=0,new_test_calls=0))
    print('MISSING STRONG TRAINING:',len(missing),'NO SMALL/IC/TEST CALLS',flush=True)
    from aca.collect import collect
    collect(root,stages=['strong'],splits=['train'],small_model=manifest['model_roles']['small'],
        strong_model=manifest['model_roles']['strong'],quantization=manifest['quantization'],
        max_new_tokens=manifest['max_new_tokens'],answer_stop=manifest['answer_stop'])
    result=fit_development(root,base/'analysis',source/'results/qwen15_methods.json')
    print('BENEFIT EXTENSION COMPLETE:',result['status'],'Fresh test remains untouched',flush=True)


if __name__=='__main__':
    try:run()
    except Exception as error:
        atomic_json('/kaggle/working/outputs/benefit_extension/run_failure.json',dict(type=type(error).__name__,error=str(error)))
        raise
