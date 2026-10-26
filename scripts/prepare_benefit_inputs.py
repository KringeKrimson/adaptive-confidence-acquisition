"""Reserve unseen questions and package only existing development caches."""
import argparse
from pathlib import Path
import io
import zipfile
import numpy as np
from aca.storage import read_json,atomic_json,fingerprint
from aca.data import prepare,question_key,validate_splits
from aca.benefit import PROTOCOL


def run(out):
    out=Path(out).resolve();repo=Path(__file__).resolve().parents[1]
    seen_ids=set();seen_questions=set();excluded=[];sources=[]
    for path in sorted((repo/'outputs').rglob('queries.json')):
        if any('benefit' in part for part in path.relative_to(repo/'outputs').parts):continue
        try:manifest=read_json(path)
        except (ValueError,KeyError):continue
        if 'queries' not in manifest:continue
        sources.append(str(path.relative_to(repo)))
        for q in manifest['queries']:
            key=question_key(q['question'])
            if q['id'] in seen_ids or key in seen_questions:continue
            seen_ids.add(q['id']);seen_questions.add(key);excluded.append(q)
    revision='0f7faf33a3908546c6fd5b73a660e0f8ff173c2f'
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'excluded.json',dict(revision=revision,queries=excluded))
    fresh=prepare(out/'fresh_test',test=512,seed=PROTOCOL['seed'],revision=revision,exclude_manifest=out/'excluded.json')
    validate_splits(fresh['queries'])
    assert not {q['id'] for q in fresh['queries']}&seen_ids
    assert not {question_key(q['question']) for q in fresh['queries']}&seen_questions
    meta=dict(protocol=PROTOCOL,protocol_hash=fingerprint(PROTOCOL),excluded_unique_queries=len(excluded),
        excluded_manifest_sources=sources,excluded_queries_hash=fingerprint(excluded),
        fresh_test_queries_hash=fingerprint(fresh['queries']),n_fresh_test=512,
        fresh_test_predictions=0,original_methods_hash=PROTOCOL['frozen_original_methods'])
    atomic_json(out/'reservation.json',meta)
    original=repo/'outputs/kaggle-qwen15-v6/outputs/qwen15screen'
    queries=read_json(original/'queries.json')['queries']
    cache={}
    for stage in ['small','strong','ic']:
        paths=list((original/stage).glob('*.json'));records=[read_json(p) for p in paths]
        if stage=='small':
            features=np.array([r.pop('features') for r in records],dtype=np.float64)
            assert features.shape==(1280,1536) and np.array_equal(features,features.astype(np.float32).astype(np.float64))
            buffer=io.BytesIO();np.savez_compressed(buffer,features=features.astype(np.float32))
            (out/'features.npz').write_bytes(buffer.getvalue())
        cache[stage]=dict(zip([p.name for p in paths],records))
    cache.update(manifests={name:read_json(original/name) for name in ['queries.json','inference_manifest.json','model_revisions.json']},
        original_methods=read_json(repo/'results/qwen15_methods.json'),fresh_test=fresh,reservation=meta)
    atomic_json(out/'cache.json',cache)
    with zipfile.ZipFile(out/'inputs.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['cache.json','features.npz']:z.write(out/name,name)
    print('Fresh reserved:',meta['fresh_test_queries_hash'],'excluded:',len(excluded),'input ZIP bytes:',(out/'inputs.zip').stat().st_size)
    return meta


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='outputs/benefit_inputs')
    run(parser.parse_args().out)
