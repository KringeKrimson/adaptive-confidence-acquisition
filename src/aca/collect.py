import importlib.metadata
import platform
import time
from pathlib import Path
import torch
from . import models
from .ic import AUTHOR_COMMIT
from .metrics import exact_match, token_f1
from .storage import atomic_json, bind_manifest, fingerprint, load_records, read_json, record_path


def runtime():
    packages = {}
    for p in ['torch', 'transformers', 'datasets', 'numpy', 'accelerate',
              'bitsandbytes', 'tokenizers', 'safetensors', 'pyarrow']:
        try:
            packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            packages[p] = None
    return dict(python=platform.python_version(), packages={
        **packages},
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def source_hash():
    return fingerprint({name: Path(__file__).with_name(name).read_text()
                        for name in ['models.py', 'ic.py', 'metrics.py', 'collect.py']})


def resolve_revisions(small=models.SMALL, strong=models.STRONG):
    from huggingface_hub import HfApi, hf_hub_download
    revisions = {name: HfApi().model_info(name).sha for name in [small, strong]}
    # Fail before GPU work if access to the gated strong model is missing.
    for name, sha in revisions.items():
        hf_hub_download(name, 'config.json', revision=sha)
    return revisions


def collect(root, stages=('small', 'strong'), max_new_tokens=32, quantization='4bit'):
    root = Path(root)
    dataset = read_json(root / 'queries.json')
    queries = dataset['queries']
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable; select a Kaggle GPU accelerator')
    revision_path = root / 'model_revisions.json'
    if revision_path.exists():
        revisions = read_json(revision_path)
    else:
        revisions = resolve_revisions()
        atomic_json(revision_path, revisions)
    config = dict(schema=1, dataset_hash=fingerprint(dataset), model_revisions=revisions,
                  max_new_tokens=max_new_tokens, quantization=quantization, batch_size=1,
                  source_hash=source_hash(), runtime=runtime(), ic_author_commit=AUTHOR_COMMIT,
                  ic_last_k=3, ic_alpha=1.0, ic_examples=[])
    bind_manifest(root, 'inference_manifest.json', config)
    for role, name in [('small', models.SMALL), ('strong', models.STRONG)]:
        relevant = [s for s in stages if (s in {'small', 'ic'} if role == 'small' else s == 'strong')]
        missing = {s: [q for q in queries if not record_path(root, s, q).exists()] for s in relevant}
        if not any(missing.values()):
            continue
        start = time.perf_counter()
        model, tokenizer = models.load_model(name, revisions[name], quantization)
        load_seconds = time.perf_counter() - start
        warmup = models.generate_answer(model, tokenizer, 'What is the capital of France?', 4,
                                        capture_features=role == 'small')
        if 'ic' in relevant:
            ic_warmup = models.estimate_ic(model, tokenizer, 'What is the capital of France?')
        else:
            ic_warmup = None
        atomic_json(root / 'overhead' / f'{role}-{time.time_ns()}.json', dict(
            model=name, load_seconds=load_seconds, qa_warmup=warmup['elapsed_seconds'],
            ic_warmup=ic_warmup['elapsed_seconds'] if ic_warmup else 0))
        try:
            for stage in relevant:
                for i, query in enumerate(missing[stage]):
                    if stage == 'ic':
                        result = models.estimate_ic(model, tokenizer, query['question'])
                    else:
                        result = models.generate_answer(model, tokenizer, query['question'],
                                                        max_new_tokens, capture_features=stage == 'small')
                        result.update(correct=exact_match(result['prediction'], query['aliases']),
                                      f1=token_f1(result['prediction'], query['aliases']))
                    result.update(id=query['id'], split=query['split'], model=name,
                                  query_hash=fingerprint(query), config_hash=fingerprint(config))
                    atomic_json(record_path(root, stage, query), result)
                    print(f'{stage}: saved {i+1}/{len(missing[stage])} ({query["id"]})', flush=True)
        finally:
            del model
            models.release_model(None)
    if all(s in stages for s in ('small', 'strong')):
        pilot_summary(root)


def pilot_summary(root):
    root = Path(root)
    queries = read_json(root / 'queries.json')['queries']
    summary, exported = {}, []
    for stage in ['small', 'strong']:
        records = load_records(root, stage, queries)
        summary[stage] = dict(n=len(records), accuracy=sum(r['correct'] for r in records)/len(records),
            mean_seconds=sum(r['elapsed_seconds'] for r in records)/len(records),
            total_input_tokens=sum(r['input_tokens'] for r in records),
            total_output_tokens=sum(r['output_tokens'] for r in records),
            truncated=sum(r['hit_token_limit'] for r in records))
        for q, record in zip(queries, records):
            exported.append(dict(query=q, stage=stage,
                                 **{k:v for k,v in record.items() if k != 'features'}))
    atomic_json(root / 'pilot_summary.json', summary)
    atomic_json(root / 'predictions.json', exported)
    print(summary, flush=True)
    return summary
