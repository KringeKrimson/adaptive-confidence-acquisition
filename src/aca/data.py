"""Closed-book TriviaQA; official validation is held out as project test."""
import hashlib
import re
from pathlib import Path
from .storage import atomic_json, fingerprint, read_json

DATASET = 'trivia_qa'
CONFIG = 'rc.nocontext'


def question_key(question):
    return ' '.join(re.sub(r'[^\w\s]', '', question.casefold()).split())


def train_partition(question, seed):
    digest = hashlib.sha256(f'{seed}:{question_key(question)}'.encode()).digest()
    return 'train' if int.from_bytes(digest[:8], 'big') % 5 else 'validation'


def validate_splits(queries):
    ids, questions = set(), set()
    for q in queries:
        if q['split'] not in {'train', 'validation', 'test'}:
            raise ValueError('Invalid split')
        key = question_key(q['question'])
        if q['id'] in ids or key in questions:
            raise ValueError('Duplicate query ID or normalized question across manifest')
        if not q['aliases']:
            raise ValueError('Missing reference answers')
        ids.add(q['id'])
        questions.add(key)


def prepare(root, train=0, validation=0, test=32, seed=4901, revision=None, exclude_manifest=None):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    path = Path(root) / 'queries.json'
    excluded = read_json(exclude_manifest) if exclude_manifest else None
    excluded_queries = excluded['queries'] if excluded else []
    if excluded:
        if revision and revision != excluded['revision']:
            raise ValueError('Excluded pilot and research must use the same dataset revision')
        revision = excluded['revision']
    requested = dict(train=train, validation=validation, test=test, seed=seed,
                     dataset=DATASET, config=CONFIG, excluded_queries_hash=fingerprint(excluded_queries))
    if min(train, validation, test) < 0 or train + validation + test == 0:
        raise ValueError('Sample counts must be nonnegative and total positive')
    if path.exists():
        saved = read_json(path)
        if saved['sampling'] != requested or (revision and saved['revision'] != revision):
            raise ValueError('Dataset sampling changed; use a new output directory')
        validate_splits(saved['queries'])
        return saved
    revision = HfApi(token=False).dataset_info(DATASET, revision=revision).sha
    queries = []
    seen_ids = {q['id'] for q in excluded_queries}
    seen_questions = {question_key(q['question']) for q in excluded_queries}
    remaining = dict(train=train, validation=validation, test=test)
    # Streaming avoids downloading the unused evidence corpus. Fixed shuffle buffer.
    for source in ['train', 'validation']:
        if source == 'train' and not (train or validation):
            continue
        if source == 'validation' and not test:
            continue
        ds = load_dataset(DATASET, CONFIG, split=source, streaming=True, revision=revision, token=False)
        ds = ds.shuffle(seed=seed, buffer_size=10000)
        for row in ds:
            split = train_partition(row['question'], seed) if source == 'train' else 'test'
            key = question_key(row['question'])
            qid = str(row['question_id'])
            if not remaining[split] or key in seen_questions or qid in seen_ids:
                continue
            answer = row['answer']
            aliases = sorted(set(answer.get('aliases', []) + [answer['value']]))
            queries.append(dict(id=qid, question=row['question'], aliases=aliases,
                                split=split, source_split=source))
            seen_ids.add(qid)
            seen_questions.add(key)
            remaining[split] -= 1
            if (source == 'train' and not (remaining['train'] or remaining['validation'])) or (
                    source == 'validation' and not remaining['test']):
                break
    if any(remaining.values()):
        raise ValueError(f'Insufficient unique data: {remaining}')
    validate_splits(queries)
    manifest = dict(sampling=requested, revision=revision, queries=queries)
    atomic_json(path, manifest)
    return manifest
