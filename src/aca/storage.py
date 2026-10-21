"""Atomic per-query checkpoints. Only a single writer per output directory."""
import hashlib
import json
import os
import tempfile
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, ensure_ascii=False, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def bind_manifest(root, name, manifest):
    path = Path(root) / name
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError(f'Cache configuration changed: {path}; use a different output directory')
    else:
        atomic_json(path, manifest)


def record_path(root, stage, query):
    return Path(root) / stage / (fingerprint(query) + '.json')


def load_records(root, stage, queries):
    records = []
    for q in queries:
        path = record_path(root, stage, q)
        if not path.exists():
            raise ValueError(f'Missing {stage} result for {q["id"]}; resume collection first')
        record = read_json(path)
        if record['query_hash'] != fingerprint(q):
            raise ValueError(f'Query mismatch: {path}')
        records.append(record)
    return records
