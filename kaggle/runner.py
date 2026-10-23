"""Thin Kaggle entry point. Source package is installed by the bundle notebook."""
import os
from pathlib import Path
from aca.cli import main


def run(mode='pilot', resume_from=None):
    import shutil
    # Both official model repositories are accessed anonymously. No secret is required.
    # Model weights are disposable; keep them outside saved experiment outputs.
    os.environ.setdefault('HF_HOME','/kaggle/temp/huggingface')
    output_name = 'research14b' if mode == 'viability14b' else mode
    root = Path('/kaggle/working/outputs') / output_name
    if resume_from:
        shutil.copytree(resume_from,root,dirs_exist_ok=True)
    if mode == 'qwen15screen':
        from aca.prospective import run as prospective_run
        try:
            prospective_run(root)
        except Exception as error:
            from aca.storage import atomic_json
            atomic_json(root/'run_failure.json',dict(type=type(error).__name__,error=str(error)))
            raise
    elif mode == 'pilot':
        main(['prepare','--out',str(root),'--test','32'])
        main(['collect','--out',str(root)])
    elif mode == 'pilot14b':
        main(['prepare','--out',str(root),'--test','32'])
        main(['collect','--out',str(root),'--strong-model','Qwen/Qwen2.5-14B-Instruct',
              '--stages','small','strong','ic'])
    elif mode in {'research','research14b','viability14b'}:
        from aca.viability import PROTOCOL, freeze_protocol
        # Reserve the deterministic pilot even when this is a fresh Kaggle session.
        pilot = root.parent / 'pilot'
        main(['prepare','--out',str(pilot),'--test','32','--revision',PROTOCOL['dataset_revision']])
        main(['prepare','--out',str(root),'--train','1024','--validation','256','--test','512',
              '--exclude-manifest',str(pilot/'queries.json')])
        if mode=='viability14b':
            from aca.storage import fingerprint, read_json, record_path
            frozen = freeze_protocol(root)
            print('Frozen viability protocol:',fingerprint(frozen),flush=True)
            test_queries = [q for q in read_json(root/'queries.json')['queries'] if q['split']=='test']
            if any(record_path(root,s,q).exists() for s in ['small','strong','ic'] for q in test_queries):
                raise ValueError('Viability run requires an untouched test prediction cache')
        strong = 'Qwen/Qwen2.5-14B-Instruct' if mode.endswith('14b') else 'Qwen/Qwen2.5-7B-Instruct'
        evaluation_splits = ['validation'] if mode=='viability14b' else ['validation','test']
        small_splits = ['train',*evaluation_splits]
        main(['collect','--out',str(root),'--strong-model',strong,'--stages','small','--splits',*small_splits])
        # Strong QA and IC on training examples are unused by probe/policy fitting.
        main(['collect','--out',str(root),'--strong-model',strong,'--stages','strong','ic',
              '--splits',*evaluation_splits])
        main(['viability' if mode=='viability14b' else 'analyze','--out',str(root)])
        if mode=='viability14b':
            assert not any(record_path(root,s,q).exists() for s in ['small','strong','ic'] for q in test_queries)
    elif mode == 'diagnostic':
        from aca.storage import atomic_json
        # The inspected pilot is diagnostic data, never research train/val/test.
        # Repeat FP16 with finite-logit checking, then compare stable FP32 compute.
        for precision in ['4bit', '4bit-fp32']:
            trial = root / precision
            main(['prepare','--out',str(trial),'--test','32'])
            try:
                main(['collect','--out',str(trial),'--quantization',precision,
                      '--stages','small','strong','ic'])
            except FloatingPointError as error:
                atomic_json(trial/'numerical_failure.json',dict(error=str(error)))
                print(f'{precision}: {error}', flush=True)
    else:
        raise ValueError('Unsupported experiment mode')


if __name__ == '__main__':
    run(os.environ.get('ACA_MODE','pilot'),os.environ.get('ACA_RESUME_FROM'))
