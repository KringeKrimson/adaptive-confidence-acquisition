"""Thin Kaggle entry point. Source package is installed by the bundle notebook."""
import os
from pathlib import Path
from aca.cli import main


def run(mode='pilot', resume_from=None):
    import shutil
    # Both official model repositories are accessed anonymously. No secret is required.
    # Model weights are disposable; keep them outside saved experiment outputs.
    os.environ.setdefault('HF_HOME','/kaggle/temp/huggingface')
    root = Path('/kaggle/working/outputs') / mode
    if resume_from:
        shutil.copytree(resume_from,root,dirs_exist_ok=True)
    if mode == 'pilot':
        main(['prepare','--out',str(root),'--test','32'])
        main(['collect','--out',str(root)])
    elif mode == 'pilot14b':
        main(['prepare','--out',str(root),'--test','32'])
        main(['collect','--out',str(root),'--strong-model','Qwen/Qwen2.5-14B-Instruct',
              '--stages','small','strong','ic'])
    elif mode == 'research':
        # Reserve the deterministic pilot even when this is a fresh Kaggle session.
        pilot = root.parent / 'pilot'
        main(['prepare','--out',str(pilot),'--test','32'])
        main(['prepare','--out',str(root),'--train','1024','--validation','256','--test','512',
              '--exclude-manifest',str(pilot/'queries.json')])
        main(['collect','--out',str(root),'--stages','small','strong','ic'])
        main(['analyze','--out',str(root)])
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
        raise ValueError('mode must be pilot, pilot14b, research, or diagnostic')


if __name__ == '__main__':
    run(os.environ.get('ACA_MODE','pilot'),os.environ.get('ACA_RESUME_FROM'))
