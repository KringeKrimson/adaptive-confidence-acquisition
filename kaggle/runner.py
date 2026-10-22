"""Thin Kaggle entry point. Source package is installed by the bundle notebook."""
import os
from pathlib import Path
from aca.cli import main


def run(mode='pilot', resume_from=None):
    import shutil
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret('HF_TOKEN')
        if token:
            os.environ['HF_TOKEN'] = token
    except Exception:
        pass  # Existing HF_TOKEN also works; gated-model preflight gives the actionable error.
    # Model weights are disposable; keep them outside saved experiment outputs.
    os.environ.setdefault('HF_HOME','/kaggle/temp/huggingface')
    root = Path('/kaggle/working/outputs') / mode
    if resume_from:
        shutil.copytree(resume_from,root,dirs_exist_ok=True)
    if mode == 'pilot':
        main(['prepare','--out',str(root),'--test','32'])
        main(['collect','--out',str(root)])
    elif mode == 'research':
        # Reserve the deterministic pilot even when this is a fresh Kaggle session.
        pilot = root.parent / 'pilot'
        main(['prepare','--out',str(pilot),'--test','32'])
        main(['prepare','--out',str(root),'--train','1024','--validation','256','--test','512',
              '--exclude-manifest',str(pilot/'queries.json')])
        main(['collect','--out',str(root),'--stages','small','strong','ic'])
        main(['analyze','--out',str(root)])
    else:
        raise ValueError('mode must be pilot or research')


if __name__ == '__main__':
    run(os.environ.get('ACA_MODE','pilot'),os.environ.get('ACA_RESUME_FROM'))
