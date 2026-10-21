"""Generate a thin, self-contained upload notebook from committed source only."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile


def build(destination, owner, mode='pilot', resume_kernel=None):
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git','status','--porcelain','--','src','pyproject.toml','kaggle/runner.py'],cwd=root).strip():
        raise RuntimeError('Commit inference source before building a reproducible bundle')
    files = subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            if name and (name.startswith('src/') or name in ['pyproject.toml','kaggle/runner.py']):
                content = subprocess.check_output(['git','show',f'HEAD:{name}'],cwd=root)
                archive.writestr(zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0)),content,
                                 compress_type=zipfile.ZIP_DEFLATED)
    data = buffer.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip()
    extraction = f'''import base64, hashlib, io, zipfile, pathlib
payload = base64.b64decode({base64.b64encode(data).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {sha!r}
source = pathlib.Path('/kaggle/working/aca-source')
source.mkdir(exist_ok=True)
zipfile.ZipFile(io.BytesIO(payload)).extractall(source)
print('Source commit: {commit}')
'''
    installation = '''import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '/kaggle/working/aca-source[gpu]'])
'''
    execution = f'''import os, runpy
os.environ['ACA_MODE'] = {mode!r}
runpy.run_path('/kaggle/working/aca-source/kaggle/runner.py', run_name='__main__')
'''
    if resume_kernel:
        execution = f"os.environ['ACA_RESUME_FROM'] = '/kaggle/input/{resume_kernel}/outputs/{mode}'\n" + execution
        execution = 'import os\n' + execution
    cells = [dict(cell_type='markdown',metadata={},source=[
        '# Adaptive Confidence Acquisition\n',
        f'Source commit `{commit}`. Mode: **{mode}**. Requires GPU, Internet and HF_TOKEN secret with Llama access.\n',
        'Save & Run All to retain per-query outputs. Experimental logic lives in Python modules.'])]
    for code in [extraction,installation,execution]:
        cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True)))
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'experiment.ipynb').write_text(json.dumps(dict(cells=cells,nbformat=4,nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'))),indent=1))
    slug = f'adaptive-confidence-{mode}'
    metadata = dict(id=f'{owner}/{slug}',title=f'Adaptive Confidence Acquisition {mode}',
        code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,
        enable_gpu=True,enable_internet=True,dataset_sources=[],competition_sources=[],
        kernel_sources=[f'{owner}/{resume_kernel}'] if resume_kernel else [],model_sources=[])
    (destination/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    print(f'Created {destination}/experiment.ipynb; source commit {commit}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--owner',default='sushobhanrai')
    parser.add_argument('--mode',choices=['pilot','research'],default='pilot')
    parser.add_argument('--out',default='outputs/kaggle-bundle')
    parser.add_argument('--resume-kernel')
    args = parser.parse_args()
    build(args.out,args.owner,args.mode,args.resume_kernel)
