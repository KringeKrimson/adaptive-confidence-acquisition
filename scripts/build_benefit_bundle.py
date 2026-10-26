"""Upload committed modules and a verified development cache, without weights."""
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np


def build(destination):
    repo=Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git','status','--porcelain','--','src','kaggle/benefit_runner.py','results/benefit_protocol.json','results/benefit_reservation.json'],cwd=repo).strip():
        raise RuntimeError('Commit the protocol and implementation before bundling')
    names=subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines()
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:
            if name.startswith('src/') or name in ['pyproject.toml','kaggle/benefit_runner.py','results/qwen15_methods.json','results/benefit_protocol.json','results/benefit_reservation.json']:
                z.writestr(name,subprocess.check_output(['git','show',f'HEAD:{name}'],cwd=repo))
        z.writestr('benefit_inputs/fresh_queries.json',(repo/'outputs/benefit_inputs/fresh_test/queries.json').read_bytes())
        cached=repo/'outputs/kaggle-qwen15-v6/outputs/qwen15screen'
        feature_names=[];features=[]
        for p in cached.rglob('*'):
            if p.is_file() and (p.relative_to(cached).parts[0] in ['small','strong','ic','overhead'] or p.name in ['queries.json','inference_manifest.json','model_revisions.json']):
                name='seed-cache/qwen15screen/'+str(p.relative_to(cached))
                if p.parent.name=='small':
                    record=json.loads(p.read_text());feature_names.append(p.name);features.append(record.pop('features'))
                    z.writestr(name,json.dumps(record))
                else:z.write(p,name)
        array=np.asarray(features,dtype=np.float64)
        assert np.array_equal(array,array.astype(np.float32).astype(np.float64))
        feature_buffer=io.BytesIO();np.savez_compressed(feature_buffer,features=array.astype(np.float32))
        z.writestr('seed-cache/features.npz',feature_buffer.getvalue())
        z.writestr('seed-cache/feature_names.json',json.dumps(feature_names))
    data=buffer.getvalue();sha=hashlib.sha256(data).hexdigest()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    extract=f'''import base64,hashlib,io,zipfile,pathlib
payload=base64.b64decode({base64.b64encode(data).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={sha!r}
source=pathlib.Path('/kaggle/working/aca-source');source.mkdir(exist_ok=True)
zipfile.ZipFile(io.BytesIO(payload)).extractall(source)
print('Source commit: {commit}',flush=True)
'''
    install="import subprocess,sys\nsubprocess.check_call([sys.executable,'-m','pip','install','/kaggle/working/aca-source[gpu]'])\n"
    run="import subprocess,sys,os\nsubprocess.check_call([sys.executable,'/kaggle/working/aca-source/kaggle/benefit_runner.py'],env=dict(os.environ,PYTHONUNBUFFERED='1'))\n"
    cells=[dict(id='description',cell_type='markdown',metadata={},source=[f'# Final benefit-aware extension\nSource commit `{commit}`. Development only: missing1024 strong training answers. Verified development cache from version361511726 is embedded. Fresh512 test reserved but NOT executed.\n'])]
    for name,code in [('extract',extract),('install',install),('run',run)]:
        cells.append(dict(id=name,cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True)))
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    (out/'experiment.ipynb').write_text(json.dumps(dict(cells=cells,nbformat=4,nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')))))
    print('Bundle:',out/'experiment.ipynb','bytes:',(out/'experiment.ipynb').stat().st_size,'commit:',commit)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='outputs/benefit_bundle')
    build(parser.parse_args().out)
