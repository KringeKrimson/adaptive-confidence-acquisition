"""Build a whitelist-only course code archive: no caches, data or fitted checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

REPO=Path(__file__).resolve().parents[1]
README='''# COMP4901B: Adaptive Confidence Acquisition

This is the clean code companion to the separately submitted conference-format PDF.
It contains no API keys, raw questions/predictions, private caches, LLM weights,
fitted probe/ridge checkpoints, virtual environments or dependency source trees.
`MANIFEST.json` records every included file hash and the intentional omissions.

## Install and reproduce the principal saved results (CPU; no inference)

Use Python 3.11+ in a fresh virtual environment. From this archive's root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 python -m pytest -q
python scripts/reproduce_paper.py --out outputs/paper-reproduction
```

The last command recreates all three paper figure pairs and the principal tables
from saved aggregate JSON. It copies the frozen intervals, without refitting,
bootstrap resampling, method selection or GPU calls. Inspect `tables.md` and
`provenance.json` in that output. This verifies the reported summaries, not an
independent reconstruction of paired uncertainty from per-query data.

## Build the PDF

Install a TeX distribution with pdflatex, BibTeX and standard packages, then run:

```bash
python scripts/build_paper.py --out outputs/paper-build
```

The source is `deliverables/REPORT.tex`, with the official, unchanged ACL style
used by EMNLP (acl-org/acl-style-files commit
`d5adc823ff0f80f98c80405ca0ab66c68e684409`). The builder checks for 6--8 main pages, unresolved citations and
overfull boxes. The team author block still needs the final author names.

## Configurations, prompts and evaluation

`configurations/original_routing.json` gives all eight original operating points,
20-seed random controls, exact model revisions, IC settings and recorded runtime;
`configurations/benefit_routing.json` gives development-selected regularization and
thresholds without fitted coefficients. `results/*protocol.json` defines costs,
matching, seeds, practical-effect rules and uncertainty. Exact QA/IC prompts live
in `src/aca/models.py` and `src/aca/ic.py` and are printed in the PDF appendix.
`confirmatory.py`, `exploratory_mixture.py`, `benefit.py` and `pregeneration.py`
contain the evaluation code; `routing.py` implements actions and cost replay.

The original confirmation is negative. The independent model-preselection result
is post-test exploratory. Both later studies are development-only; the reserved
fresh test is unused. Do not treat the unselected IC=.5 illustration as a new
confirmatory operating point.

## Optional new GPU work, not needed to reproduce saved tables

Use a supported CUDA/Linux environment and `requirements-gpu.txt`. The historical
image was Python 3.13.15, torch 2.11.0+cu128, CUDA 12.8 on Tesla T4. Requirements do
not freeze the OS/Kaggle image. Models/data are public; no HF token is needed.
This audit/package does not execute a GPU experiment.

For a Kaggle source snapshot, the generator requires a Git commit. Either use the
original repository, or initialize a local repository in the extracted archive,
add `src`, `pyproject.toml` and `kaggle/runner.py`, and commit with your own Git
identity. Then generate, without launching, a notebook for your account:

```bash
python scripts/kaggle_bundle.py --owner YOUR_KAGGLE_USERNAME --mode qwen15screen --out outputs/kaggle-bundle
```

Upload the generated `experiment.ipynb`, enable Internet and T4x2 in your own
eligible Kaggle account. Run All consumes GPU quota: this mode smoke-tests then
collects the historical development screen. The default pilot is a different,
earlier Phi pair. Keep future experiments in new outputs/protocols and never tune
on the consumed test. GitHub access does not grant private Kaggle resource access.

## Deliberate reproducibility boundary

Raw records, fresh-test manifests and fitted probe/ridge coefficients are omitted
to satisfy the course's no-data/cache/checkpoint archive requirement. Therefore
historical per-query replay needs separately controlled original caches and frozen
method artifacts, or independent reconstruction from public models/data. The
original test/benefit runners will not work without their expected frozen methods;
do not disable their hash guards or substitute the stripped configuration files.
Aggregate result fields containing per-query extension predictions/times were also
removed from archive copies; numerical summaries/intervals are unchanged.

`results/*provenance.json` maps those external records and their hashes.
No public private-cache download is configured. The independent development
runner can start from public resources without the omitted original checkpoints.
The existing presentation is not part of this code-only submission and its
proprietary authoring runtime is not required for the paper/tables.
'''


def build(destination):
    files={}
    origins={}
    def include(path):
        data=(REPO/path).read_bytes();files[path]=data
        origins[path]=hashlib.sha256(data).hexdigest()
    for path in ['.gitignore','pyproject.toml','requirements.txt','requirements-gpu.txt',
                 'deliverables/REPORT.tex','deliverables/references.bib','deliverables/acl.sty',
                 'deliverables/acl_natbib.bst']:
        include(path)
    for directory,pattern in [('src','*.py'),('tests','*.py'),('kaggle','*.py'),('scripts','*.py')]:
        for p in sorted((REPO/directory).rglob(pattern)):
            if '__pycache__' not in p.parts:include(p.relative_to(REPO).as_posix())
    for p in sorted((REPO/'deliverables/paper_figures').glob('*.pdf')):
        include(p.relative_to(REPO).as_posix())
    result_names=['qwen15_confirmatory','qwen15_exploratory_mixture','benefit_development',
                  'benefit_diagnostics','pregeneration_audit','pregeneration_diagnostics',
                  'confirmatory_protocol','benefit_protocol','pregeneration_protocol',
                  'qwen15_confirmatory_provenance','benefit_provenance','pregeneration_provenance']
    omissions={}
    for name in result_names:
        path=f'results/{name}.json';data=(REPO/path).read_bytes();d=json.loads(data)
        if name=='benefit_development':
            omitted=['assessment_predictions','measured_router_seconds']
            for key in omitted:d.pop(key,None)
            d['submission_omitted_fields']=omitted
            omissions[path]=omitted
            data=(json.dumps(d,indent=2)+'\n').encode()
        files[path]=data;origins[path]=hashlib.sha256((REPO/path).read_bytes()).hexdigest()
    original=json.loads((REPO/'results/qwen15_methods.json').read_text())
    config={k:original[k] for k in ['schema','scope','probe_hash','inference_manifest','selected',
            'penalties','thresholds','widths','selection','random_control','reporting',
            'assessment_use','implementation_hash']}
    config['probe_configuration']={'C':original['probe']['C'],'dimensions':1536,
        'fitted_coefficients':'Excluded from code archive; original artifact required for frozen replay'}
    files['configurations/original_routing.json']=(json.dumps(config,indent=2)+'\n').encode()
    b=json.loads((REPO/'results/benefit_methods.json').read_text())
    fields=['protocol_hash','fit_info','training_reference_seconds','policies','original_methods_hash','same_grid_correctness']
    files['configurations/benefit_routing.json']=(json.dumps({k:b[k] for k in fields},indent=2)+'\n').encode()
    files['README.md']=README.encode()
    forbidden={'.pt','.pth','.safetensors','.bin','.npy','.npz','.pkl','.pickle','.ipynb','.zip'}
    for name in files:
        assert not any(p in {'outputs','cache','.cache','.venv','node_modules','.git','__pycache__'} for p in Path(name).parts)
        assert Path(name).suffix not in forbidden,name
        assert Path(name).name not in ['kaggle.json','.env','access_token'],name
    manifest=dict(research_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        archive_type='Whitelist code/source/aggregate submission; no private data/caches/checkpoints',
        files={n:hashlib.sha256(d).hexdigest() for n,d in sorted(files.items())},
        original_repository_file_sha256=origins,
        aggregate_copy_omissions=omissions,
        excluded=['All raw per-query data/predictions/features/cost records','Fresh test manifests',
                  'Fitted qwen15_methods.json and benefit_methods.json checkpoints','LLM weights',
                  'Credentials and account authentication','Git history','Virtual environments/dependency source',
                  'Presentation and unrelated assignments'])
    files['MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo('COMP4901B_code/'+name,date_time=(2026,11,22,0,0,0))
            info.external_attr=0o100644<<16
            z.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED)
    print(f'Clean code archive: {destination}; {len(files)} files; {destination.stat().st_size} bytes')
    print('SHA256:',hashlib.sha256(destination.read_bytes()).hexdigest())
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='deliverables/COMP4901B_code.zip')
    build(p.parse_args().out)
