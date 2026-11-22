"""Typeset the course paper with official ACL style; no research evaluation runs."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from pypdf import PdfReader

REPO=Path(__file__).resolve().parents[1]


def build(out, pdf=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    source=REPO/'deliverables'
    pdflatex=shutil.which('pdflatex');bibtex=shutil.which('bibtex')
    if not pdflatex or not bibtex:
        raise RuntimeError('Install TeX Live/MacTeX with pdflatex and bibtex, or upload the included LaTeX source to Overleaf')
    subprocess.run([sys.executable,str(REPO/'scripts/reproduce_paper.py'),'--out',str(out/'assets')],check=True)
    env=dict(os.environ,BIBINPUTS=str(source)+os.pathsep,BSTINPUTS=str(source)+os.pathsep)
    def run(command, name, cwd):
        result=subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True)
        (out/name).write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f'Typesetting failed; inspect {out/name}')
    arg=r'\def\paperfigures{'+(out/'assets/figures').as_posix()+r'/}\input{REPORT.tex}'
    command=[pdflatex,'-interaction=nonstopmode','-halt-on-error','-jobname=COMP4901B_report',
             f'-output-directory={out}',arg]
    run(command,'latex-pass-1.txt',source)
    run([bibtex,'COMP4901B_report'],'bibtex.txt',out)
    for i in [2,3]:run(command,f'latex-pass-{i}.txt',source)
    log=(out/'COMP4901B_report.log').read_text()
    if 'undefined references' in log or 'There were undefined citations' in log:
        raise RuntimeError('Resolve undefined references/citations before delivery')
    if re.search(r'Overfull \\[hv]box',log):
        raise RuntimeError('Resolve overfull boxes before delivery; see the TeX log')
    result=out/'COMP4901B_report.pdf';reader=PdfReader(result)
    text=[p.extract_text() or '' for p in reader.pages]
    references=next(i for i,t in enumerate(text) if 'References' in [line.strip() for line in t.splitlines()])
    if not 6<=references<=8:
        raise RuntimeError(f'Main text is {references} pages; course requires 6--8 excluding references/appendices')
    receipt=dict(total_pages=len(text),main_pages=references,references_start_page=references+1,
                 saved_results_only=True,new_research_inference=False)
    (out/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if pdf:
        destination=Path(pdf).resolve();destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(result,destination);result=destination
    print(f'Validated {references} main pages, {len(text)} total pages: {result}')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='outputs/paper-build')
    parser.add_argument('--pdf',help='Optional final PDF destination')
    args=parser.parse_args();build(args.out,args.pdf)
