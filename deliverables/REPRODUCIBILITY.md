# Reproducing the completed COMP4901B study

## Course submission, 22 November 2026

Submit [COMP4901B_report.pdf](COMP4901B_report.pdf) and
[COMP4901B_code.zip](COMP4901B_code.zip). The PDF has eight main pages, one page of
references and two appendix pages. [REPORT.tex](REPORT.tex) is the typesetting
source; the older Markdown manuscript below remains the detailed evidence record.
The official ACL style used by EMNLP is included unchanged from
[`acl-org/acl-style-files`, commit d5adc823](https://github.com/acl-org/acl-style-files/tree/d5adc823ff0f80f98c80405ca0ab66c68e684409).
The official style was approved for this report. The author block lists
Sushobhan Rai and Noah Breidung with the existing HKUST affiliation.

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/reproduce_paper.py --out outputs/paper-reproduction
python scripts/build_paper.py --out outputs/paper-build
python scripts/build_submission.py --out deliverables/COMP4901B_code.zip
```

The first script reads only tracked aggregate JSON, reproduces principal tables
and three paper figure pairs, and copies saved intervals unchanged. It does not
fit, bootstrap, select policies or access raw query caches. The PDF builder needs
pdfLaTeX/BibTeX and checks the main-page limit and typesetting errors. It uses no
proprietary presentation runtime. The `paper_figures/` variants improve label
legibility; all six earlier figure pairs and original evaluations remain intact.

The ZIP is built from an explicit whitelist with a per-file hash manifest. It
omits raw data, per-query records/features/times, credentials, LLM weights, fitted
probe/ridge checkpoints, dependency source and unrelated assignments. Its copies
of `benefit_development.json` omit the two per-query prediction/timing fields;
the repository original and all aggregate values/intervals are unchanged. Original
policy thresholds and runtime settings are exported separately under
`configurations/`, without fitted coefficients. This code-only archive therefore
cannot perform full historical per-query replay without separately controlled
caches and fitted method artifacts. It can regenerate the reported aggregate
tables/figures immediately, and includes collection/evaluation code for independent
work. Its README documents the Git snapshot requirement for the Kaggle generator.

`requirements.txt` pins direct/critical CPU analysis and test dependencies;
`requirements-gpu.txt` adds the historical CUDA build and quantization package.
Neither is a full operating-system/Kaggle-image lock. No new research experiment
was conducted while preparing these submission files.

Submission validation used a separately extracted ZIP and fresh CPU environment:
pinned installation, `pip check`, all 47 offline tests, aggregate table/figure
regeneration and PDF compilation passed. The rebuilt PDF has identical extracted
text on all 11 pages. Archive hashes/exclusions and a redacted Gitleaks scan passed;
no credential candidates were detected. The original scientific files and six
legacy figure pairs retain their recorded hashes.

## Existing research records

[REPORT.md](REPORT.md) is the consolidated manuscript.
[COMP4901B_research.pptx](COMP4901B_research.pptx) is its 12-slide presentation.
The [figures](figures/) contain six PNG/PDF pairs. Earlier summaries and the
standalone architecture audit remain historical records, not competing reports.

## Evidence and evaluation structure

| Stage | Authoritative numerical records in `results/` | Implementation / cached outputs |
|---|---|---|
| Frozen original confirmation | `qwen15_methods.json`, `confirmatory_protocol.json`, `qwen15_confirmatory.json`, `qwen15_confirmatory_provenance.json` | `src/aca/confirmatory.py`; `outputs/kaggle-confirmatory-v7/outputs/confirmatory/` |
| Post-test model preselection | `qwen15_exploratory_mixture.json` | `src/aca/exploratory_mixture.py`; same original-test cache |
| Benefit-aware development STOP | `benefit_protocol.json`, `benefit_methods.json`, `benefit_development.json`, `benefit_diagnostics.json`, `benefit_provenance.json`, `benefit_reservation.json` | `src/aca/benefit.py`, `benefit_analysis.py`; `outputs/kaggle-benefit-development-v1/` |
| Pre-generation development audit, `ffe94cf` | `pregeneration_protocol.json`, `pregeneration_selected.json`, `pregeneration_audit.json`, `pregeneration_diagnostics.json`, `pregeneration_provenance.json` | `src/aca/pregeneration.py`, `scripts/report_pregeneration_audit.py`; `outputs/pregeneration-audit/` and original development cache `outputs/kaggle-qwen15-v6/outputs/qwen15screen/` |

Training/calibration/assessment are 1,024/128/128 official-training questions.
The original 512-question test was evaluated once under its frozen protocol, then
reused for explicitly exploratory analysis. Development data were reused for both
follow-ups. The reserved fresh 512 questions exclude all 3,104 earlier distinct
queries and remain unused. No new final-test evaluation belongs to the final
writing/figure-validation workflow.

All intervals use 10,000 paired query bootstrap samples. Original confirmation
uses seed 4901 and Bonferroni-8 familywise intervals; post-test exploration uses
4903 and unadjusted pointwise intervals; the benefit gate uses 4904 and
Bonferroni-16. The architecture audit uses 4905 with selected-family Bonferroni-40
and matched-budget Bonferroni-9 as separate families. Its retrospectively
highlighted .5-cut intervals are pointwise, unadjusted and not confirmatory.
Zero differences for identical endpoint mixtures do not establish signal equivalence.

## Preservation and local archives

Code, pinned inference settings, frozen evaluations and saved numerical results
are retained. Neural weights, credentials, raw per-query caches and temporary
outputs remain outside Git under ignored `outputs/`. A fresh clone can inspect
summaries and view all figures/slides, but reproducing per-query analyses requires
the original private local archives.

The benefit archive is `outputs/benefit-results(1).zip`, SHA256
`9488d43cd27cab50e19b860ca38eaf7d1f75924d3cafa44f96c8d042ea1c6228`, extracted
at `outputs/kaggle-benefit-development-v1/`. It contains paired predictions,
features, costs, manifests, portable predictors, scores and decisions. Its stale
`run_failure.json` records an initial CPU packaging error after collection;
lossless aligned feature restoration and successful recovery are documented in
`benefit_provenance.json`. The final `analysis/development.json` is authoritative.
The GPU session was stopped after the complete ZIP was preserved and verified.

The benefit protocol was committed at `5f2149b` before missing training inference;
packaging at `8506d5a` preserved the method. The architecture protocol was committed
at `35cdebd` before replay. Its later ID-key repair left calibration choices and
rules unchanged. Every audit output is development-only; its .5 example was
highlighted after replay and was never selected for confirmation.

Warm component replay is not deployment latency. Ordinary QA prefill/decoding
and retained-KV/interposition costs were not measured separately; the audit's
prefill fractions are hypothetical. The proposed development timing measurement
was not run. No new GPU calls, fitting or policy searches are needed for final
manuscript preparation.

## Collaborator setup and Kaggle execution

Use the [README setup](../README.md#collaborator-setup) for a fresh CPU environment.
Python 3.11 was tested; the GPU extra is for a supported CUDA/Linux environment,
not the Mac CPU audit. Dependencies are declared in `pyproject.toml`; some versions
float, so a successful install is not an exact historical inference environment.

The final pair's pinned [1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
and [14B](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) model configurations and
weight-shard metadata, and pinned
[TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa) parquet metadata,
were accessible with `token=False` on 10 November 2026. No gated Llama access or
HF token is required. This check did not download weights or run inference.

There is **no tracked `.ipynb`**. The committed generator embeds only committed
source and required frozen settings into an output-free upload notebook:

```bash
python scripts/kaggle_bundle.py --owner YOUR_KAGGLE_USERNAME --mode qwen15screen --out outputs/kaggle-own-account
```

Replace the owner explicitly; the historical generator default names the author.
Upload `outputs/kaggle-own-account/experiment.ipynb` to a new private notebook in
your account, enable Internet and T4×2, and complete Kaggle's account/GPU eligibility
requirements. This mode has no attached private input dependencies. It runs an
automatic eight-query engineering smoke, then the historical 1,024/256 development
screen; **Run All spends GPU quota**. It does not evaluate the original or fresh
final test. These are reproduction instructions, not additional planned research.
The `pilot` default uses the earlier Phi pair and 32-token settings.

For CLI upload, authenticate with your own account using `kaggle auth login`, then
`kaggle kernels push -p outputs/kaggle-own-account`; alternatively use the browser.
See [official authentication instructions](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication).
Keep tokens in the platform secret store or user credential directory, never in
source, notebook cells/outputs or shared archives. Model/data access here is anonymous.
Save a completed Kaggle version and download its outputs before ending a session.
Resume only with the matching cache and explicit input attachment; a private
`--resume-kernel` source requires a separate Kaggle permission grant.

Exact historical inference needs the runtime recorded in
[`qwen15_methods.json`](../results/qwen15_methods.json), not merely `.[gpu]`:
Python 3.13.15, torch 2.11.0+cu128/CUDA 12.8, transformers 4.46.3,
accelerate 1.1.1, bitsandbytes 0.49.2, tokenizers 0.20.3, numpy 2.1.3,
datasets 3.1.0, pyarrow 18.1.0, pandas 2.2.3 and safetensors 0.8.0, on Tesla T4.
The notebook installer does not pin the Kaggle base image or the floating torch
dependency. Check and match this manifest before attempting a historical rerun;
do not weaken the frozen evaluator's version/hash guards to accept a new runtime.
Hardware timing will still vary. `--mode confirmatory` can generate the historical
test runner without a private cache, but it spends inference and touches the
already-consumed original test; it is unnecessary for inspecting the saved result.
No new final-test evaluation was performed during this audit.

For an independent engineering smoke on CUDA, a new directory and explicit final
pair avoid the legacy defaults:

```bash
aca prepare --out outputs/my-smoke --train 8 --validation 8 --test 0 --seed 4902 --revision 0f7faf33a3908546c6fd5b73a660e0f8ff173c2f
aca collect --out outputs/my-smoke --small-model Qwen/Qwen2.5-1.5B-Instruct --strong-model Qwen/Qwen2.5-14B-Instruct --quantization 4bit-fp32 --max-new-tokens 64 --answer-stop first-line --stages small ic strong --splits train validation
```

This is a mechanics check, not a new accuracy estimate. For later experiments,
commit source changes before bundling, use a separate branch/output/protocol,
and keep the existing frozen artifacts intact. `src/aca/routing.py` contains
policy actions and cost replay; `benefit.py` and `pregeneration.py` contain the
extensions; `collect.py`, `models.py` and `storage.py` handle inference/resume.
Tests exercise these interfaces with synthetic caches and tiny random CPU models.
Changing inference settings or code requires a new cache; existing manifests
deliberately reject incompatible reuse. New scientific claims require new
development/test separation, not reuse of the consumed original test.

### Private files needed for full per-query replay

GitHub collaborator access does not share the author's private Kaggle resources.
No public cache download is configured. Ask the owner to share only the relevant
archives/directories, preserving their relative paths:

| Required replay | Separately shared material |
|---|---|
| Original confirmation and post-test mixture | `outputs/kaggle-confirmatory-v7/outputs/confirmatory/`, including scores, manifests, component records and decisions; archive hash is in `qwen15_confirmatory_provenance.json` |
| Original development and pre-generation audit | `outputs/kaggle-qwen15-v6/outputs/qwen15screen/` and `outputs/pregeneration-audit/`, including its results and assessment decisions |
| Benefit development STOP | Complete `outputs/kaggle-benefit-development-v1/` or the verified benefit ZIP identified above; includes restored features and portable analysis models |
| Exact reservation / benefit bundle reconstruction | `outputs/benefit_inputs/excluded.json`, `reservation.json` and `fresh_test/queries.json`; preserve the reservation hashes and leave the fresh test unused |

Use a private file transfer or separately grant access to the matching Kaggle
version/data artifact. Validate archives against the tracked provenance hashes;
do not share the entire local `outputs/` tree, credentials or weights. The benefit
bundle generator depends on these caches and cannot work from a clone alone.
The exact fresh reservation cannot safely be recreated by scanning an empty
clone's `outputs/`; share the existing manifests instead.

After restoration, offline mixture replay can write a separate result directory:

```bash
python -m aca.exploratory_mixture --cache outputs/kaggle-confirmatory-v7/outputs/confirmatory --out outputs/mixture-reproduction
python -m aca.benefit_analysis --root outputs/kaggle-benefit-development-v1/development --analysis outputs/kaggle-benefit-development-v1/analysis --out outputs/benefit-review --original-methods results/qwen15_methods.json
```

These commands need no GPU and do not select a new policy. Do not overwrite the
tracked numerical summaries with a rerun.

## Viewing and regenerating deliverables

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/build_report_figures.py
```

The figure builder reproduces the original-confirmatory, exploratory-preselection,
correctness/rescue, break-even and benefit-development pairs from preserved
summaries. The sixth pair, `pregeneration_audit.png/.pdf`, is also tracked; its
original builder is `scripts/report_pregeneration_audit.py`. That script requires
the original development cache and recomputes read-only diagnostics, without
fitting or selecting policies. It was not rerun for the manuscript consolidation.

The presentation builder edits the existing deck while retaining its dimensions,
style, original routing diagram and original native charts. Original chart XML
and embedded workbooks are preserved from the source after import; the new audit
chart uses the saved diagnostic values. It uses the Codex
bundled Node/artifact-tool/Python runtime and installed Presentations skill:

```bash
RUNTIME_NODE_MODULES=/absolute/path/to/runtime/node_modules \
RUNTIME_PYTHON=/absolute/path/to/runtime/python3 \
PRESENTATIONS_SKILL_DIR=/absolute/path/to/presentations/skill \
/absolute/path/to/runtime/node scripts/build_research_presentation.mjs outputs/reproduced/COMP4901B_research.pptx
```

The output path must be new. The existing presentation is the default source;
an optional second positional argument selects a preserved source copy. Private
source copies, slide renders and validation receipts stay in `outputs/report-build/`.
After validation, the final writing workflow replaces the existing deliverable
with the validated file. Charts/tables remain native and editable. Chart data uses
12 significant digits for workbook portability; analytical JSON retains full precision.
Structural/import checks and visual review do not establish native PowerPoint or
Google Slides compatibility. Check any course-specific presentation format/timing
requirements before submission.

## Collaborator audit, 10 November 2026

At `efc0fcc`, the fetched remote exposed one branch (`main`), no tags and 29
reachable commits. Gitleaks 8.30.1 scanned all-refs history, all 175 unique
historical file blobs, the 85 tracked files, and 8,978 pre-existing local output
files, including generated notebooks/archives (archive depth four, decode depth
five). Additional checks covered provider-token/password patterns, sensitive
filenames, credential-bearing URLs, personal absolute paths, extracted historical
PDF/Office text and local Git authentication configuration. **No genuine
credential or tracked personal absolute-path candidate was detected.** Public
account names, historical Kaggle links and the overridable notebook-owner default
remain; these are identifiers, not credentials.

A clean GitHub clone installed `.[dev]`, passed `pip check` and all 47 tests with
network/model downloads disabled during tests, verified frozen implementation
hashes, and rebuilt five figure pairs. Generated development/confirmatory notebooks
passed notebook-schema, Python-cell compilation, payload-hash and own-account
metadata checks. Their GPU cells were not executed. All six published figure pairs
are available in Git, but the sixth builder requires private development caches.
Tables and saved intervals can be inspected from tracked JSON; independent
per-query/bootstrap regeneration requires the separately shared records above.
Rebuilding the deck additionally needs the non-PyPI bundled presentation runtime
described above; viewing/editing the existing PPTX does not.

The credential assessment is a bounded **PASS**; independent Kaggle execution and
full result regeneration are **PARTIAL** pending GPU/runtime verification and cache
sharing; code extensibility passes the offline interface/tests check. This is not
proof of the absence of every possible secret, an OCR/encrypted-file audit, a
dependency-vulnerability audit, or coverage of inaccessible/deleted remote refs.
No history rewrite or credential rotation was indicated by these findings.

`.gitignore` excludes common credentials, notebook checkpoints, weights and private
archives. It does not protect already tracked files or forced adds. Before sharing
or committing, review staged filenames and run an installed
[Gitleaks](https://github.com/gitleaks/gitleaks) scanner with redaction:

```bash
git diff --cached --stat
gitleaks git --staged --redact=100 --ignore-gitleaks-allow .
gitleaks git --log-opts="--all" --redact=100 --ignore-gitleaks-allow .
```

Keep scanner reports under ignored `outputs/` and never paste raw suspected
credentials. If a future scan finds a real credential, notify its owner/provider,
revoke or rotate it, and agree on history cleanup separately; deletion alone does
not remove prior commits.
