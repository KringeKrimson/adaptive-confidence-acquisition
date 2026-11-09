# Reproducing the completed COMP4901B study

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
Google Slides compatibility. Author names and course-specific format/timing are
the only remaining submission details.
