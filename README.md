# Adaptive Confidence Acquisition for Cost-Aware LLM Routing

COMP4901B experiment: does acquiring query-level Internal Confidence (IC) only
near an ambiguous cheap-probe decision improve a small-to-strong cascade?
The current prospective pair is Qwen2.5-1.5B → Qwen2.5-14B. It passed the frozen
development screen; adaptive IC's benefit is not yet established. Historical
Phi/Qwen pilots and the unchanged failed Phi→Qwen-14B screen are retained below.

## Run

```sh
python -m pip install -e '.[gpu,dev]'
aca prepare --out outputs/pilot --test 32
aca collect --out outputs/pilot
aca summary --out outputs/pilot
```

CUDA and Internet are required. Both official model repositories are accessed
anonymously; no Hugging Face token, approval, or notebook secret is needed.
The pilot uses `microsoft/Phi-3-mini-4k-instruct` and
`Qwen/Qwen2.5-7B-Instruct`, greedy decoding, 32 maximum answer tokens,
batch size one, and NF4 4-bit weights (FP32 compute and unquantized modules by default). Both models run sequentially
on GPU 0 to fit a 16 GB Kaggle GPU. This quantized experiment is **not** a
reproduction of the papers' full-precision results. `--quantization none` is
available on sufficiently large GPUs.

The original Llama-3.1-8B-Instruct target was replaced **before any research
predictions were collected**, because access was denied. Qwen2.5-7B-Instruct
is an openly accessible Apache-2.0 model with 7.61B parameters, keeping the
strong-model size close to the proposed 8B model
([official model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)).
The hypothesis remains selective confidence acquisition; this experiment makes
no claim to reproduce the paper's Phi-3/Llama cascade. "Strong" is a designated
role, not an assumed accuracy result: report the paired pilot correctness and
validation improvement/degradation counts before expanding.

For a prespecified larger-capacity robustness experiment, use
`aca collect --out outputs/research-qwen14b --strong-model Qwen/Qwen2.5-14B-Instruct`
after preparing that directory's data with the same sampling/exclusions. The
14B model is also publicly accessible and appears in the IC paper's model suite;
it changes the cost gap and is a separate experiment, not a replacement chosen
after inspecting test results. Model identity and revisions are stored in caches;
switching model pairs in an existing cache is rejected.

For Kaggle, after committing source:

```sh
python scripts/kaggle_bundle.py --owner sushobhanrai
kaggle kernels push -p outputs/kaggle-bundle
kaggle kernels status sushobhanrai/adaptive-confidence-acquisition-pilot
kaggle kernels output sushobhanrai/adaptive-confidence-acquisition-pilot -p outputs/downloaded
```

Alternatively import `outputs/kaggle-bundle/experiment.ipynb` into Kaggle.
Enable Internet and select a GPU. No notebook secrets are required.
The generated notebook only extracts the source snapshot, installs the package,
and starts `kaggle/runner.py` in a fresh Python process. This prevents retained
notebook imports from mixing NumPy binary versions after installation, including
on Kaggle's Python 3.13 image. It embeds no credentials or model weights and
does not need GitHub credentials for the private repository. Use **Save & Run
All** to retain outputs after the session.

Each query/stage is an atomic JSON checkpoint; rerunning skips complete work.
To resume another Kaggle session, attach a previous notebook's saved output
and set `ACA_RESUME_FROM` to its `outputs/pilot` (or `outputs/research`) directory.
The bundle generator also accepts `--resume-kernel SLUG`. The source/runtime,
model revisions, prompts, dataset manifest and inference settings must match;
incompatible caches fail rather than silently mixing runs. Atomic writes survive
process interruption on retained storage; Kaggle must retain/download that storage
to survive destruction of the whole session. Use bounded jobs and saved versions.

After inspecting the pilot's predictions/truncation rate:

```sh
aca prepare --out outputs/research --train 1024 --validation 256 --test 512 --exclude-manifest outputs/pilot/queries.json
aca collect --out outputs/research --stages small --splits train validation test
aca collect --out outputs/research --stages strong ic --splits validation test
aca analyze --out outputs/research
```

Or build the Kaggle bundle with `--mode research`. This is a separate dataset
manifest; inspected pilot IDs and normalized questions are excluded from all
research splits. The research Kaggle runner reserves the deterministic pilot
before sampling, including in a fresh session. If changing pilot seed or dataset
revision, carry its actual manifest into that session as well.

## Method and accounting

- Closed-book TriviaQA `rc.nocontext`, pinned dataset commit. A seeded streaming
  shuffle (10,000 buffer) samples the data. Official training data is split by
  hashed normalized question into project train/validation; official validation
  supplies project test. Duplicate IDs/normalized questions are removed across
  selected splits. This is a deterministic buffered sample, not uniform sampling
  of the entire dataset. Exact match against all supplied aliases is primary;
  token F1 and token-limit flags are also recorded. No substring correctness.
- The cheap correctness probe uses a standardized logistic regression on the
  final normalized hidden state at the last QA prompt position. A hook captures
  it during the ordinary generation prefill; there is no separate probe forward
  pass. Probe training uses train labels only; regularization is selected by
  validation log loss. Batch-one CPU prediction time is measured and charged.
- All cascades use an **answer-first protocol**: ordinary small QA is already
  computed before deciding whether to defer. Small-only and strong-only each
  pay only their own QA cost. Always-IC pays small QA + IC + any strong QA, but
  does not pay for an unused probe. This is an acquisition ablation under the
  same protocol, not a reproduction of a pre-generation IC-first cascade.
- IC follows the authors' released default API and implementation at
  [`5c15b32`](https://github.com/tigerchen52/query_level_uncertainty/tree/5c15b32a6f9f36688098d0a749cb68bad11e9f23),
  checked against [Chen et al.](https://arxiv.org/html/2506.09669v4), §3.2,
  equations 1–4. A **separate** Yes/No self-evaluation prompt yields a matrix of
  binary Yes-vs-No probabilities. Gaussian weights `exp(-alpha*distance²)` are
  normalized separately over layers and final tokens; the center is last layer,
  last token, `alpha=1`, and Phi-3 uses the last **three** tokens, as in the
  authors' `eval_ql_uncertainty/run_ic.py`. We preserve the released code's
  embedding state and final normalized hidden state and do not add intermediate
  RMS normalization. Selecting just two unembedding rows is algebraically the
  same binary projection; softmax uses float32 for stability. Token IDs are
  obtained from the actual tokenizer and single-token Yes/No is asserted.
- The released API's **no-demonstration** prompt is used. The paper describes
  positive/negative demonstrations, so this is explicitly the released-default
  IC variant, not an exact replication of its benchmark prompt/results. The
  complete prompt code and source version are recorded. The ordinary QA prompt
  is also a new fixed short-answer prompt. Related routing context:
  [Dekoninck et al., A Unified Approach to Routing and Cascading](https://arxiv.org/abs/2410.10347).
- Adaptive decisions acquire IC when `abs(p_cheap - threshold) < width`; acquired
  decisions use an IC threshold. Cheap/IC thresholds and widths are selected
  exclusively on validation by `accuracy - lambda * relative_service_seconds`.
  Test data evaluates the frozen policies; it does not choose thresholds or the
  best operating point. Both models' correctness can be zero or one independently.
- The random IC control matches the adaptive policy's **IC acquisition seconds**,
  not its total cascade cost. A seeded random permutation selects a prefix;
  its boundary query is fractional to match the budget exactly in expectation.
  This is an explicitly retrospective timing-oracle control using cached runtimes,
  not a deployed online schedule or a single realized randomized trial. Random
  routing thresholds use validation only; multiple seeds and paired query
  bootstrap intervals are reported. Compare actual total cascade costs as well.
- Reported cost is synchronized warm batch-one **additive service-time replay**:
  small QA + cheap scoring + acquired IC + deferred strong QA. IC includes its
  full prompt pass, hidden-state extraction, two-token projection and aggregation.
  Input/output tokens and model/forward calls are saved separately per stage.
  Loading, warmup, probe training, and offline cache collection are separate
  experimental overhead. Replay is not measured interleaved deployment latency;
  a deployment would require resident models or additional model-swap costs.
  Offline collection computes IC/strong predictions even when a simulated policy
  would not; replay only charges computations that policy actually requires.

Outputs include `queries.json`, immutable inference settings/model revisions,
per-stage per-query records, `pilot_summary.json`, `predictions.json`, portable
`probe.json`, per-query CPU probe times in `probe_scores.json`, validation-selected
policies, `analysis.json`, `accuracy_cost.csv`
and `accuracy_cost.png`. Experiment outputs, weights, tokens and environments
are ignored by Git.

## Tests

```sh
python -m pytest -q
```

Tests include exact-match evaluation, split leakage guards, numerical IC
equivalence, incremental cost accounting, random acquisition budgets,
configuration compatibility, and real forward/generation integration with
tiny random Phi-3/Qwen/Llama models. Tiny-model or synthetic tests are engineering
checks, **not research accuracy measurements**.

## First saved Kaggle pilot (diagnostic, not hypothesis evidence)

Saved version [361392122](https://www.kaggle.com/code/sushobhanrai/adaptive-confidence-acquisition-pilot?scriptVersionId=361392122)
ran commit `2277cb9` on a Tesla T4 in 334.6 seconds, including installation,
model downloads, loading and warmup. All 32 paired predictions were downloaded.
Phi-3: 16/32 exact matches, 0.449 s/query, no token-limit hits.
Qwen-7B: 11/32, 0.722 s/query, four token-limit hits; two rescues and seven harms.
Several Qwen answers contain anomalous `!` tokens and malformed strings.
These are unvalidated numerical results, not evidence against the acquisition
hypothesis. The next diagnostic repeats the same reserved pilot with a finite-logit
guard and compares NF4 FP16 against NF4 FP32 compute and unquantized-module dtype
(`--quantization 4bit-fp32`). The guard's synchronization cost is included in
subsequent measured generation times. It prevents NaN/Inf logits from silently
becoming greedy token IDs; it does not repair or filter answers. A failed
numerical check stops collection rather than caching an invalid prediction.


The numerical diagnostic [version 361396801](https://www.kaggle.com/code/sushobhanrai/adaptive-confidence-acquisition-pilot?scriptVersionId=361396801)
completed in 443.7 seconds at commit `88a3a0f`. The finite-logit guard rejected
FP16 Qwen inference. NF4 with FP32 compute/modules completed 32 paired QA calls
and 32 IC estimates: Phi-3 16/32, 0.611 s/query; Qwen-7B 14/32, 0.875 s/query;
no truncations. Qwen rescued two errors and introduced four. Mean IC cost was
0.397 s/query (65% of small QA time); all saved IC matrices were independently
reaggregated. FP32 is now the default; legacy FP16 remains an explicit diagnostic
option `--quantization 4bit`. This pilot is too small for model-quality conclusions,
but does not establish the intended strong-model advantage. The next reserved-pilot
check uses the prespecified Qwen-14B alternative (`--mode pilot14b`); no research
thresholds or probes have yet been trained on real model outputs.


The larger-model diagnostic [version 361463736](https://www.kaggle.com/code/sushobhanrai/adaptive-confidence-acquisition-pilot?scriptVersionId=361463736)
completed in 584.4 seconds at commit `e62e630`, using the identical reserved cohort.
All 64 QA labels, 32 feature vectors and 32 IC matrices were downloaded and checked.

| Stable NF4 FP32 pilot | Exact match | Warm mean seconds/query | Token-limit hits |
| --- | ---: | ---: | ---: |
| Phi-3 (14B-pair run) | 16/32 (50.0%) | 0.613 | 0 |
| Qwen-7B (previous run) | 14/32 (43.8%) | 0.875 | 0 |
| Qwen-14B | 15/32 (46.9%) | 1.894 | 0 |

Qwen-14B rescues five Phi errors and introduces six; a correctness-oracle router
would reach 21/32, an unattainable diagnostic upper bound. Mean IC cost in this
run was 0.351 s/query. These are single-pass warm timings, not repeated latency
benchmarks. Neither candidate yet demonstrates the intended aggregate accuracy
advantage. All these questions are reserved diagnostics, not research test data.
The next methodological checkpoint is whether to assess Qwen-14B on development
data before proceeding to the held-out hypothesis experiment.

A concrete development-only runner is ready:

```sh
python scripts/kaggle_bundle.py --mode viability14b --out outputs/kaggle-viability-bundle
```

It reserves the pilot, prepares the 1024/256/512 manifest in `outputs/research14b`,
collects small QA/features on train+validation and strong QA/IC on validation,
and writes `viability.json`. It performs no test inference and reads no test
predictions. Its probe C and one routing policy are selected only on a fixed calibration
half of validation. The other half assesses routing viability. These remain
development diagnostics, not final-test hypothesis evidence.
The `research14b` mode can subsequently extend the same cache to test and run
all baseline comparisons; attach the saved development output when resuming.
Both research runners skip strong QA and IC on training examples because those
calls are mathematically unused. The ordinary small-model training features and
correctness labels remain necessary. `collect --splits ...` schedules subsets
without changing the per-query inference configuration or overwriting results.
The equivalent local development readout is `aca viability --out OUTPUT_DIRECTORY`.

**Executed:** paired GPU pilots, IC diagnostics, and the development viability
screen reported below, including real probe training and calibration selection.
**Implemented but not yet executed on real final-test data:** adaptive policy
selection and held-out baseline/accuracy–cost comparisons. Synthetic end-to-end
checks cover those code paths; they are not research evidence.


## Frozen viability criteria (before the 1024/256 run)

User-authorized Qwen-14B development check. Rules are versioned in
`src/aca/viability.py`, written to `viability_protocol.json` **before** any query
prediction, and bound to the dataset manifest. Later changes are rejected.
The dataset/model commits are fixed to those used in the verified pilot.
The runner checks that no final-test prediction exists before and after the run.

All gates must pass:

1. On all 256 validation queries, Qwen-14B exact match minus Phi exact match
   must be at least **0.05** (13 or more net additional correct answers).
2. Rescue at least **20%** of Phi errors; introduced errors must be no more
   than **50%** of rescues. Zero rescues fails this gate.
3. One calibration-selected routing rule must pass the assessment half:
   accuracy improvement over Phi of at least `max(0.03, 0.5 * max(Qwen-Phi, 0))`,
   with warm additive service cost at most **0.80** of Qwen-only cost. This
   requires both at least three percentage points of improvement and recovery
   of at least half the strong-model advantage where positive. The same
   constraints must hold in calibration for the rule to be eligible.

The 256 query IDs are sorted by SHA256(`4901:query_id`). First 128 form
calibration; last 128 form assessment. Logistic weights use the 1024 training
labels only. Choose C from `[0.01, 0.1, 1, 10]` by calibration log loss.
Enumerate cheap-only and always-IC policies at thresholds `[-0.01, 0, 0.05,
..., 1, 1.01]`. Among eligible calibration policies choose greatest accuracy,
then lowest mean cost; fixed enumeration breaks remaining ties. Freeze this
one policy before assessment. No adaptive acquisition policy is tuned here.
Cheap routing pays small QA + CPU probe + deferred strong QA; always-IC pays
small QA + IC + deferred strong QA. Loading/warmup/training are separate.
If no calibration policy is eligible, retain a diagnostic policy and fail gate 3.

Report every gate, both validation partitions, rescue/harm counts, truncations,
per-query scores/times, probe weights and a 10,000-repeat paired-bootstrap 95%
interval (seed 4901) for the model accuracy difference. Intervals are descriptive,
not an extra significance gate. These effect-size thresholds are project screening
choices, not claims of a published standard. Single-pass timings remain estimates.
No optional stopping, criterion relaxation or repeated pair changes to obtain a
pass. A failed check triggers a model-pair recommendation before any full hypothesis
run; it does not by itself disprove selective IC acquisition.

## Development viability result: FAIL (2026-10-23)

[Kaggle version 361483712](https://www.kaggle.com/code/sushobhanrai/adaptive-confidence-acquisition-pilot?scriptVersionId=361483712)
completed in 1972.6 seconds using source commit `f356c74`, committed before
collection. The frozen protocol/dataset hash is
`d5501a68905e4ff6f1939980ac8d61fc2f053ca4f63c2413730931a0075e39b9`.
The criteria above were unchanged. Aggregate results are saved in
[`results/viability14b.json`](results/viability14b.json).

Trained the cheap probe on 1024 training queries; C=0.01 was selected on the
128-query calibration half. The other 128 validation queries assessed the fixed
rule. Downloaded and verified all 1280 Phi QA/features, 256 Qwen QA and 256 IC
records, including labels, finite features, IC aggregation, portable probe scores,
cache hashes, partition membership and timing replay. Reserved pilot IDs/questions
do not overlap. The final 512-query test has **zero inference records**.

| Frozen gate | Observed validation result | Verdict |
| --- | --- | --- |
| Qwen advantage ≥5 percentage points | Phi 101/256 (39.45%); Qwen 160/256 (62.50%); +23.05 points | PASS |
| Rescue ≥20% of Phi errors; harms ≤50% of rescues | 74/155 errors rescued (47.74%); 15 harms (20.27% of rescues) | PASS |
| Recover ≥half the accuracy advantage, at ≤80% of Qwen-only cost, on calibration and assessment | No calibration candidate eligible; fixed diagnostic rule has zero assessment gain | FAIL |

The descriptive paired-bootstrap 95% interval for Qwen minus Phi is
[16.41, 29.69] percentage points. Mean validation service times were Phi 0.700 s,
Qwen 1.661 s, and additional IC 0.409 s/query. Thus answering with Phi first
already consumes 42.2% of Qwen-only time; always acquiring IC raises the base to
66.8% before any strong-model calls. These are single-pass additive timing
estimates, including required CPU probe scoring, not production latency benchmarks.

The budget-feasible diagnostic rule (cheap confidence <0.10 defers) was chosen
on calibration despite no eligible rule. It attained 58/128 (45.31%) at 1.273 s
(75.55% of Qwen-only cost) on calibration, below the required 50.78% accuracy.
On assessment it attained 52/128 (40.63%) at 1.154 s (70.49% of Qwen-only),
matching Phi and below the required 51.17%. It is a diagnostic, not a successful
selected policy. The least-cost calibration cheap rule meeting the accuracy target
cost 92.19% of Qwen-only; the corresponding always-IC minimum was 111.14%.
Assessment cheap/IC correctness AUROCs were 0.648/0.624.

The fixed 32-token cap was reached by 69/1024 Phi training answers, 16/256 Phi
validation answers and 1/256 Qwen answers; all were incorrect under the unchanged
strict alias exact-match rule. Some Phi outputs contain an answer followed by
unsolicited text. No post-hoc extraction, token-limit changes or rescoring were
used to turn this failure into a pass. Findings apply to this quantized prompting,
decoding and scoring configuration. The result does not establish that adaptive
acquisition itself fails, since that hypothesis experiment was not launched.

**Recommendation before a full experiment:** screen
[`Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
as small and retain Qwen2.5-14B-Instruct as strong. Qwen's official card identifies
the public Apache-2.0 model as 1.54B parameters. A cheaper first model is a
reasoned candidate for reducing the measured overhead; its speed, rescue pattern
and routing benefit are **unmeasured**, not established improvements. Changing
the small model requires new features/probe and explicit verification of the IC
transfer and tokenizer settings; Phi's confidence cache cannot be reused for it.
A new development protocol must be frozen before that check, with a matched
short-answer/stopping rule fixed in advance and the final test still unused.
Do not switch pairs repeatedly until a pass or promote development comparisons
to final-test hypothesis evidence. At this checkpoint no replacement run had been launched.

## Prospective Qwen-1.5B → Qwen-14B screen (pre-inference protocol)

The user authorized this second pair on 2026-10-23. `aca.prospective.PROTOCOL`
copies the original criteria without mutating them: identical sample counts,
three gates, probe C/threshold grids, calibration/assessment split rule, cost
definition and selection logic. Only the following necessary settings change:

- New small model pinned to `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`;
  the strong model keeps its original pinned revision.
- Fresh 1024 training + 256 validation queries from official training data,
  sampling seed 4902. Exclude the entire old development cohort and reserved
  pilot by ID and normalized question. This run has no final-test queries.
- Same zero-shot QA prompt, greedy decoding; stop at the first nonempty answer
  line or EOS with a 64-token ceiling. Apply the identical rule to both models.
  Strict alias EM; no reference-aware answer extraction. New timings include
  the stopping checks. Old predictions/labels/timings/verdict remain unchanged.
- Rebuild 1536-dimensional final normalized ordinary-QA prompt features and
  fit a new train-only logistic correctness probe. No Phi features/weights reused.
- Transfer the authors' Qwen six-final-token IC setting, alpha=1, no examples;
  verify tokenizer suffix and Yes/No IDs `[9454,2753]`. Qwen-1.5B was not in the
  paper's evaluated model set, so this is an explicit model-transfer variant.
  Include all 29 hidden states; preserve the released projection/aggregation.
- Both models retain NF4 double quantization with FP32 compute and unquantized
  modules, eager attention, batch one, sequential GPU0 on T4×2. No BF16 assumption
  or precision search based on observed accuracy. Record actual peak memory.

`--mode qwen15screen` freezes the protocol before an automatic separate eight-query
training smoke. Engineering checks require finite features/logits/IC, positive
timings, expected dimensions/IDs, GPU full-vocabulary/reduced-head IC equivalence,
no OOM and >256 MiB reserved-memory headroom. There is no smoke accuracy gate.
Success automatically advances to 1024/256 collection and the unchanged screen.
Per-query atomic checkpoints and the existing resume-input mechanism are retained.
Loads, warmups, projection verification and smoke are recorded separately from
warm per-query routing costs. The runner never collects the final test.

Offline diagnosis of the original failed screen is saved in
[`results/diagnosis_phi14b.json`](results/diagnosis_phi14b.json). Cheap correctness
AUROC was 0.721 overall / 0.648 assessment, but low cheap confidence predicted a
rescue at AUROC 0.493 overall / 0.425 assessment. Correctness is not escalation
benefit: benefit is strong correctness minus small correctness, including harms.
The original diagnostic rule escalated 74 queries: 16 rescues, seven harms,
51 unchanged outcomes, missing 58 rescues. A free label-and-timing oracle could
meet the validation target at 51.9% of strong-only cost; thus overhead alone does
not make the goal impossible. Calibration-only exploratory adaptive replay adds
one correct answer (0.78 points) over the best budget-feasible cheap rule, still
below the target. This coarse-grid difference does not establish an IC benefit:
a calibration-only randomized mixture of two cheap thresholds reaches 49.65%
at the same expected 1.329 s cost, versus adaptive IC's 46.09% (−3.55 points).
The mixture is descriptive and is not an original-screen candidate. No assessment
optimization or original policy changes were made.
First-line-only diagnostic rescoring finds ten extra Phi validation matches,
six among 16 cap hits; this motivates prospective stopping, not retroactive repair.

After a prospective PASS, `aca freeze-methods --out OUTPUT` retains the fitted
probe and selects the existing adaptive/cheap/always-IC grids and 20-seed random
control on calibration only. It binds the weights, grids, selected rules and
implementation hash in `methods_frozen.json`. `aca compare-development --out OUTPUT`
then reports all eight fixed operating points on the assessment half, including
incremental IC cost and paired intervals. Neither command collects or reads final
test predictions. A future final-test evaluator must consume this frozen artifact
without refitting on the combined development data.

## Prospective result: PASS; adaptive benefit remains unestablished

[Kaggle version 361511726](https://www.kaggle.com/code/sushobhanrai/adaptive-confidence-acquisition-pilot?scriptVersionId=361511726)
completed in 24m38s on T4×2, using pre-inference commit `cc15cdd` and frozen
protocol/data hash `765a1ebffc54e7d4c10d92be26cd6aa4000ad51f646bd29250b9731414f65b18`.
The automatic eight-training-query smoke passed and scaled without intervention.
All 1280 small QA/features, 256 strong QA and 256 IC records were downloaded
and verified; separate smoke outputs contain eight records per stage. No old
development/pilot overlaps and no final-test queries or inference records.

| Frozen criterion | Measured result | Verdict |
| --- | --- | --- |
| Strong advantage ≥5 points | Small 82/256 (32.03%); strong 152/256 (59.38%); +27.34 points | PASS |
| Rescue ≥20%; harms ≤half rescues | 76/174 errors rescued (43.68%); six harms (7.89% of rescues) | PASS |
| Required gain at ≤80% strong-only cost in both halves | Calibration 47.66% vs small 27.34%, at 77.88% cost; assessment 48.44% vs small 36.72%, at 77.99% cost | PASS |

The calibration-selected rule defers when cheap confidence <0.15. Assessment
gain is 11.72 points against a required 10.16; cost is 1.273 s/query versus
strong-only 1.632 s. Three calibration candidates were eligible. The model
accuracy-gap bootstrap95 is [21.09,33.59] points. C=0.01; assessment correctness
AUROC is 0.789 cheap / 0.697 IC, whereas rescue-prediction AUROCs are 0.563/0.560.

Mean whole-validation times: small QA 0.338 s, strong QA 1.629 s, additional IC
0.134 s. Small QA+IC is 28.9% of strong-only time before escalation. Peak reserved
GPU0 memory was 1.62 GiB small / 12.66 GiB strong, against 14.56 GiB available
per T4. Collection ran sequentially on GPU0; warm two-model service assumes one
resident model per T4. These additive service timings do not measure a live
two-GPU deployment's queue/communication latency. Loads, warmups, smoke and IC
projection checks are separately recorded. Actual-GPU full-vocabulary versus
reduced IC projection error was 9.54e-7 in both checks. No small answer reached
the 64-token cap; two strong validation answers did.

After PASS, committed selector `2a34b21` froze eight operating points and the
portable probe in [`results/qwen15_methods.json`](results/qwen15_methods.json),
hash `59ef6e267603260190ade5468e7e54d248c25ba11ababec32cdb276d220fa8ed`.
Only calibration rows were used; NaN-corruption tests confirm assessment values
cannot affect selection. All fixed choices were then replayed on assessment,
with all costs included and random acquisition matched in expected IC seconds.
The assessment replay is a development check, not final-test hypothesis evidence.

At penalties 0–0.2 the selected adaptive rule uses IC for 4.69% of assessment
queries: 56.25% accuracy at 1.833 s, versus cheap-only 55.47% at 1.892 s, but
strong-only is 57.03% at 1.632 s. At penalty 0.4, adaptive gives 40.63% at
0.587 s versus random IC's 42.63% at 0.812 s; equal IC cost does not imply equal
total cost. The 0.8 and 1.6 points acquire no IC. Every adaptive-minus-random
accuracy interval includes zero. These outcomes do not establish an IC advantage;
the pair's viability PASS must not be confused with a positive hypothesis result.

Aggregates: [`qwen15_viability.json`](results/qwen15_viability.json) and
[`qwen15_development.json`](results/qwen15_development.json). Raw predictions,
features, IC matrices, timings, overheads and plots remain in ignored `outputs/`
and the immutable Kaggle output. The next experiment is evaluation of these
frozen choices on the reserved final test, without changing grids/probe/selection
after this readout. That final-test experiment has not been run.
