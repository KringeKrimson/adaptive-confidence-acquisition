# Adaptive Confidence Acquisition for Cost-Aware LLM Routing

COMP4901B experiment: does acquiring query-level Internal Confidence (IC) only
near an ambiguous cheap-probe decision improve a Phi-3 → Qwen cascade?

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
aca collect --out outputs/research --stages small strong ic
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
