# D09: baseline and hybrid support workflows

This capstone connects actual saved D01-D08 artifacts. The default comparison has 32 independently authored synthetic requests, ten pipeline variants, and a live corpus update. Every variant executes every request; rejected and escalated requests remain in the task-success denominator. See [PROGRESS.md](PROGRESS.md) for the current execution status; smoke runs are not a completed benchmark.

## Setup and component selection

Complete the CPU, retrieval, language-model and RLVR walkthroughs first. Install the root, MLX, retrieval and RLVR environments from their separate lockfiles. The integration controller runs in the root environment and starts local JSON-line workers to isolate incompatible Transformers versions. MLX needs Metal access. Retrieval and RL query workers use CPU; the larger teacher uses MLX on this Apple Silicon host.

The selected component run files under `artifacts/validation/selected_*_runs.json` identify the measured CPU and structured runs. The singular distillation and retrieval selector files use `selected_*_run.json`. Supply the actual completed RLVR run explicitly. All artifacts are checksum-verified before the full integration run. A smoke run may explicitly use a smoke RLVR checkpoint to test wiring; a full local comparison refuses it.

```bash
.venv/bin/ml-demo prepare hybrid
.venv/bin/ml-demo run hybrid --profile smoke \
  --source-run artifacts/rlvr/<completed-run>
.venv/bin/ml-demo run hybrid --profile local \
  --source-run artifacts/rlvr/<completed-run>
```

These commands create new artifact directories without changing component weights. The 12 routing calibration requests and 32 final scenarios are frozen separately in `data/generated/hybrid-v1`. The runtime accepts only `public` request fields. Expected outcomes, query ASTs and hidden fixture seeds are passed only to the evaluator after execution.

## What happens to a request

1. D06's actual ONNX model scores the numerical metadata using its frozen validation threshold. A blocked request stops here. Viewer requests for an update fail explicit authorization before language-model execution.
2. D05's saved tree predicts risk for the selected held-out synthetic transaction. D07 recomputes TreeSHAP contributions. These values are informational; risk accuracy and attribution coverage are separate from support-task success.
3. D01 routes the user's text. Invalid output, an explicit escalation or sequence confidence below the calibration-selected threshold calls the local 14B teacher. Confidence is the geometric mean probability of emitted tokens; it is not calibrated correctness. The small 12-case calibration set cannot establish reliable coverage on arbitrary inputs.
4. An escalation calls the teacher to ask one bounded clarification question. Other branches search the D08 corpus. The default uses the validation-selected tuned bi-encoder without a reranker, because D08's tuned reranker hurt the complete retrieval pipeline. Retrieved resource, version and operation must match the explicit caller context.
5. Reference requests return bounded facts parsed from the retrieved page. Queries use either D02's tuned generator or D03's RL policy according to the caller's explicit backend choice. Device requests use the D04 validation-selected adapter and a fresh simulator.
6. All generated tool payloads pass external schema/range/authorization gates. If the bounded typed parser can derive a query specification from the user's wording, public execution fixtures check semantics too. This check is explicitly unavailable for unsupported wording. Invalid student payloads trigger a teacher repair attempt; failed repairs are recorded as rejections.

A valid device command can still misunderstand the user. The pipeline does not claim that syntax or bounds establish intent. Clarification grading checks the chosen branch and question schema, not a human judgment of usefulness. No physical device, customer record or external support service is modified.

## Comparisons and dynamic knowledge

| Variant | Replacement |
| --- | --- |
| `baseline` | Prompted teacher, BM25, untrained hard rate ceiling, fixed risk prior and the same deterministic tool gates |
| `hybrid` | All measured components above |
| `no_screen` | Hard rate ceiling replaces D06 |
| `no_router` | Teacher replaces D01 |
| `no_retrieval` | Untuned bi-encoder replaces the tuned D08 bi-encoder |
| `no_schema` | Teacher replaces D02 on schema-backend requests |
| `no_rlvr` | The identical fused SFT initialization replaces RL on RL-backend requests |
| `no_device` | Teacher replaces D04 |
| `no_tabular` | Fixed untrained risk prior replaces D05, so feature attribution is inapplicable |
| `no_attribution` | Keep D05 predictions, omit D07 attribution |

After the comparisons, the actual hybrid retrieves an invoice v2 page, answers from it, edits one response-field fact in the in-memory corpus, reindexes embeddings and executes the pipeline again. `knowledge_update.json` preserves both complete traces, changed corpus hashes and unchanged component-manifest hashes. The changed response establishes that retrieval supplied the new fact; it does not show that model weights learned it.

## Reading and replaying the evidence

`predictions.json` and append-only `traces.jsonl` retain actual stage inputs, raw model outputs, model identities, timing, gates and fallback reasons. `metrics.json` includes all-request task success, false rejection, fallback, wrong-version retrieval, invalid gate rejections, latency and separate risk/attribution measures. `startup.json` separates shared worker initialization from warm requests. Worker RSS is cumulative and shared across variants, so do not sum duplicated component entries or treat them as isolated per-variant memory.

```bash
.venv-repro/bin/ml-demo evaluate hybrid --run-dir artifacts/hybrid/<run> \
  --output-dir artifacts/validation/hybrid-replay --clean-workers
```

The root and all three model `.venv-repro` environments must already have been independently installed from their locks. Replay verifies component/dataset checksums and regenerates every request for every variant. It compares raw output text, model identities, probabilities, rankings, grades and the document update; nondeterministic timing/RSS are excluded. This is an artifact reload and inference reproduction, not an independent retraining replication.

For one fresh request, save its public JSON fields without an `expected` key and run:

```bash
.venv/bin/ml-demo run hybrid --run-dir artifacts/hybrid/<run> \
  --request-json /path/to/public-request.json --variant hybrid
```

The [comparison UI](COMPARISON_UI.md) provides this same workflow with explicit request context and per-stage evidence. Runtime defaults are chosen from component validation, not from final capstone results. One synthetic, single-host pilot does not establish production quality or precise tail latency. No hosted API fees were incurred; energy and hardware depreciation were not measured.
