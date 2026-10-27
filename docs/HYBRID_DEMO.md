# D09: baseline and hybrid support workflows

Visual companions: [D09 notebook](../notebooks/D09_hybrid_capstone.ipynb). These plot the recorded evidence and include editable examples without loading models; see the [notebook guide](../notebooks/README.md).

This capstone connects actual saved D01-D08 artifacts. The default comparison has 32 independently authored synthetic requests, ten pipeline variants, and a live corpus update. Every variant executes every request; rejected and escalated requests remain in the task-success denominator. See [PROGRESS.md](project/PROGRESS.md) for the current execution status; smoke runs are not a completed benchmark.

## The learning question

Do the component improvements survive contact with an end-to-end task? This capstone teaches how to trace a result through routing, retrieval, generation and execution, and how to decide whether each extra stage earns its complexity.

## The concept: a pipeline has its own objective

A component can be correct on its own benchmark and still be ineffective in a system. A router sends the request to the wrong branch; retrieval returns the wrong API version; a valid query drops one predicate. Any one can prevent the user's intended outcome.

For an illustrative chain of five independent, necessary stages each succeeding with probability 0.95, joint success would be `0.95^5 ≈ 0.774`. Actual pipeline errors are dependent, branches differ and fallback changes the path, so this is a teaching calculation rather than a predictor of the lab's score. It explains why the capstone evaluates whole requests instead of averaging component accuracies.

~~~mermaid
flowchart TD
    A["Public request and caller context"] --> B["Screen and authorize"]
    B --> C["Risk and explanation"]
    C --> D["Student route with teacher fallback"]
    D --> E["Retrieve matching version and operation"]
    E --> F["Reference, query, device or clarification branch"]
    F --> G["External payload and available semantic checks"]
    G --> H["Response or simulated action"]
    H --> J["Hidden evaluator grades complete outcome"]
~~~

The evaluator's expected answer stays outside the runtime. The runtime may use public checks that would actually be available for a supported request. When wording is outside the typed parser's grammar, it does not get a hidden oracle as a substitute.

### Separate the outcomes

Primary task success answers whether the workflow satisfied the expected request, including appropriate rejection or clarification. Risk quality and attribution coverage are separate: the risk score is informational, so removing it cannot change task success by design. A component ablation with unchanged task score is not proof that its separate purpose has no value.

Fallback also needs context. Teacher use can fall without reducing generated tokens or p95 latency, because student work, repairs and extra stages still take time. Compare total requests, teacher calls, fallback reasons, token counts and elapsed time together.

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


## What was learned, and production tradeoffs

The [recorded hybrid](HYBRID_RESULTS.md) succeeded on 27/32 requests versus 19/32 for the baseline. Teacher usage fell from 93.75% to 43.75%, but p95 latency rose. Replacing the tuned retriever with the untuned one reduced success to 14/32; replacing the RL query model with its SFT initialization left 27/32 unchanged.

These results prioritize further retrieval work and question the marginal value of RL in this specific composition. They do not estimate universal component importance: there are only 32 scenarios, and ablations replace one component inside a fixed surrounding policy.

For production, add a stage only when its measured user outcome or operating benefit warrants the failure paths and maintenance it introduces. Stage traces should make rejection and fallback diagnosable. Circuit breaking, capacity limits, trusted identity, durable action state and deployment rollback would require additional implementation beyond these local workers.

## Try it and reason about the result

1. Inspect `capstone-20`: the query omitted an age condition. Find why the payload gate accepted the shape and why supported public semantic checking was unavailable. Explain what the hidden evaluator could detect that the runtime could not.
2. Compare `no_router` with `hybrid`. Their task scores match, but teacher usage differs. Explain the useful contribution without claiming an accuracy gain.
3. Compare screening false rejections with admitted abusive requests. Report both, instead of selecting whichever count makes the screen look better.
4. Read `knowledge_update.json`. Check that the response changed from `result` to `payload` while component hashes stayed fixed. Identify the knowledge source that changed.
5. Compare Brier scores only on aligned risk cohorts. Explain why the common all-negative cohort favors a low fixed prior and cannot establish useful discrimination.

Trace [hybrid.py](../src/ml_training/hybrid.py) for orchestration, [hybrid_policy.py](../src/ml_training/hybrid_policy.py) for gates, [hybrid_workers.py](../src/ml_training/hybrid_workers.py) for model execution, and [hybrid_data.py](../src/ml_training/hybrid_data.py) for public versus evaluation-only fields. Then write your own next-step proposal using the [production tradeoff guide](PRODUCTION_TRADEOFFS.md).

For an architectural choice, compare the use/avoid criteria and alternatives for [D09](ARCHITECTURE_DECISION_GUIDE.md#d09--compose-a-workflow-or-agent) in the decision guide.
