# D01, D02 and D04: local language-model training

Visual companions: [D01 notebook](../notebooks/D01_distillation.ipynb), [D02 notebook](../notebooks/D02_structured_generation.ipynb), [D04 notebook](../notebooks/D04_local_inference.ipynb). These plot the recorded evidence and include editable examples without loading models; see the [notebook guide](../notebooks/README.md).

These lessons explore three different reasons to customize a language model: imitate a teacher on a narrow task, produce an executable structured answer, and run a compact model locally. They share a pinned 0.5B student and MLX trainer, but have separate adapters and evaluation contracts. Read the [learning guide](LEARNING_GUIDE.md) first and keep the [observed results](SLM_RESULTS.md) open while working through the examples.

## Shared background: what SFT and LoRA change

Supervised fine-tuning (SFT) increases the likelihood of target answer tokens given a prompt. In simplified notation, the loss is the average of `-log p(answer_token | prompt, earlier_answer_tokens)`. The prompt provides context, while this trainer's loss mask targets the completion, including its end token. Lower training loss says the model fits those targets better; it does not establish correct intent on new requests.

LoRA freezes the base weights and learns a low-rank update. Abstractly, a linear layer becomes `W' = W + sBA`, where the inner dimension of `B` and `A` is rank `r` and `s` scales the update. This reduces trainable parameters while retaining the base computation. See the [LoRA paper](https://arxiv.org/abs/2106.09685). Rank is a capacity choice, not a universal quality dial.

The implementation adapts query/value projections in the last four transformer layers. Its MLX configuration uses rank 8 and `scale=16`; do not blindly equate that field with another library's `alpha` or `alpha/r` convention. D03 uses a separate PEFT configuration. When comparing libraries, inspect the actual update definition rather than copying a number by name.

~~~mermaid
flowchart TD
    X["Prompt and target completion"] --> B["Frozen base plus trainable LoRA"]
    B --> L["Completion-token loss"]
    L --> U["Update adapter weights"]
    U --> V["Select trained epoch on validation"]
    V --> S["Save and reload before test"]
~~~

An adapter depends on the base model revision, tokenizer and prompt format. A small adapter file does not mean the whole model fits in that space. Trace [training_batch and update_step](../src/ml_training/mlx_models.py) and [train_adapter](../src/ml_training/mlx_training.py) to see masking, optimizer updates and checkpoint selection.

## Setup

```bash
uv sync --project environments/mlx --frozen
source environments/mlx/.venv/bin/activate
hf download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775
hf download mlx-community/Qwen2.5-14B-Instruct-4bit --revision dad510143ae5cdb1466778bde3161460f9e9f604
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
```

Model loading expects those exact snapshots in the Hugging Face cache. `ML_DEMO_MODEL_CACHE` can point to another cache root. Both model cards declare Apache 2.0 licenses; the teacher's upstream model is Qwen2.5-14B-Instruct. The observed host is an Apple M2 Max with 96 GB RAM. Metal access is required for this MLX profile; a process inside a restrictive sandbox may not have it.

The task trainer uses Adam at 1e-4, batch 8 for routing and batch 4 for schema/device tasks, maximum sequence length 512, completion-only loss, and two epochs. LoRA has rank 8, scale 16, no dropout, and updates query/value projections in the last four transformer layers. Each task selects its best trained epoch on validation accuracy; earlier epochs win ties. Saved adapter hashes must differ from initialization and match after reload. Full schema/device runs with batch 8 hit a macOS GPU interactivity error; failed runs remain recorded, and their retries use batch 4.

## D01: distillation

### Learn the distinction: distillation versus trusted labels

The goal is to replace repeated large-model routing with a smaller task model. Distillation transfers behavior from a teacher to a student; the [original distillation paper](https://arxiv.org/abs/1503.02531) studies learning from teacher probability information. Here the implementation is **sequence-level teacher-label distillation**: the teacher generates a final JSON answer, an independent check accepts or rejects it, and SFT trains on accepted answer tokens. No teacher logits or hidden states are matched.

Consider “The card was charged twice. Account C1234.” Routing needs both a billing intent and an exact account value. Producing JSON is only the format requirement. A ticket involving both billing and an API query may expose a semantic mistake even when every field is valid.

Teacher errors therefore belong in the dataset story. The synthetic oracle can check intent and extraction in this bounded world; for a real application you would need independently justified labels and an adjudication process. Teacher agreement alone can reproduce the same teacher mistake in a smaller model.

### Run and inspect

```bash
ml-demo prepare distillation --profile local --seed 41
ml-demo train distillation --profile local --seed 41
ml-demo run distillation --run-dir artifacts/distillation/<run> \
  --query "The card was charged twice. Account C1234." --variant tuned_student
ml-demo evaluate distillation --run-dir artifacts/distillation/<run>
```

The data has 2,016 training tickets, 72 validation tickets, 144 test tickets and 12 independent challenges. Phrasing families and account identifier domains are split before teacher generation. The six intents include explicit escalation; extraction requires an exact account identifier or null.

The local 14B teacher generates final JSON for every training request. Strict schema and independent synthetic labels determine whether a response enters training. `teacher_curation/outputs.jsonl` preserves accepted and rejected raw responses. The resumable cache is bound to the frozen dataset, prompt and model revision; it is never substituted with hand-written answers. Canonical JSON serialization is applied only after validating an actual teacher response.

The comparisons are TF-IDF/logistic routing plus documented regex extraction, the prompted base student, the teacher and the tuned student. Validation selection requires macro-F1 >= .95, argument exact match >= .95 and schema validity >= .99, then prefers measured p95 latency. If none qualifies, the result says so. Interactive `--variant selected` displays the tuned student when no variant qualifies and explicitly reports that it failed the validation floor; this is an educational display, not a production promotion.

Raw predictions, independent challenge results, routing coverage/error, failed examples, curation exclusions, update hashes and local compute estimates are saved. The break-even calculation is an illustrative time-only ratio, not dollars, energy, throughput or proven financial savings. Batch generation timing is not per-request latency. Benchmark samples are small and should not support precise tail-latency claims.

### What was learned, and production tradeoffs

The distilled model reached 100% joint accuracy on 144 generated test requests, but 6/12 on independently authored challenges and zero recall on four escalation challenges. The classical baseline achieved 8/12 challenges. The [result breakdown](SLM_RESULTS.md#d01-strong-template-result-weak-challenge-generalization) makes the difference between learning a task distribution and robustly understanding arbitrary support text visible.

A stable, frequent, well-labeled task can make distillation attractive. Frequent intent changes, rare consequential errors or broad ambiguous language increase curation and retraining demands. Route uncertain cases to a suitable fallback, but evaluate coverage and error together: a policy that sends everything to the teacher has avoided student errors by giving up most of its intended savings.

The local speed difference is useful evidence for a follow-up, not permission to accept the challenge failures. Production economics would include labeling, evaluation, periodic refreshes, serving utilization and fallback usage at an acceptable quality level.

### Try it and reason about the result

1. Compare the same new request with `classical`, `teacher`, `base_student` and `tuned_student` through `--variant`. Use a clear request, a negation and an ambiguous request. Predict the expected intent before looking at responses.
2. Inspect the 42 rejected teacher labels. Explain why excluding them improves target correctness but also leaves fewer accepted examples for one class.
3. Compare generated-test and challenge escalation outcomes. Write a selection rule that would have required an independently authored development set before promotion, then reserve new untouched examples for final evaluation.
4. Explain why a time-only amortization calculation is meaningless if the cheaper model fails the required task quality.

Follow [routing_data.py](../src/ml_training/routing_data.py) for the label contract and [distillation.py](../src/ml_training/distillation.py) for teacher curation, comparisons and selection.

## D02: strict syntax and executable semantics

### Learn the distinction: syntax, types and meaning

The model translates language into an abstract syntax tree (AST): a data structure describing a program. This lesson's program filters a bounded ticket table and applies count, IDs, sum or mean. No generated Python or SQL is evaluated.

For “Count tickets where priority equals urgent AND amount is greater than 50,” the intended AST is:

~~~json
{
  "filter": {
    "all": [
      {"field": "priority", "op": "eq", "value": "urgent"},
      {"field": "amount", "op": "gt", "value": 50}
    ]
  },
  "aggregate": {"op": "count"}
}
~~~

There are three checks: can the text parse as JSON, does it match the allowed typed grammar, and does its execution match the requested meaning? Replacing `gt` with `ge` preserves JSON and types but changes the answer for a row whose amount is exactly 50. That is why the verifier includes boundary witnesses.

Run this small semantics exercise in the root CPU environment; it needs no model download:

~~~python
from ml_training.query_dsl import execute

query = {
    "filter": {"all": [
        {"field": "priority", "op": "eq", "value": "urgent"},
        {"field": "amount", "op": "gt", "value": 50},
    ]},
    "aggregate": {"op": "count"},
}
rows = [
    {"id": "a", "priority": "urgent", "amount": 50},
    {"id": "b", "priority": "urgent", "amount": 51},
]
assert execute(query, rows) == 1
query["filter"]["all"][1]["op"] = "ge"
assert execute(query, rows) == 2
~~~

The example is a designed counterexample, not a measured model prediction. It teaches why checking only an empty table or a convenient random table can miss a wrong program.

### Run and inspect

```bash
ml-demo train schema_dsl --profile local --seed 41
ml-demo run schema_dsl --run-dir artifacts/schema_dsl/<run> \
  --query "Count tickets where priority equals urgent AND amount is greater than 50."
ml-demo evaluate schema_dsl --run-dir artifacts/schema_dsl/<run>
```

The dataset contains 1,000 training, 100 validation and 160 test requests, plus eight independent challenges. A quarter of test examples combine fields never paired during training. Phrasing families are held out, although primitive ASTs can recur. The task is deliberately bounded and synthetic.

The interpreter accepts an exact typed JSON grammar for filters and count/IDs/sum/mean aggregates. It rejects duplicate keys, unexpected fields, nonfinite values, unsupported operations and excessive complexity. It never executes arbitrary Python or SQL. Semantic evaluation compares results on independent randomized tables, threshold-boundary witnesses and empty tables. Valid JSON with incorrect behavior remains a failure.

Compare base and tuned generation with `--variant base`, `--variant tuned`, and `--variant typed_slot`. The last is a deterministic parser for a documented bounded language, not constrained neural decoding. It receives only request text, with no oracle label. Independent challenge wording exposes its limited coverage. All variants use the same runtime gate.

### What was learned, and production tradeoffs

The tuned model's generated-test schema validity was 93.13%, while execution correctness was 90.63%. Correctness fell to 62.5% on unseen field combinations. The deterministic parser achieved 100% on generated tasks, but handled only 1/8 independent challenges. See [the measured comparison](SLM_RESULTS.md#d02-syntax-semantics-and-a-bounded-parser).

The parser illustrates a useful trade: explicit command languages can offer exact behavior inside their grammar, at the cost of accepting less language. A neural model may cover more wording while making subtle semantic errors. Fine-tuning, constrained decoding and external validation address different parts of this problem; constrained neural decoding is not implemented here.

In production, version the schema and executor together, keep data access authorization outside generation, and test new combinations separately from familiar primitives. If the desired semantics cannot be independently specified, passing schema validation alone cannot justify execution.

### Try it and reason about the result

1. Run the boundary example, then remove the amount-50 row. Explain why both programs now return the same result and the reduced fixture fails to expose the bug.
2. Try `typed_slot` on supported wording and “Do not count tickets where status equals open.” The latter should be rejected as outside its grammar. Rejection demonstrates a boundary, not understanding of arbitrary negation.
3. Compare base, tuned and typed-parser outputs for a new composition. Record parsing, schema and execution outcomes separately.
4. Design another witness for `AND` versus `OR`. Your fixture must contain rows on which the two predicates disagree; two all-matching rows cannot distinguish them.

Read [query_dsl.py](../src/ml_training/query_dsl.py) for parsing/execution and [schema_data.py](../src/ml_training/schema_data.py) for split construction, fixtures and the typed parser. The latter receives request text, not the hidden expected AST.

## D04: local model and simulated device

### Learn the distinction: small, local and safe

This lesson combines two operations. Task adaptation learns a command format. Fusion and quantization produce a standalone representation with lower-precision weights. Grouped low-bit weights also need scales and other metadata, so “4-bit” is not an exact total-file-size calculation. [MLX's quantized linear layer](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.QuantizedLinear.html) documents the representation's configuration.

Quantization changes numerical computation and can change token choices. It should be evaluated after export and reload. A smaller file is not a complete memory budget: runtime buffers, activations, caches and initialization matter too.

The external dispatcher owns the permitted actions and ranges. A request for 40 Celsius should fail the 16–28 range gate. But a valid command setting 22 Celsius can still be wrong if the user said “do not change the thermostat.” Payload validity and intent fidelity require separate evaluation.

### Run and inspect

```bash
ml-demo train edge --profile local --seed 41
ml-demo run edge --run-dir artifacts/edge/<run> \
  --query "Set the thermostat to 22 degrees." --variant quantized_tuned
ml-demo evaluate edge --run-dir artifacts/edge/<run>
```

The dataset contains 512 training, 48 validation and 80 test requests, plus ten independent challenges. Template families are disjoint. Test includes 22 and 26 degree settings absent from training. Commands read status, set a temperature from 16 through 28 Celsius, schedule an integer hour from 0 through 23, or explicitly reject a request.

An external dispatcher validates exact keys, action names, argument types and ranges before updating a copied simulator state. The predictions show before/after states. Invalid outputs cannot bypass this gate; schema-valid but misunderstood commands remain a model accuracy failure. No physical device is controlled.

The three variants are base, task-adapted, and a fused then 4-bit-quantized task model. The quantized model is saved and reloaded before evaluation; its footprint is measured separately from the source model and adapter. Training-process memory is a cumulative high-water mark, not isolated per-model usage.

To prove local inference with networking blocked on macOS, run the validator under an OS policy:

```bash
sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  environments/mlx/.venv-repro/bin/python scripts/validate_edge_offline.py \
  artifacts/edge/<run> artifacts/validation/edge-offline.json
```

The script requires a permission-denied TCP connection attempt, loads the actual model in a fresh process, generates new answers and runs the simulator. A closed port or offline environment flag alone is insufficient. The resulting artifact records fresh-process time through first inference, peak RSS, challenge outputs and the observed OS denial. It does not flush the OS file cache or measure microcontroller performance.

### What was learned, and production tradeoffs

The [observed export](SLM_RESULTS.md#d04) was about 3.55 times smaller than the source weights. The quantized model scored better on the generated test, but worse on validation, and matched the tuned model's 6/10 challenge result. The frozen validation choice was retained. This reversal does not show that quantization generally improves accuracy.

More revealingly, the quantized model issued schema-valid state changes on seven test/challenge cases whose correct action was rejection. The dispatcher kept arguments inside the API contract but could not recover the intended meaning. Offline inference passed a separate OS-enforced network-denial check; that proves a different property again.

Local inference can help with connectivity and keeping request processing on a device. Production would still require measurement on the actual hardware, thermal/battery behavior, update distribution, current-state and permission checks, and command acknowledgement. A workstation running a simulator establishes none of those device-specific properties.

### Try it and reason about the result

1. Compare “Set the thermostat to 22 degrees” with an out-of-range request and an explicit non-action. Inspect raw output, dispatch acceptance and before/after state separately.
2. Compare artifact bytes, fresh-process RSS and warm latency across variants. Explain why the smallest file need not have the fastest startup.
3. Read the OS-network-denial record. Explain why an offline environment variable or a connection-refused error would provide weaker evidence than a denied connection.
4. Design a device-adapter test for an already-applied command being retried. This is a proposed production extension; the demo creates fresh simulator state.

Trace [device_data.py](../src/ml_training/device_data.py) for the action contract and [structured.py](../src/ml_training/structured.py) for training, export and evaluation. The command runner and the model have separate responsibilities.

## Clean environment and evidence

```bash
UV_PROJECT_ENVIRONMENT=.venv-repro uv sync --project environments/mlx --frozen
environments/mlx/.venv-repro/bin/python scripts/validate_slm.py \
  artifacts/distillation/<run> artifacts/schema_dsl/<run> artifacts/edge/<run> \
  --output artifacts/validation/slm.json
```

Reproduction verifies artifact hashes, reloads each neural variant and classical/typed alternative, and checks exact test/challenge outputs as well as metrics. Each run contains a frozen dataset reference, source hashes, dependency versions, training losses and update evidence, checkpoint selection, raw predictions, failures and a readable report. The complete lab still requires RLVR, hybrid integration and final reproduction even after these three tracks finish.

For an architectural choice, compare the use/avoid criteria and alternatives for [D01](ARCHITECTURE_DECISION_GUIDE.md#d01--distill-a-repeated-language-task), [D02](ARCHITECTURE_DECISION_GUIDE.md#d02--tune-structured-generation) and [D04](ARCHITECTURE_DECISION_GUIDE.md#d04--run-a-compact-model-locally) in the decision guide.
