# D01, D02 and D04: local language-model training

These tracks share a pinned 0.5B student and MLX LoRA trainer, but use separate task adapters and evaluation contracts. Consult `PROGRESS.md` for which runs have actually completed. Code availability and compatibility probes do not establish an evaluated demo.

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

## D02: strict syntax and executable semantics

```bash
ml-demo train schema_dsl --profile local --seed 41
ml-demo run schema_dsl --run-dir artifacts/schema_dsl/<run> \
  --query "Count tickets where priority equals urgent AND amount is greater than 50."
ml-demo evaluate schema_dsl --run-dir artifacts/schema_dsl/<run>
```

The dataset contains 1,000 training, 100 validation and 160 test requests, plus eight independent challenges. A quarter of test examples combine fields never paired during training. Phrasing families are held out, although primitive ASTs can recur. The task is deliberately bounded and synthetic.

The interpreter accepts an exact typed JSON grammar for filters and count/IDs/sum/mean aggregates. It rejects duplicate keys, unexpected fields, nonfinite values, unsupported operations and excessive complexity. It never executes arbitrary Python or SQL. Semantic evaluation compares results on independent randomized tables, threshold-boundary witnesses and empty tables. Valid JSON with incorrect behavior remains a failure.

Compare base and tuned generation with `--variant base`, `--variant tuned`, and `--variant typed_slot`. The last is a deterministic parser for a documented bounded language, not constrained neural decoding. It receives only request text, with no oracle label. Independent challenge wording exposes its limited coverage. All variants use the same runtime gate.

## D04: local model and simulated device

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

## Clean environment and evidence

```bash
UV_PROJECT_ENVIRONMENT=.venv-repro uv sync --project environments/mlx --frozen
environments/mlx/.venv-repro/bin/python scripts/validate_slm.py \
  artifacts/distillation/<run> artifacts/schema_dsl/<run> artifacts/edge/<run> \
  --output artifacts/validation/slm.json
```

Reproduction verifies artifact hashes, reloads each neural variant and classical/typed alternative, and checks exact test/challenge outputs as well as metrics. Each run contains a frozen dataset reference, source hashes, dependency versions, training losses and update evidence, checkpoint selection, raw predictions, failures and a readable report. The complete lab still requires RLVR, hybrid integration and final reproduction even after these three tracks finish.
