# Observed local language-model results

Measured on 2026-10-26 on Apple M2 Max, 96 GB RAM. These are synthetic, single-seed pilot experiments. Smoke results are separate and do not establish held-out performance.

## D01: strong template result, weak challenge generalization

Run: [`local-s41-20261026T171916Z-abf0ce`](../artifacts/distillation/local-s41-20261026T171916Z-abf0ce/run.json).

| Variant | Test joint accuracy, 144 tickets | Test schema validity | Independent challenge joint accuracy, 12 tickets | Challenge escalation recall |
| --- | --- | --- | --- | --- |
| TF-IDF/logistic + regex | 93.06% | 100% | 66.67% (8/12) | 25% |
| Local 14B teacher | 95.83% | 100% | 83.33% (10/12) | 75% |
| Prompted 0.5B student | 1.39% | 25% | 8.33% (1/12) | 0% |
| Distilled 0.5B student | 100% | 100% | 50% (6/12) | 0% |

Joint accuracy requires the correct intent and account extraction in the strict JSON schema. Invalid JSON is a failure; no fence stripping or automatic output repair inflates the baseline. Account extraction was perfect for the teacher, classical baseline and tuned student on test and challenge. The challenge failure is therefore about routing and abstention rather than merely formatting.

The tuned student met the frozen validation floor and was selected before test evaluation, but failed every ambiguous challenge that required escalation. It also failed challenge cases involving negation or competing topics. The post-evaluation recommendation is to keep it as a bounded educational router, not promote it to an open-ended support router. The teacher's challenge result is better but still imperfect, with too few examples to establish reliability. No additional training or threshold adjustment was performed using these test outcomes.

The teacher generated 2,016 actual responses, of which 1,974 passed the independent synthetic oracle. Forty-two outputs incorrectly routed query requests about billing cases to the billing intent; those outputs remain in the curation log and were excluded. Accepted class counts were 336 for each intent except query, which retained 294. The training set contains repeated scenario families with different identifiers; its size is not evidence of equivalent real-world diversity.

Student training changed 90,112 LoRA parameters over 494 optimizer steps. Validation selected epoch 1 at step 247; epoch 2 tied its 100% validation joint accuracy. The selected adapter was saved and reloaded, and its parameter hash was checked before final evaluation. The full training/validation loop took 120.0 seconds. Teacher batch generation took 1,380.1 seconds and produced 31,001 tokens.

| Variant | Warm validation p50 | Warm validation p95 | Samples |
| --- | --- | --- | --- |
| Classical | 0.252 ms | 0.338 ms | 12 |
| Teacher | 885.3 ms | 974.2 ms | 12 |
| Base student | 92.3 ms | 152.7 ms | 12 |
| Tuned student | 87.4 ms | 97.7 ms | 12 |

These use the same two validation requests per intent, batch 1, concurrency 1 and one warmup, with a maximum of 64 generated tokens. They are in-process generation/prediction timings, not HTTP service latency. The initial artifact's `scope` string mistakenly includes parsing; the timed function did not validate neural JSON. This report corrects that description, and subsequent code labels the scope accurately. Twelve samples are too few for precise tail-latency estimates.

The tuned student was about ten times faster than the teacher in this small matched latency sample, but its challenge accuracy was lower. The illustrative time-only amortization is about 1,711 requests, using batch labeling/training time divided by batch-1 p95 savings. It is not money, energy, throughput or a deployment recommendation. No hosted API was used; energy and depreciation costs were not measured.

Evidence: `training.json`, `selection.json`, `teacher_curation/outputs.jsonl`, `metrics.json`, `challenge.json`, `predictions.json`, `failure_examples.json`, `benchmarks.json` and `costs.json` in the run directory. Clean-environment reproduction passed for every test/challenge output and metric; see the reproduction section below.

## D02: syntax, semantics and a bounded parser

Run: [`local-s41-20261026T175054Z-61eada`](../artifacts/schema_dsl/local-s41-20261026T175054Z-61eada/run.json).

| Variant | Test schema validity | Test execution correctness, 160 tasks | Unseen-composition correctness, 40 tasks | Independent challenge, 8 tasks |
| --- | --- | --- | --- | --- |
| Prompted base | 0% | 0% | 0% | 0/8 |
| Tuned adapter | 93.13% | 90.63% (145/160) | 62.5% (25/40) | 4/8 |
| Typed-slot parser | 100% | 100% | 100% | 1/8 |

The base model failed the strict output contract; outputs are preserved rather than repaired. The tuned model's JSON parse rate was 93.75%, schema validity 93.13%, and execution correctness 90.63%. These are distinct checks. On the challenge, every tuned output parsed as JSON, but only five satisfied the schema and four had correct semantics. Fine-tuning therefore does not remove the need for external validation or semantic checks.

The typed parser uses only the input request and handles the published bounded grammar. It won validation selection and all generated test cases, including novel field combinations. Its poor challenge coverage limits it to an explicit command language. The neural adapter generalized to some independent wording but missed half the small challenge set and struggled with new compositions; neither result establishes broad natural-language reliability.

The full run executed 500 optimizer steps at batch 4 across two epochs. The initial batch-8 run failed with a Metal command-buffer interactivity error before producing a trained checkpoint; it is retained as `local-s41-20261026T174816Z-937c71` with failed status. Retrying at batch 4 completed. This was a hardware-driven change, not test-driven tuning.

## D04

Run: [`local-s41-20261026T175513Z-eae775`](../artifacts/edge/local-s41-20261026T175513Z-eae775/run.json).

| Variant | Test command accuracy, 80 requests | Test schema validity | Independent challenge, 10 requests | Warm validation p95 |
| --- | --- | --- | --- | --- |
| Prompted base | 56.25% | 58.75% | 5/10 | 148.2 ms |
| Tuned adapter | 67.5% | 76.25% | 6/10 | 91.9 ms |
| Fused, 4-bit tuned model | 82.5% | 92.5% | 6/10 | 71.3 ms |

Validation selected the unquantized tuned model at 81.25%, ahead of 70.83% for the quantized model. The final test reversed that ranking. The frozen validation choice remains unchanged; this single-seed result does not establish that quantization improves accuracy. Neither variant is reliable enough for unattended real device control.

The adapter received 256 optimizer steps, with epoch 2 selected, in 43.1 seconds including validation. It updates 90,112 parameters. The fused 4-bit weight artifact is 278,065,225 bytes, versus 988,097,824 bytes for the source weights, about 3.55 times smaller. Quantization includes scale/metadata overhead, so effective storage exceeds exactly four bits per weight. The model was reloaded from exported weights before evaluation.

The dispatcher rejected invalid schemas and out-of-range payloads before state mutation, as separately checked by the negative-case suite. That guarantee is narrower than matching user intent. Among 24 test/challenge requests whose oracle action was rejection, schema-valid predictions changed the simulator state in 3 cases for the base model, 3 for the tuned model, and 7 for the quantized model. Those are failures even though the resulting settings stayed within the allowed API range. Runtime checks cannot infer whether a valid command was what the user meant.

All state changes were in the deterministic simulator. Both tuned variants had zero explicit-rejection accuracy on the four rejection challenge cases; some invalid outputs were blocked by the dispatcher instead. The report retains raw predictions and before/after states so a schema-validity percentage cannot obscure this limitation.

Warm timing uses 12 validation samples, batch/concurrency 1 and one warmup, including prompt processing, generation and the runtime gate. Peak MLX memory in the training process is cumulative; fresh-process/offline reports provide separate observed startup and memory evidence. The initial batch-8 local run failed with a Metal interactivity error; the batch-4 retry completed.

## Clean reproduction and offline execution

`artifacts/validation/slm.json` records a fresh environment installed from `environments/mlx/uv.lock`. It verified every run's artifact hashes, reloaded the trained/quantized models, and reproduced exact raw outputs plus metrics for D01's 144 test and 12 challenge tickets, D02's 160 test and eight challenge tasks, and D04's 80 test and ten challenge requests. All comparison variants were rerun, including the actual teacher, classical router and typed parser.

Fresh processes measured the following launch-to-exit times, including imports, recursive artifact checksums, model load, one validation request and result serialization. Each is one observed sample, with uncontrolled OS file cache; do not interpret it as a cold-cache latency distribution.

| Task/variant | Fresh process seconds | Process peak RSS, decimal GB |
| --- | --- | --- |
| D01 classical | 1.864 | 0.224 |
| D01 teacher | 3.971 | 8.733 |
| D01 base | 2.942 | 1.408 |
| D01 tuned | 2.407 | 1.404 |
| D02 base | 1.598 | 1.273 |
| D02 tuned | 1.547 | 1.275 |
| D02 typed parser | 0.950 | 0.090 |
| D04 base | 1.552 | 1.273 |
| D04 tuned | 1.530 | 1.271 |
| D04 quantized | 1.755 | 0.678 |

`artifacts/validation/edge-offline.json` records a separate fresh process run under macOS `sandbox-exec` with `(deny network*)`. A TCP connect attempt to loopback returned an OS permission denial, after which the actual quantized device model loaded and generated new predictions for the first request and all ten independent challenge cases. The challenge remained 6/10. This proves local execution in that observed blocked-network process; it does not prove semantic safety, performance on other edge hardware or that a cached replay is acceptable.
