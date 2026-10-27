# ML Training Demo Lab: Implementation Plan

This document is the engineering design behind the tutorials. New learners should start with the [learning guide](../LEARNING_GUIDE.md); return here to see how a broad use case becomes an experiment with a baseline, a data contract and a completion criterion. The original plan and acceptance boundaries are preserved so the measured findings can be judged against what was specified before evaluation.

Created: 2026-10-26. Status: P0–P6 implemented and validated locally. All eight source categories plus the hybrid capstone have evaluated artifacts and clean-environment replay evidence. See [FINAL_REPORT.md](FINAL_REPORT.md), [CONFORMANCE.md](CONFORMANCE.md) and [PROGRESS.md](PROGRESS.md). The scope and completion criteria below remain unchanged.

## Goal and completion boundary

Build, run, evaluate, and document eight independently runnable demo tracks covering every distinct use-case category in the supplied PDF, then connect them in one hybrid application. Each training track must include a real model update, a saved artifact, a reload-and-infer check, and held-out evaluation against a simpler or untuned baseline. The final result is a reproducible educational lab with measured tradeoffs, not a production deployment.

Completion means all milestones below have execution evidence and the final walkthrough works from documented setup instructions. A mock, cached output, configuration file, or training wrapper alone does not complete a training track. A valid experiment with no improvement can complete a track if its result, failure analysis, and deployment decision are explicit. Hardware-blocked training remains incomplete.

## Source coverage

Source: `/Users/markcastillo/Downloads/Fine-Tuning and ML Use Cases - Google Gemini.pdf`, six pages. The first response lists eight categories; the second expands four overlapping pipelines. We implement the union rather than only the four expanded pipelines. Each category gets one representative application; illustrative industry examples such as credit underwriting and ad auctions are not separate required products.

| ID | PDF category and pages | Demo | Observable result |
| --- | --- | --- | --- |
| D01 | Distillation and unit economics, pp. 1, 3 | Support-request intent router and argument extractor | Teacher versus untuned student versus tuned student quality, latency, and cost |
| D02 | Rigid schemas and custom DSLs, p. 1 | Natural language to a typed support-query AST | Syntax validity, semantic correctness, and constrained-generation comparison |
| D03 | RFT / RLVR, pp. 1, 3-4 | Train a small policy to solve executable query tasks | Actual GRPO updates, verifier pass rates, and compute-matched controls |
| D04 | Edge and on-device deployment, p. 1 | Offline device-command interpreter | Local adapter inference, bounded commands, memory and latency measurements |
| D05 | Tabular business prediction, pp. 2, 4-5 | Synthetic transaction anomaly scoring | Logistic versus boosted-tree discrimination and calibration |
| D06 | Ultra-low-latency classifiers, pp. 2-3, 4-5 | Request-metadata abuse classifier | Native versus ONNX correctness and measured CPU throughput |
| D07 | Explainability and attribution, pp. 2, 5 | Explain individual D05 risk predictions | Coefficients, SHAP contributions, additivity checks, and readable reason codes |
| D08 | Bi-encoders and cross-encoders, pp. 2, 5 | Version-aware API documentation retrieval | Separately trained retriever and reranker with retrieval ablations |
| D09 | Dynamic agent baseline and hybrid stack, pp. 2-3, 5-6 | Local support and operations assistant | End-to-end traces, changing knowledge, routing and fallback ablations |

## Design choices and assumptions

- Use a fictional service, **Harbor Support**, for support tickets, versioned API docs, transaction telemetry, and a simulated device API. Shared vocabulary makes the integrated demo understandable; each track retains independent datasets and commands.
- Start on the current macOS arm64 host. P0 subsequently verified Apple M2 Max, 96 GB RAM and sufficient local disk. Actual load/update/reload probes established viable backend and batch settings; the environment and task run manifests record those observations.
- Use Python and small local datasets. Prefer CPU for classic ML, MLX for compatible Apple Silicon causal-language-model tuning, and PyTorch/Sentence Transformers for retrieval. Keep the RLVR environment separate so dependencies do not destabilize the CPU demos.
- Start with a compatible approximately 0.5B-1.5B student, a larger locally runnable teacher if memory permits, and a small pretrained encoder. Exact model IDs, revisions, licenses, tokenizer versions, and quantization choices are selected and locked after a load/train/save/reload probe. Model size alone does not establish teacher quality.
- Use actual teacher-generated final labels or structured answers, validated against independent task oracles. Never label rule-generated data as teacher distillation. Do not require private chain-of-thought or train against unverified reasoning traces.
- Prefer local teacher generation. A hosted teacher is an optional replacement with an explicit spend cap and recorded provider/model details. No cloud deployment or paid compute is required by this plan.
- If local RLVR support is insufficient, finish independent tracks and document the missing compute requirement. A Linux/CUDA profile may be prepared, but an actual run is still required for D03 completion. Do not silently substitute rejection sampling or SFT for reinforcement learning.
- Use a local CLI plus a small Streamlit comparison app. Build the UI after experiment artifacts exist. Keep external vector databases, distributed serving, Kubernetes, Rust services, and adapter hot-swapping out of the first version.

## Claims we will test carefully

The PDF is framing material rather than a benchmark specification. We will not assume 5-10x speedups, microsecond service latency, a fixed number of sufficient examples, or a fixed percentage of traffic requiring a larger model.

Fine-tuning cannot guarantee valid schemas; compare it with validation and constrained output where supported. SHAP explains a specified model under specified assumptions; it is not a causal explanation or a compliance certification. A verifier can have blind spots. RLVR results do not establish general reasoning or self-correction unless those behaviors are separately evaluated. Tree performance is dataset-dependent, so simpler models remain real contenders.

## Project structure and interface

```text
pyproject.toml
README.md
docs/                    # plan, walkthroughs, experiment reports
configs/                 # locked smoke, local, and optional CUDA profiles
src/ml_training/
  common.py              # artifacts, seeds, metrics, timing
  *_data.py              # seeded generators and provenance
  distillation.py        # D01
  structured.py          # D02 and D04, with task-specific contracts
  rlvr.py                # D03
  tabular.py             # D05
  fast_classifier.py     # D06
  explainability.py      # D07
  retrieval.py           # D08
  hybrid.py              # D09, with isolated model workers
app/                     # local comparison UI
tests/                   # dataset, verifier, artifact, and integration contracts
data/fixtures/           # independently authored challenge fixtures
data/generated/         # frozen generated data and manifests
artifacts/<demo>/<run>/  # models, predictions, metrics, environment, report
```

The implemented CLI provides `ml-demo list`, `ml-demo doctor`, and task-specific `ml-demo <prepare|train|evaluate|run> <demo> --profile <profile>` commands. `ml-demo verify all --profile smoke` runs data/verifier/artifact contracts; `ml-demo report all` aggregates real evidence. D07 consumes a trained D05 artifact through `run`; the hybrid runs existing components rather than training another model. Walkthroughs specify the appropriate environment, dependency artifacts and supported options for each task.

Profiles: `smoke` uses tiny data to check wiring, `local` runs the actual bounded experiment on this host, and `cuda` is optional for D03. Smoke results must be labeled separately from evaluation results. Every track also gets a short walkthrough explaining its learning objective, commands, expected output shape, observed results, and limitations.

## Shared data and evaluation contract

1. Build seeded generators for noisy tickets, versioned docs, bounded query tasks, device commands, and event streams. Label all synthetic data. Add a small independently authored challenge set to avoid testing only generator templates. Synthetic performance establishes mechanics, not effectiveness on real customers.
2. Define train/validation/test splits before teacher labeling or tuning. Split text by template/scenario family and retrieval by document/query family, including version conflicts. Split telemetry chronologically, with entity-aware checks and a future-period drift slice. Use rolling features computed only from information available at prediction time.
3. Fit transformations, calibration, threshold selection, and negative mining on training or validation partitions only. Freeze test manifests and hashes. Keep final test queries, labels, and verifier cases out of model selection and training rewards.
4. Begin with approximately 2,000-5,000 training tickets, 50,000 events, 200-500 documentation chunks, 1,000 labeled retrieval queries, and 500-1,000 executable task specifications. These are starting budgets, not sufficiency claims. Log actual counts and class coverage; expand only when error analysis supports it.
5. Record seed, input checksums, code identity (Git revision when available, otherwise source hashes), dependency lock, model revision, tokenizer/prompt, device, precision, training settings, duration, memory, and evaluation configuration in `run.json`.
6. Save raw held-out predictions and per-example errors, not just aggregate metrics. Include matched baseline outputs and uncertainty estimates where the sample size permits. Run three seeds for cheap classic-ML experiments; use a single explicitly labeled pilot seed for expensive tuning, then replication before broad superiority claims.
7. Benchmark cold start separately from warm inference. Report p50/p95/p99, batch size, concurrency, thread count, input/output lengths, warmups, sample count, and hardware. Separate model-only latency from preprocessing and end-to-end service latency. Never compare different hardware or decoding budgets without labeling the difference.
8. Record teacher labeling cost, training time, artifact size, and per-request inference cost. Distinguish measured API charges from estimated local compute cost. Estimate break-even request volume only when the assumptions and quality requirements are stated.
9. Set quality selection criteria using validation data before opening final test results. Report achieved values even when the tuned model loses. Final artifacts must identify `planned`, `smoke`, `trained`, and `evaluated` evidence explicitly; skipped steps are never counted as passes.

## Demo implementation contracts

### D01: Distilled support router

Task: map a support ticket to one of six intents, extracted arguments, and an abstain/escalate outcome. Include ambiguous tickets, negation, unseen wording, malformed inputs, and conflicting instructions inside ticket text.

Implementation: build a TF-IDF/logistic baseline, a prompted untuned student, and a larger teacher; collect and validate teacher outputs; train a student LoRA adapter; save/reload it and run identical held-out inputs. Keep a separate record of teacher errors and curation exclusions. Use external validators and explicit labels for correctness rather than teacher agreement alone.

Measure macro-F1, argument exact match, JSON validity, abstention coverage versus error, p95 latency, output tokens, and labeling/training cost. Show cases where each model wins or fails.

Acceptance: real teacher provenance and real adapter updates are present; the reloaded student completes evaluation; all three neural variants and the classical baseline have comparable results. Adopt the student only if it meets the frozen quality floor and improves a measured resource tradeoff; otherwise record a negative result.

### D02: Schema and DSL tuning

Task: translate user requests into a small typed JSON AST for filtering and aggregating support tickets. Support a bounded set of fields, operators, nesting depth, and aggregates. Use a deterministic interpreter over fixture tables, not arbitrary Python or unrestricted SQL.

Implementation: write the schema, interpreter, semantic oracle, and adversarial fixtures first. Compare prompted base generation, a separate LoRA adapter, and a constrained-generation baseline with a compatible backend. If the local backend lacks constrained decoding, implement a bounded typed-slot alternative and label it as a different baseline. Include unseen compositions of known operators.

Measure parse validity, schema validity, execution equivalence, token count, and latency separately. Valid JSON with wrong semantics is a failure. Apply the same runtime validation to all variants.

Acceptance: saved/reloaded adapter, held-out semantic evaluation, and an explicit syntax-versus-semantics comparison. Malformed or unsupported operations never reach execution. A loss to constrained generation is a legitimate finding.

### D03: RLVR with deterministic rewards

Task: generate the D02 query AST for specifications with executable expected outcomes. Reuse the language and interpreter but create disjoint training and hidden evaluation tasks. Start from the same SFT checkpoint for the RL policy and SFT control.

Implementation: first test the verifier against known-good, known-bad, timeout, empty-result, and reward-hacking examples. Give training rollouts public checks; reserve additional hidden cases for final evaluation. Run actual GRPO through a supported trainer, recording sampled groups, rewards, reward variance, optimizer steps, and parameter changes. Bound rollout lengths and interpreter work; no generated arbitrary-code execution is necessary.

Use a bounded pilot (for example, 50-200 optimizer steps and 4-8 candidates per group), selected after a resource probe. Track groups with constant reward; any filtering uses training prompts only and must not change test difficulty. Zero group variance removes the within-group reward advantage, but does not justify a blanket claim that every possible loss term has zero gradient.

Compare untuned base, SFT, SFT plus verifier-based best-of-N, and RLVR at matched inference budgets. Measure pass@1, verifier-selected success at N, hidden-case generalization, output length, training compute, and reward exploits. Add a separate fixed-budget repair test if claiming self-correction.

Acceptance: an actual reward-driven optimizer run produces a reloadable checkpoint and hidden evaluation. A verifier demo, selected completions, or training config is insufficient. Report no improvement honestly; do not use repeated test-driven tuning to manufacture a win.

### D04: On-device command interpreter

Task: convert a natural-language instruction to a bounded local device API: read status, set a simulated temperature within a permitted range, or schedule a simulated job. No real device is controlled.

Implementation: tune a task-specific small adapter; compare base and tuned inference, plus a quantized variant where supported. Run after all artifacts are cached with outbound network access disabled. Use a deterministic allowlist and range checks outside the model, and display the simulator state before and after accepted commands.

Measure command exact match, valid-action rate, rejected invalid actions, warm and cold latency, memory, model size, and offline startup behavior. Apple Silicon results do not establish performance on a microcontroller or other edge hardware.

Acceptance: actual local training/inference, cached offline run, saved/reloaded model, observed resource report, and a negative-case suite proving invalid model output cannot bypass the dispatcher. Replaying cached answers does not count as offline model inference.

### D05: Tabular prediction

Task: score synthetic transaction anomalies using amounts, merchant categories, event timing, and causal rolling-window features. Include noise, missing values, class imbalance, entity variation, and temporal drift; avoid a trivial label rule directly exposed in one feature.

Implementation: build an event-to-feature pipeline with DuckDB or Polars; compare a dummy classifier, logistic regression, and one boosted-tree implementation (initial default: XGBoost). Fit preprocessing inside the training pipeline and calibrate using validation data. Persist the complete feature/preprocessing/model contract.

Measure PR-AUC, ROC-AUC, recall at a validation-selected false-positive budget, Brier score, subgroup slices, training time, and inference latency. Explain why PR-AUC matters for the chosen base rate.

Acceptance: all baselines trained and reloaded, reproducible temporal evaluation, leakage checks, and a report on drift sensitivity. The strongest validated baseline wins; a tree is not required to win.

### D06: Fast CPU classifier

Task: classify request metadata as benign or suspicious with a fixed-size numerical feature vector. Reuse selected D05 feature infrastructure while using a separate labeled request dataset.

Implementation: compare logistic regression and a small tree or tiny MLP; export a supported model to ONNX; check numerical prediction parity; benchmark native and ONNX execution at batch 1 and batched loads. Add a thin local HTTP endpoint only after model-only profiling works.

Measure parity error, classification quality, p50/p95/p99 model-only and end-to-end latency, throughput at stated concurrency, and memory. Use a provisional engineering target of p95 model-only latency below 1 ms on the measured host, explicitly labeled a target. A missed target requires profiling and an explanation, not hidden exclusions.

Acceptance: a real exported model executes through ONNX Runtime, parity passes a declared tolerance, and benchmark conditions are reproducible. Do not claim 100,000 requests/second or microsecond service latency from a tight prediction loop.

### D07: Prediction explanations

Task: explain why the D05 system assigned risk to a transaction and how the explanation changes under a controlled feature edit.

Implementation: report logistic coefficients with preprocessing context and TreeSHAP local/global attributions for the boosted tree. Record the background dataset, feature dependence assumption, and output space. Verify that base value plus contributions reconstructs the model output within a declared tolerance. Generate reason text from feature names and signed contributions, without an LLM inventing causal claims.

Measure attribution additivity residual, prediction consistency, and stability under small perturbations. Show a correlated-feature example where explanations can shift. Probability-space and log-odds-space contributions must be labeled correctly.

Acceptance: a user can select a held-out row, see its prediction and reproducible contributions, change an allowed feature, and rerun both. The report states the assumptions and distinguishes attribution from causality and legal compliance.

### D08: Retrieval and reranking

Task: retrieve the correct API documentation version despite near-identical endpoint names, acronyms, outdated pages, and plausible wrong documents.

Implementation: build a fixed corpus and relevance judgments; establish BM25 and untuned dense baselines. Mine negatives from training data and screen for false negatives. Fine-tune a small bi-encoder using contrastive training, then separately fine-tune a cross-encoder on labeled query-document pairs. Start with exact in-process vector search to isolate model effects; a database can be added later if needed.

Measure Recall@5/10, MRR@10, nDCG@10, wrong-version rate, and latency. Run the full ablation: BM25; base bi-encoder; tuned bi-encoder; tuned bi-encoder plus base reranker; tuned bi-encoder plus tuned reranker. Include a reranker comparison on a fixed candidate pool so retrieval changes do not hide reranker effects. Report the candidate-recall ceiling.

Acceptance: both encoders have actual saved/reloaded trained artifacts, disjoint evaluation, and independently reported effects. Merely calling a pretrained embedding API or adding vector search does not complete this track.

### D09: Hybrid capstone and no-training agent baseline

Task: a user submits a support request; metadata is scored, intent is routed, relevant versioned docs are retrieved, and a bounded tool action or query is proposed. A more capable local model handles ambiguous open-ended cases when resources permit; a hosted fallback is optional and must be labeled.

Implementation: first build an untuned prompting + retrieval + tools baseline. Then connect D06 screening, D01 routing, D08 retrieval, D02/D03 verified query generation, D04 simulated device actions, and D05/D07 risk explanations. Keep deterministic authorization/validation around all actions. Derive escalation thresholds on validation data and display the reason for each route.

Show the effect of updating a document without retraining the language model. Run all requests through both the baseline and hybrid pipeline, then disable one learned component at a time to measure its contribution. Do not hide failed requests by excluding escalations from the denominator.

Measure end-to-end task success, false rejection, fallback rate, wrong-version retrieval, invalid actions rejected, total latency, and resource/cost breakdown. Display per-stage inputs, decisions, model revision, timing, and evidence status.

Acceptance: one local walkthrough exercises all branches using actual models and artifacts, includes errors and fallbacks, and produces a baseline-versus-hybrid report. The UI clearly distinguishes simulator state from real model execution.

## Milestones and dependencies

| Phase | Work | Exit evidence |
| --- | --- | --- |
| P0: Foundation | Hardware/dependency probes; choose model revisions; project skeleton; shared contracts; seeded datasets; frozen challenge sets | Environment manifest, dataset manifests, leakage checks, model feasibility results |
| P1: CPU vertical slice | D05, then D06 and D07 | Trained tabular models, ONNX parity and timing report, attribution checks, CLI walkthrough |
| P2: Retrieval | D08 base retrieval, bi-encoder tuning, cross-encoder tuning | Both trained artifacts and retrieval ablation report |
| P3: SLM tuning | D01 teacher curation and distillation; D02 AST tuning; D04 offline device demo | Three task adapters, base/tuned evaluations, offline execution evidence |
| P4: Verifiable RL | D03 verifier hardening and GRPO pilot | Actual RL checkpoint, training log, hidden evaluation and compute-matched comparisons |
| P5: Integration | D09 baseline, component integration, local comparison UI | Full scenario traces, baseline/hybrid ablations, measured fallback behavior |
| P6: Reproduction | Clean setup checks, artifact reloads, walkthroughs, final reports | All nine demos runnable, evidence index, limitations and reproducibility report |

Dependencies: P1/P2 follow P0; P3 follows P0; D03 depends on D02 and its SFT checkpoint. P5 can begin with the untuned baseline while training runs are pending, but cannot complete until each required component has real evidence. P6 follows P1-P5. Execute sequentially by default; no parallel agents are required.

First implementation slice: finish P0, then deliver D05 end to end before adding other tracks. This validates the experiment/artifact/report contract with inexpensive training.

## Resources and bounded execution

This is a bounded lab whose feasibility was established in P0. Observed execution times and accepted settings are recorded in the result reports; they are not a general compute estimate for other hosts. Keep model downloads sequential and check free disk and license compatibility. Use small pilots before larger runs; each profile records maximum examples, steps, sequence length, checkpoint count, and rollout count.

CPU experiments should remain small enough for repeated local execution. Transformer training first proves load, one update, save, reload, and generation; that probe does not count as a meaningful experiment. Full local runs need their own recorded evaluation. If a required run exceeds measured memory, reduce batch/sequence/model size or use another compatible local profile; document the changed comparison.

Remote resources, paid teacher APIs, public uploads, and production deployment are separate decisions if they become necessary. Do not invent an available GPU or treat an old host configuration as current. Continue independent local work when one backend is blocked.

## Final definition of done

- All eight source categories have their own runnable demos and walkthroughs; the hybrid capstone is runnable too.
- D01/D02/D03/D04/D08 contain actual parameter-update evidence and saved/reloaded artifacts; D08 includes both retrieval and reranking training. D05/D06 contain trained classical models; D07 uses their actual held-out predictions.
- Every demo has frozen data provenance, a baseline, honest held-out results, failure examples, and documented resource requirements.
- Contract tests cover data leakage, semantic verifier failures, export parity, artifact reloads, invalid-action rejection, and attribution reconstruction where applicable. Smoke tests do not substitute for model evaluation.
- The comparison UI and CLI consume the same real run artifacts. Missing results remain visibly missing.
- Reproduction has been attempted from a clean environment using the documented commands, with any remaining platform differences stated.
- The final report identifies where training helped, where it did not, what was simulated, and what remains unproven on real data or production hardware.

## Implementation references

Primary documentation checked during planning on 2026-10-26; lock compatible versions after P0 probes rather than assuming latest releases work together.

- [MLX-LM LoRA/QLoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) for Apple Silicon adapter training and configuration.
- [TRL GRPO trainer](https://huggingface.co/docs/trl/grpo_trainer) for actual reward-driven policy updates and reward-group diagnostics.
- [Sentence Transformers cross-encoder training](https://sbert.net/docs/cross_encoder/training_overview.html) for training and evaluating the reranker separately from retrieval.
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) for train-only preprocessing and leakage prevention.
- [ONNX Runtime threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) for explicit benchmark thread settings.
- [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) for background-data, output-space, and feature-dependence configuration.

## Learning exercise: turn a use case into an experiment

Choose one track and write its hypothesis without naming the model. Define the input, required outcome, simplest baseline, failure cost and held-out population. Then select a training method and explain what evidence would justify its extra complexity.

Compare your proposal with this plan's acceptance criteria. Notice that a real model update and a valid negative result can complete an educational experiment. “The tuned model wins” is a hypothesis to test, not a result to force. Use [PRODUCTION_TRADEOFFS.md](../PRODUCTION_TRADEOFFS.md) to extend your proposal beyond the lab's scope.
