# What the experiments teach

The central lesson is to choose model complexity through a measured comparison. Training helped several narrow synthetic tasks, but additional stages did not consistently improve the complete system. Strong template scores also concealed failures on independently authored language.

Read this chapter as the curriculum's synthesis after the [learning guide](LEARNING_GUIDE.md) and individual tutorials. It connects each technique to its observed result, the inference that result supports, and the next question a production application would need to answer.

## Source coverage and findings

| ID | Implemented use case | Observed finding | Evidence |
| --- | --- | --- | --- |
| D01 | Actual teacher-label distillation into a 0.5B router | Tuned joint accuracy was 100% on generated tests, but 6/12 on independent challenges; every escalation challenge failed | [SLM results](SLM_RESULTS.md) |
| D02 | Fine-tuned JSON query AST with bounded execution | Tuned semantic correctness was 90.63%, falling to 62.5% on new compositions; the typed parser won generated tasks but covered only 1/8 challenges | [SLM results](SLM_RESULTS.md) |
| D03 | Actual TRL GRPO with public execution rewards | Fifty updates changed weights, but only two reward groups varied; held-out accuracy matched SFT and the control trained for the same wall time | [RLVR results](RLVR_RESULTS.md) |
| D04 | Local device-command adapter and quantized export | Actual offline inference worked under OS-denied networking; schema-valid commands still sometimes misunderstood user intent | [SLM results](SLM_RESULTS.md) |
| D05 | Tabular anomaly prediction with calibration and drift | XGBoost beat logistic average precision across three seeds, but future FPR rose to 10.3–13.4% from a 5% validation budget | [CPU results](CPU_RESULTS.md) |
| D06 | Native and ONNX request classifiers plus localhost HTTP | Logistic won validation; parity error stayed below 2.4e-7; measured HTTP p95 was 0.34–0.43 ms at concurrency 1 | [CPU results](CPU_RESULTS.md) |
| D07 | Actual TreeSHAP and coefficient explanations with feature edits | Additivity passed on 100 held-out rows; 196 small edits reproduced exactly; duplicate predictors illustrated non-unique attribution despite identical predictions | [CPU results](CPU_RESULTS.md) |
| D08 | Separately trained bi-encoder and cross-encoder | Tuned retrieval reached 99.3% top-1; adding the tuned reranker reduced the best pipeline to 88.9%, despite improving its fixed-pool comparison | [Retrieval results](RETRIEVAL_RESULTS.md) |
| D09 | No-training baseline, actual hybrid, eight component ablations and live knowledge update | Hybrid solved 27/32 versus baseline 19/32, used the teacher on 43.75% of requests, and had higher measured p95 latency | [Hybrid results](HYBRID_RESULTS.md) |

The PDF's first list and later overlapping pipeline examples were mapped to these eight source categories plus the capstone. Its embedded prompts were treated as source material, not instructions. Its illustrative speed, scale and fallback claims were not adopted as acceptance thresholds or presented as observed results.

## What actually ran

All eight source categories have actual evaluated local implementations, and the hybrid capstone completed its full comparison. Clean-environment artifact replay passed for every track, the comparison UI passed, and the final evidence audit passed all eight check groups. P0–P6 are complete within the local educational scope. [CONFORMANCE.md](CONFORMANCE.md) maps each original requirement to evidence; [PROGRESS.md](PROGRESS.md) records completion.


D01 used 2,016 local teacher generations, retaining 1,974 after independent validation, and executed 494 student optimizer steps. D02 executed 500 steps; D04 executed 256 and exported a fused 4-bit model. D03 executed 50 GRPO steps and 290 continued-SFT steps from the identical initial checkpoint/adapter, then evaluated base/SFT/RL/control with matched greedy and four-sample inference budgets. D08 executed 106 bi-encoder and 158 cross-encoder steps, selecting checkpoints using validation before final tests.

D05 and D06 trained and persisted real classical models across three seeds. D06 executed the exported ONNX model and sent real loopback HTTP requests. D07 recomputed explanations from D05's saved weights. D09 ran 320 requests across ten complete pipelines, including all branches, explicit authorization, rejected outputs and fallbacks. A separate corpus edit changed an actually regenerated answer without retraining model weights.

Failed and smoke runs remain visible and are excluded from these completion claims. Two MLX batch-8 structured runs failed with Metal errors before succeeding at batch 4. The longer MPS RL run was rejected for unraised command-buffer errors; the accepted run used CPU with unchanged data and hyperparameters. No wrapper, mock output or cached-answer replay is counted as a trained or evaluated model.

## Completed validation

The final audit passed selected-artifact/training checks, all-track reproduction, actual hybrid branch coverage, comparison UI checks, 94 contract tests plus the separate public-only RL selection contract, resource profiles, small-perturbation explanation reproduction, and walkthrough/report presence. It verifies that supplemental measurements refer to the selected run hashes. [Machine-readable audit](../artifacts/validation/final-audit.json) retains each result.

D03's second locked environment regenerated exact greedy and sampled outputs for all four variants on 96 test tasks and eight challenges. Its dedicated CPU profile then measured warm inference and separate fresh processes. D06's supplemental profile compared native and ONNX execution on identical batch-1 and batch-256 inputs across three seeds, with maximum probability difference 8.94e-8 on those batches. Both backends met the provisional 1 ms batch-1 model-only p95 target. D07's 196 conditional ±1% amount edits reproduced exactly; 83.67% left probability unchanged. These supplements preserve the original frozen experiment artifacts.

## Runtime and evidence boundaries

The observed host is Apple M2 Max with 96 GB RAM. CPU, retrieval, MLX, RLVR and UI dependencies are pinned separately. Model revisions, licenses, prompts, training settings, source/data hashes, saved artifacts, raw predictions, failures and measurements are recorded in the run directories. Generated data and artifacts stay local and are excluded from Git; source and small fixtures are versioned, with no remote publishing performed.

Most data is generated from bounded synthetic contracts, supplemented by small independently authored challenge sets. Three cheap CPU seeds provide a limited replication check; neural training uses a single pilot seed. A second installed environment can test reload/inference reproducibility but does not replace independent retraining, a different machine, real customer data or operational validation.

Warm inference, fresh-process startup, teacher batch generation, training time and HTTP service latency are reported separately. Model size and adapter storage are measured where applicable. OS file caches are uncontrolled, timing samples are small, and GPU/CPU results are not interchangeable. The capstone keeps several models resident for comparisons, so its process peaks are not isolated deployment memory estimates. No hosted model API was called; local electricity and hardware depreciation were not measured.

The AST interpreter executes only a bounded declarative language. Device commands affect a fresh simulator. Authorization and schema/range checks reject invalid actions, but valid outputs can still have wrong semantics. Attribution describes a specified model under stated assumptions and does not establish causality. The future-period tabular drift and capstone false rejections show why synthetic success cannot establish production readiness.

## Use the lab

Start with the [learning guide](LEARNING_GUIDE.md) and an individual tutorial; use the [reproduction guide](REPRODUCTION.md) when running it. The [implementation plan](IMPLEMENTATION_PLAN.md) explains the original experimental design. The [local comparison UI](COMPARISON_UI.md) reads the same artifacts as the CLI, separates saved results from fresh execution, exposes failure examples and supports real model queries, feature edits and hybrid stage traces.

```bash
.venv/bin/ml-demo list
.venv/bin/ml-demo report all
environments/ui/.venv/bin/streamlit run app/main.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true \
  --browser.gatherUsageStats false
```

Open `http://127.0.0.1:8501`. For a device request, inspect the simulator before/after states. For a query, inspect semantic checks and raw AST. For the hybrid, compare baseline and ablations alongside teacher usage, latency and failed requests. For explanations, edit a held-out row and observe both the recomputed score and contribution changes.

The generated [evidence index](../artifacts/report.md) lists every run with its status and manifest. The [passed final audit](../artifacts/validation/final-audit.json), [requirement checklist](CONFORMANCE.md) and [reproduction guide](REPRODUCTION.md) provide the completion evidence. Completion covers implemented and evaluated demos, including honest negative findings; it does not establish effectiveness on real customer data or production hardware.

## Five ideas to take away

1. **Learning a format is different from learning a task.** D02's parse, schema and semantic scores separate those layers; D04's valid-but-unintended actions show why that distinction matters.
2. **A useful baseline can be very small.** A typed parser won generated command tasks, a logistic model won classifier validation, and the SFT query model matched RL. Their narrower assumptions should be explicit.
3. **The evaluation distribution determines the claim.** D01's generated-test success and challenge failures are compatible. D05's changed future FPR exposes a threshold's dependence on the population.
4. **More computation needs an incremental benefit.** D03 spent training work without a measured gain; D08's extra reranking stage hurt the strongest pipeline; D09's lower teacher usage did not imply lower p95.
5. **Evidence has a scope.** Export parity, offline execution, additivity and exact replay establish useful engineering properties. None substitutes for representative task evaluation or production operation.

As a final exercise, choose a real application and write a one-page design using the [architecture decision guide and record template](ARCHITECTURE_DECISION_GUIDE.md#write-a-small-architecture-decision-record), then develop its operating consequences with the [production tradeoff guide](PRODUCTION_TRADEOFFS.md). Name the simplest baseline, the failure you expect training to address, your quality and operating budgets, and the evidence that would make you abandon the more complex option. Use the lab's results as examples of reasoning, not as accuracy or latency forecasts for your application.
