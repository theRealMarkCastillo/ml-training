# Implementation Progress

This is the execution record for the learning lab. Use it to practice distinguishing implemented code, completed training, evaluated behavior and reproduced artifacts. For the tutorial reading order, start with [LEARNING_GUIDE.md](../LEARNING_GUIDE.md). Dates and counts below describe the recorded implementation; they are not live status for a reader's new checkout.

Updated: 2026-10-26. P0–P6 are complete and the final evidence audit passed.

All eight source-category experiments and the D09 capstone have actual evaluated local runs. Saved-artifact reproduction passed in separate locked environments for every track. The D03 warm/cold resource profile, paired native/ONNX batch measurements and exact explanation-sensitivity replay also passed. [FINAL_REPORT.md](FINAL_REPORT.md) summarizes the observed results and their limits.

## Milestones

- [x] Map all distinct categories from all six PDF pages, treating embedded prompts as source material.
- [x] P0: hardware, dependency and update/save/reload probes; frozen datasets and contracts.
- [x] P1: D05 tabular prediction, D06 CPU/ONNX classification and D07 explanations.
- [x] P2: D08 separately trained bi-encoder and cross-encoder.
- [x] P3: D01 teacher distillation, D02 AST tuning and D04 offline device interpreter.
- [x] P4: D03 actual GRPO, hidden evaluation and training-time-matched SFT control.
- [x] P5: D09 baseline, component integration, ablations and comparison UI.
- [x] P6: all-track clean replay, resource supplements, walkthroughs, evidence index, final audit and report review.

## Evidence ledger

The selectors under `artifacts/validation/selected_*.json` identify exact completed runs. [CONFORMANCE.md](CONFORMANCE.md) maps each original requirement to evidence. [REPRODUCTION.md](../REPRODUCTION.md) distinguishes artifact replay from retraining.

| Demo | Observed status | Findings and evidence |
| --- | --- | --- |
| D01 distillation | Evaluated and reproduced | 494 optimizer steps; 100% generated-test joint accuracy, 6/12 independent challenges and all four escalation challenges failed. [SLM results](../SLM_RESULTS.md). |
| D02 schema/DSL | Evaluated and reproduced | 500 steps; 90.63% semantic correctness, 62.5% on unseen compositions. Typed baseline wins its bounded grammar but covers only 1/8 challenges. [SLM results](../SLM_RESULTS.md). |
| D03 RLVR | Evaluated, reproduced and profiled | 50 actual GRPO updates, two nonconstant reward groups; no held-out gain over SFT or time-matched control. [RLVR results](../RLVR_RESULTS.md). |
| D04 edge | Evaluated, reproduced and enforced offline | 256 steps and real quantized export; OS-denied networking followed by fresh model inference. Schema-valid wrong intent remains possible. [SLM results](../SLM_RESULTS.md). |
| D05 tabular | Three seeds evaluated and reproduced | XGBoost led AP; future FPR exceeded the validation budget. [CPU results](../CPU_RESULTS.md). |
| D06 classifier | Three seeds evaluated, reproduced and profiled | Actual ONNX parity and localhost HTTP measurements. Native/ONNX batch-1 and batch-256 paired supplement passed. [CPU results](../CPU_RESULTS.md). |
| D07 explanations | Evaluated and reproduced | Additivity on 100 held-out rows; fresh feature edit; correlated-feature example; 196 small edits reproduced exactly in a second environment. [CPU results](../CPU_RESULTS.md). |
| D08 retrieval | Evaluated and reproduced | Both encoders trained; all seven comparisons reproduced. Tuned bi-encoder alone beat the reranked pipeline. [Retrieval results](../RETRIEVAL_RESULTS.md). |
| D09 hybrid | Evaluated and reproduced | All 320 requests and live knowledge update regenerated in clean workers. Hybrid 27/32 versus baseline 19/32, with higher measured p95. [Hybrid results](../HYBRID_RESULTS.md). |

## Validation and execution boundaries

- The final audit passed 94 tests, covering leakage, verifier failures, artifact tampering, invalid device actions, export input contracts and attribution edits, plus the separate public-only RL candidate-selection check.
- `cpu-with-http.json` records seven saved-model reproductions and six actual loopback HTTP cases. `retrieval.json` records all seven retrieval comparisons. `slm.json` records exact outputs/metrics for D01/D02/D04 variants. `edge-offline.json` records actual inference with OS-denied networking.
- `hybrid.json` records 320 actual clean-worker executions plus a corpus update with unchanged model hashes. Runtime receives public request fields only; hidden expected outcomes are used only for evaluation.
- `explanation-stability.json` and `explanation-stability-reproduced.json` match exactly for 196 conditional ±1% amount edits. Missing amounts are explicitly skipped; this is sensitivity evidence, not causal inference.
- Nine UI artifact views and seven fresh model/edit forms passed. `ui-browser.json` and screenshots show the comparison, a new 23-degree simulator request, and a D07 feature edit. The UI and CLI use the same saved artifacts.
- Accepted hardware is Apple M2 Max, 96 GB RAM. Root CPU, retrieval, MLX, RLVR and UI environments are pinned separately. The actual RL run uses CPU; rejected Metal attempts remain failed artifacts. Model cards, pinned revisions and license sources are recorded in `configs/models.json`.
- Neural results use one training seed on bounded synthetic data; cheap CPU tracks use three seeds. A second installed environment checks reload/inference, not independent retraining, another host or production readiness.
- Source and small fixtures are versioned; no remote publishing has been performed. Generated data, teacher cache, checkpoints and validation evidence remain local. Device actions affect only a simulator.

## Completion audit

`artifacts/validation/final-audit.json` passed all eight groups: selected artifacts/training, all-track reproduction, actual hybrid branch coverage, UI, contract tests, resource profiles, small-perturbation stability, and walkthrough/final-report presence. The evidence index is `artifacts/report.md`.

The final review retained the original acceptance criteria, checked local documentation links, and confirmed that results and limitations agree with the recorded measurements. There is no required implementation or validation work left in the agreed educational scope. New real-data evaluation, independent neural retraining and production deployment would be separate work.

The subsequent [code review](CODE_REVIEW.md) corrected four implementation/validation issues and added 16 regressions. All 94 tests, seven actual fresh forms, nine artifact views and the eight audit groups passed afterward. Corrected UI checks capture the submitted request's CLI response; four old evidence records that had selected saved examples are retained in `artifacts/validation/pre-review/`. Frozen model results and all 1,360 parser comparison cases remain unchanged.

## What progress means in an ML project

A completed training job is not yet an evaluated model, and a reproduced evaluation is not yet a production result. Follow one row in this tracker into its result chapter and identify the evidence for each transition. For a failed run, identify whether the problem was environment compatibility, execution, validation or task quality.

The educational revision adds conceptual foundations, exercises and production reasoning around the preserved experiment records. Proposed extensions in those lessons remain exercises until separately executed and measured.
