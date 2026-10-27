# CPU demo results

Use this chapter as the worked case study for [D05–D07](CPU_DEMOS.md). First predict which model should win and which operational constraint could still fail. Then read the tables in order: ranking quality, exported execution, and explanation fidelity. The measurements below are retained from the recorded runs; the interpretation questions do not introduce new experiments.

Observed locally on 2026-10-26. These results establish functioning synthetic-data demos, not production effectiveness.

## Validation

- Apple M2 Max, 96 GB RAM, macOS arm64, Python 3.12.13.
- The initial CPU-specific suite passed 25 contract tests in a second virtual environment installed from the frozen lockfile; the completed lab's final audit passed all 94 contract tests.
- Three D05 runs, three D06 runs, and one D07 run reproduced from saved artifacts in that environment.
- Six real loopback HTTP cases passed: a valid request and five invalid request variants.
- [Machine-readable reproduction evidence](../artifacts/validation/cpu-with-http.json).
- [Sanitized environment](../artifacts/environment.json).

## D05: tabular anomaly scoring

Each seed generates 50,000 synthetic events, with 30,000 for training, 5,000 for calibration, 5,000 for selection, and 10,000 for future-period evaluation. The generator has stochastic labels, unobserved entity effects, missing values and drift. Models and thresholds are selected without using test labels.

| Seed | Test prevalence | Dummy AP | Logistic AP | XGBoost AP | Validation winner | XGBoost test recall | XGBoost test FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 41 | 0.0357 | 0.0357 | 0.1039 | 0.1215 | xgboost | 0.3417 | 0.1031 |
| 42 | 0.0448 | 0.0448 | 0.1295 | 0.1407 | xgboost | 0.3795 | 0.1135 |
| 43 | 0.0436 | 0.0436 | 0.1085 | 0.1181 | xgboost | 0.3555 | 0.1335 |

XGBoost has higher held-out average precision in all three runs. This is a bounded result on this synthetic generator. The future false-positive rate is allowed to differ from the 5% validation budget; the table exposes that shift. Low recall illustrates why this is not a deployable fraud detector.

## D06: fast request classification

Each seed trains logistic and tree models on a separate 50,000-row synthetic dataset. The table uses the validation-selected model. HTTP and inference measurements are sequential, concurrency 1; ONNX has one intra/inter-op thread.

| Seed | Winner | Test AP | ONNX max error | Warm ONNX p95 ms | HTTP p95 ms | Fresh process seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 41 | logistic | 0.3086 | 2.38e-07 | 0.005750 | 0.343621 | 1.104 |
| 42 | logistic | 0.3062 | 1.79e-07 | 0.005377 | 0.426432 | 1.142 |
| 43 | logistic | 0.3030 | 1.79e-07 | 0.004502 | 0.385930 | 1.190 |

Parity tolerance is 1e-5 on all 10,000 held-out rows. Warm timings use repeated fixed inputs to isolate overhead; they are not production service-level guarantees. Fresh-process timing includes interpreter startup, imports, loading and first inference; OS caches are uncontrolled. The recorded fresh-process figures overstate ONNX startup: the child process imported the training module, so most of each figure is loading scikit-learn, skl2onnx and polars, which an ONNX scorer does not need. Re-measured on the same host and the same three saved models, five fresh processes each, the minimal scorer (onnxruntime, numpy and joblib only) had a median of 0.096 s for every seed, while the old import path took 1.03–1.04 s. New runs use the minimal scorer. See `artifacts/validation/onnx-fresh-process.json` and [`scripts/profile_onnx_startup.py`](../scripts/profile_onnx_startup.py). Startup cost depends on what the process imports, not only on the model. HTTP includes serialization, validation and a new loopback connection per request.

A supplemental profile measured both native and ONNX execution on identical inputs at batch sizes 1 and 256, after the other model work stopped. The original run artifacts were preserved. Each pair used 20 warmups and 200 measurements, concurrency 1, single-thread BLAS/OpenMP and one ONNX intra/inter-op thread; preprocessing is included.

| Seed | Batch size | Native p95 ms per batch | ONNX p95 ms per batch | Native rows/s from mean | ONNX rows/s from mean |
| --- | --- | --- | --- | --- | --- |
| 41 | 1 | 0.118833 | 0.004627 | 9,534 | 240,352 |
| 42 | 1 | 0.110435 | 0.004502 | 9,605 | 247,573 |
| 43 | 1 | 0.106730 | 0.004166 | 9,889 | 249,857 |
| 41 | 256 | 0.136308 | 0.008958 | 2,171,105 | 29,447,136 |
| 42 | 256 | 0.117146 | 0.008375 | 2,275,305 | 31,044,320 |
| 43 | 256 | 0.115925 | 0.008375 | 2,354,644 | 31,049,930 |

The maximum native/exported probability difference in these fixed batches was 8.94e-8. [Raw paired profile](../artifacts/validation/classifier-batches.json) records p50/p95/p99, means, conditions and the unchanged source-manifest hashes. These large batched row rates describe a tight in-process loop over repeated inputs; they exclude HTTP, traffic variability and production concurrency. Both backends met the provisional batch-1 model-only p95 target of 1 ms on this host. The separate measured HTTP distribution above remains the service observation.

## D07: feature attribution

TreeSHAP reconstructed raw model log-odds for 100 held-out rows with maximum residual 3.03e-06, below the 1e-5 tolerance. The background contains 100 training rows; interventional feature perturbation is explicit.

For held-out row 7, changing amount to 450 changed calibrated risk from 0.0120 to 0.0345. The artifacts contain complete before/after contributions and signed reason codes.

A separate small-perturbation study selected the first 100 held-out rows, skipped two with missing amounts, and applied both -1% and +1% amount changes to the remaining 98 rows (196 edits). Calibrated probabilities were unchanged for 83.67% of edits; the maximum absolute probability change was 0.004564 (0.456 percentage points). The p95 attribution L1 change was 0.1024 raw-log-odds units, with maximum 0.3479; the top absolute-contribution feature changed in 2/196 edits. The maximum error reconstructing the raw-margin change from contribution changes was 5.33e-7. These exact results reproduced in the separate CPU environment. See `artifacts/validation/explanation-stability.json` and `explanation-stability-reproduced.json`. Amount alone was edited, holding history and other features fixed; this is a local sensitivity measurement, not a causal or general robustness claim.

The duplicate-feature example produced prediction difference 0 while moving attribution between identical predictors. This demonstrates why attribution is not causal identification. Logistic coefficients include their training standardization context.

## Exact evidence runs

- tabular-41: [report](../artifacts/tabular/local-s41-20261026T163036Z-3c6401/report.md), [manifest](../artifacts/tabular/local-s41-20261026T163036Z-3c6401/run.json).
- tabular-42: [report](../artifacts/tabular/local-s42-20261026T163041Z-2b586b/report.md), [manifest](../artifacts/tabular/local-s42-20261026T163041Z-2b586b/run.json).
- tabular-43: [report](../artifacts/tabular/local-s43-20261026T163045Z-2328af/report.md), [manifest](../artifacts/tabular/local-s43-20261026T163045Z-2328af/run.json).
- fast_classifier-41: [report](../artifacts/fast_classifier/local-s41-20261026T163410Z-e369ea/report.md), [manifest](../artifacts/fast_classifier/local-s41-20261026T163410Z-e369ea/run.json).
- fast_classifier-42: [report](../artifacts/fast_classifier/local-s42-20261026T163413Z-7840a0/report.md), [manifest](../artifacts/fast_classifier/local-s42-20261026T163413Z-7840a0/run.json).
- fast_classifier-43: [report](../artifacts/fast_classifier/local-s43-20261026T163417Z-e84941/report.md), [manifest](../artifacts/fast_classifier/local-s43-20261026T163417Z-e84941/run.json).
- explainability-41: [report](../artifacts/explainability/local-s41-20261026T163345Z-d633f1/report.md), [manifest](../artifacts/explainability/local-s41-20261026T163345Z-d633f1/run.json).

## Other tracks and retained failures

Later milestones have their own [retrieval results](RETRIEVAL_RESULTS.md), [language-model results](SLM_RESULTS.md), and [comparison UI](COMPARISON_UI.md). [PROGRESS.md](project/PROGRESS.md) records current completion and reproduction status.

The earliest smoke/local tabular attempts failed while writing the final manifest because XGBoost represents missing values with NaN. The serialization fix is in place; those attempts remain explicitly marked failed. A socket-restricted classifier smoke run retains a blocked HTTP measurement. Neither is used in the tables above.

## Interpreting the three lessons together

The tabular model, exported classifier and explanation answer different questions. D05's AP improvement is evidence about ranking on a future synthetic period. D06's parity result is evidence that two implementations agree numerically. D07's additivity result is evidence that an explanation reconstructs the chosen model output. None can stand in for the others.

| Observation | Supported interpretation | Tempting inference to avoid |
| --- | --- | --- |
| XGBoost wins AP across three seeds | The tested generator contains signal this configuration captures better than logistic regression | The threshold will retain its 5% FPR budget later |
| ONNX reproduces probabilities and is fast locally | Export is faithful within tolerance and useful on this measured workload | The same row rate is available as HTTP throughput |
| SHAP additivity passes | Contributions reconstruct raw log-odds under the selected setup | The model is correct, fair or causally explained |
| Most small amount edits leave probability unchanged | Those edits usually stayed within regions with the same calibrated output | Amount is irrelevant or the model is globally robust |

**Check your understanding:** would lowering the D05 threshold necessarily increase AP? It changes decisions and therefore recall/FPR, but AP is computed from the score ranking across thresholds. Would changing the explanation background necessarily change the model prediction? No: the explanation can change while the predictor stays fixed. These distinctions are more reusable than the exact benchmark numbers.

Before proposing production use, choose a failure budget and a representative population, then use the [production guide](PRODUCTION_TRADEOFFS.md) to design the next evaluation.
