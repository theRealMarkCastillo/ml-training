# D05–D07: prediction, serving and explanations

Visual companions: [D05 notebook](../notebooks/D05_tabular_prediction.ipynb), [D06 notebook](../notebooks/D06_inference_optimization.ipynb), [D07 notebook](../notebooks/D07_feature_attribution.ipynb). These plot the recorded evidence and include editable examples without loading models; see the [notebook guide](../notebooks/README.md).

These three lessons form a first learning path through applied ML. D05 asks whether a model can rank risky events; D06 asks how to serve a classifier efficiently; D07 asks what an explanation of a prediction actually means. You will train and reload real models, inspect errors, and practice separating model quality from deployment behavior.

Read the [learning guide](LEARNING_GUIDE.md) for metric definitions. Start with D05 and a single seed; repeat the three-seed experiments after you understand the artifacts. D07 requires a completed D05 run. All datasets are synthetic, and the [results chapter](CPU_RESULTS.md) is a worked interpretation of the recorded experiments.

## Setup

Use Python 3.12 and `uv`. On macOS, XGBoost additionally requires OpenMP:

```bash
brew install libomp
uv sync --frozen
source .venv/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ml-demo doctor
ml-demo list
ml-demo verify all --profile smoke
```

`doctor` writes a sanitized environment report to `artifacts/environment.json`. `verify` without a run directory runs contract tests; it does not train models or reproduce their measured scores. `uv.lock` pins the CPU environment. The neural lessons use separate locked environments. All commands below run from the repository root; replace `<run>` with the path printed by training.

## D05: learn a tabular risk function

By the end of this lesson, you should be able to explain why a better ranking does not guarantee a usable alerting policy.

### The concept: scores, probabilities and decisions

A logistic model learns a weighted sum of input features and maps it through a sigmoid: `p = 1 / (1 + exp(-z))`. Its additive score is a useful baseline. A boosted tree ensemble adds trees that fit remaining prediction errors through an optimization objective; it can represent interactions such as an amount being unusual only for a particular activity pattern. [XGBoost's original paper](https://arxiv.org/abs/1603.02754) explains the tree-boosting system.

This demo compares both with a constant prior. It is **supervised binary risk prediction** using synthetic labels, even though some reports call the use case anomaly scoring. It is not an unsupervised outlier detector.

There are three separate operations:

1. Fit a model that orders events by risk.
2. Fit a sigmoid calibration mapping on a later, separate slice.
3. Choose a threshold on validation data, maximizing recall under an empirical 5% false-positive budget.

The calibrator consumes the model's raw probability converted to log-odds. It does not retrain the tree. The threshold turns a score into an action. These steps have separate datasets because fitting and evaluating the same adjustment on the same rows makes the result optimistic.

### Why time matters

Imagine predicting an event at 10:00. A count of earlier events is available; a count including 10:05 is not. Even an otherwise ordinary rolling feature leaks information if its window includes the future. The implementation orders events and constructs history using strictly earlier observations.

~~~mermaid
flowchart LR
    A["Earlier 60%: fit model"] --> B["Next 10%: fit calibration"]
    B --> C["Next 10%: select model and threshold"]
    C --> D["Latest 20%: evaluate drift"]
~~~

Entity IDs are excluded as predictors, while returning and new entities can still behave differently. A time split answers a future-period question; a random row split could mix those periods and hide drift.

### Run and inspect

```bash
ml-demo prepare tabular --profile local --seed 41
ml-demo train tabular --profile local --seed 41
ml-demo train tabular --profile local --seed 42
ml-demo train tabular --profile local --seed 43
```

Every train command prints its unique run directory. Substitute one of those paths below:

```bash
ml-demo evaluate tabular --run-dir artifacts/tabular/<run>
ml-demo verify tabular --run-dir artifacts/tabular/<run>
```

The dataset contains 50,000 time-ordered events. The first 60% trains the model, the next 10% calibrates probabilities, the next 10% selects thresholds and the winning model, and the final 20% evaluates future drift. Recurring entities are allowed but their IDs are excluded from predictors. New entities in the final period are evaluated separately. Rolling metadata features use only strictly earlier events; event labels never enter them.

Read `report.md` and `metrics.json` for dummy, logistic, and XGBoost results. Average precision should be interpreted relative to prevalence. A 5% validation false-positive budget is not a guaranteed future rate. `selection.json` records the choice before test evaluation. `predictions.parquet` contains all held-out labels and predictions; `failure_examples.json` retains actual errors. The independently authored challenge pair checks plausibility, not empirical outcome accuracy.

All three models are saved and reloaded before evaluation. `evaluate` reconstructs the held-out metrics from those artifacts and checks exact agreement. The model files contain a training-fitted preprocessing pipeline and, for nonconstant models, a sigmoid calibrator. Load only locally produced/trusted joblib artifacts.

### What was learned, and where it applies

The [recorded results](CPU_RESULTS.md#d05-tabular-anomaly-scoring) show higher XGBoost test AP than logistic AP across three seeds. Yet future-period FPR rose to 10.3–13.4%, despite the 5% validation budget. The model learned a useful ordering on this generator while the chosen operating point failed to preserve its earlier error rate.

For production, compare the operational cost of missed positives with the burden of false alerts. Recalibrating probabilities and changing a threshold are different responses to drift; neither repairs a feature whose meaning changed. A simple logistic model may remain preferable when the measured tree gain is too small to cover maintenance, explanation or monitoring needs.

### Try it and reason about the result

1. In `metrics.json`, compare AP with prevalence for each model, then compare test recall/FPR with the frozen validation threshold. Explain why “higher AP” does not imply “fewer false alerts at this threshold.”
2. Use the seed-41 table to estimate flagged test negatives: `(10,000 - 357) × 0.1031 ≈ 994`. Compare that with roughly `357 × 0.3417 ≈ 122` detected positives. Rounded published rates make these approximate counts. The point is to translate rates into workload.
3. As a separate extension, compare a random split with the chronological split while keeping the original result intact. Predict which sources of leakage or distribution mixing could create an apparent gain. There is no promised numerical outcome; inspect the time and entity composition.

Trace the implementation through [data.py](../src/ml_training/data.py), [tabular.py](../src/ml_training/tabular.py) and [metrics.py](../src/ml_training/metrics.py). Find where preprocessing is fitted, calibration is fitted, and the threshold is selected before reading the test-evaluation loop.

## D06: export and time a request classifier

By the end, you should be able to distinguish a model export's numerical fidelity from its performance benefit.

### The concept: export changes execution

ONNX represents a computation for another runtime to execute. Exporting the selected logistic model does not teach it new behavior. The relevant first question is whether the native and exported paths compute sufficiently close probabilities, including preprocessing and feature order. Only after checking parity is the timing comparison meaningful.

A small classifier is a useful example because runtime overhead can dominate its arithmetic. Faster matrix execution can produce a large model-only speedup while JSON decoding, validation, connections and queueing still dominate the application. ONNX Runtime exposes threading, profiling and other performance controls; their benefits depend on the workload. [Official performance guide](https://onnxruntime.ai/docs/performance/tune-performance/).

### Run and inspect

```bash
ml-demo train fast_classifier --profile local --seed 41
ml-demo train fast_classifier --profile local --seed 42
ml-demo train fast_classifier --profile local --seed 43
ml-demo evaluate fast_classifier --run-dir artifacts/fast_classifier/<run>
```

This uses a separate 50,000-row request dataset, training a logistic model and a small decision tree. Validation chooses the exported model. ONNX Runtime must reproduce native probabilities on all held-out rows within 1e-5. `model.onnx` contains the selected model, including scaling when applicable.

`benchmarks.json` separates batch-1 inference, batch-256 throughput, session loading, a fresh-process first inference, and JSON-over-HTTP latency. The HTTP server binds to loopback on an ephemeral port, handles 120 warmup/measured requests, and shuts down at the end. If local socket access is denied, the report marks that measurement blocked. Re-run with local socket access to complete it.

Batch timing uses fixed inputs to isolate execution overhead; it does not model production traffic variability. Throughput at batch 256 is not HTTP request throughput. The fresh-process check includes Python imports and model/input loading, but does not flush the OS file cache. Warm benchmarks use concurrency 1 and one ONNX intra/inter-op thread. Avoid concurrent training while collecting comparison timings.

For a complete paired native/ONNX profile at both batch sizes on an existing saved run, use:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-repro/bin/python scripts/profile_classifier_batches.py \
  artifacts/fast_classifier/<run> --output artifacts/validation/classifier-batches.json
```

You may pass all three seed run directories before `--output`. This writes a supplemental measurement and preserves the original hashed run. Future training runs also include native batch-256 timing directly.

### What was learned, and where it applies

Logistic regression won validation across the three recorded runs. Native/exported probability differences were below 2.4e-7 on held-out rows. The paired profile found much lower ONNX model-only latency, while the HTTP observations were larger than model inference and startup was larger again. See [the CPU results](CPU_RESULTS.md#d06-fast-request-classification) for each measurement boundary.

A production decision would also measure realistic arrival rates, batching delay and concurrency. Larger batches improve amortization but may make an individual request wait longer. Runtime conversion adds compatibility and release work, so retaining native inference is reasonable if it already meets the application's budget.

### Try it and reason about the result

1. Compare the reported model-only p95, HTTP p95 and fresh-process seconds. Identify the extra work in each. Do not subtract percentiles from different distributions and call the result an exact overhead.
2. Compare batch-1 and batch-256 **time per batch**, then calculate rows per second from mean duration. Explain why a million rows per second in this loop is not a million HTTP requests per second.
3. Inspect the threshold alongside parity. A probability difference below tolerance can still flip a label arbitrarily close to the threshold. Propose additional near-threshold fixtures before changing runtime or precision.

Read [fast_classifier.py](../src/ml_training/fast_classifier.py) from model selection through export, parity and the localhost handler. The [input-contract tests](../tests/test_fast_classifier.py) show why a numerical feature vector needs a defined interface.

## D07: inspect and change an explanation

By the end, you should be able to state what an attribution adds up to and why a feature edit is not a causal intervention.

### The concept: allocate a model output

SHAP assigns feature contributions relative to a reference expectation. In this lesson, the decomposition is:

`raw tree log-odds(x) = expected raw log-odds + sum(feature contributions)`

A positive contribution raises this model's raw score relative to the chosen background. A negative contribution lowers it. The [SHAP paper](https://arxiv.org/abs/1705.07874) gives the additive explanation framework.

Output space matters. For an illustrative raw margin of -2, the uncalibrated probability is about 0.119. Adding +1 to the margin makes the probability about 0.269, not 1.119 and not “one percentage point higher.” The actual demo additionally calibrates probabilities, so it displays that prediction separately from the raw-margin explanation.

TreeSHAP needs a choice about feature dependence and reference data. Here the implementation explicitly uses interventional perturbation, 100 training background rows and raw output. Changing those choices changes the question being answered. [TreeExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).

### Run and inspect

```bash
ml-demo run explainability --source-run artifacts/tabular/<run> --row-index 7 --set amount=450
ml-demo evaluate explainability --run-dir artifacts/explainability/<run>
```

Select any held-out row by zero-based index. You can repeat `--set` to change named model features. The command writes before/after predictions, signed feature contributions, and the amount that attributions changed. Invalid fields and out-of-range values are rejected.

The explanation reconstructs XGBoost's uncalibrated log-odds, while the user-facing calibrated probability is displayed separately. Its training-only background has 100 rows. Additivity must hold within 1e-5. Logistic coefficients include the numeric training means and scales so a standardized coefficient is not mistaken for an effect per raw unit.

`correlated_features.json` is a separate toy example with identical predictors. It illustrates how equivalent predictions can allocate attribution differently. Editing a feature does not regenerate its causal history; these are model explanations, not causal or compliance claims.

Measure local explanation stability using both -1% and +1% amount edits on the first 100 held-out rows:

```bash
.venv/bin/python scripts/validate_explanation_stability.py \
  artifacts/explainability/<run> --output artifacts/validation/explanation-stability.json
.venv-repro/bin/python scripts/validate_explanation_stability.py \
  artifacts/explainability/<run> --compare artifacts/validation/explanation-stability.json \
  --output artifacts/validation/explanation-stability-reproduced.json
```

Missing amounts are counted and skipped rather than silently imputed for the edit. The report records probability changes, contribution L1 changes, top-feature changes and reconstruction of the raw-margin delta. The second invocation requires exact reproduction of every measured record.

### What was learned, and where it applies

In the [recorded example](CPU_RESULTS.md#d07-feature-attribution), changing row 7's amount to 450 moved calibrated risk from about 0.0120 to 0.0345. Raw-margin additivity passed. Most ±1% amount edits left probability unchanged, which is compatible with a tree model's piecewise behavior; it is not evidence that the model ignores amount everywhere.

The duplicate-feature example produced identical predictions with different attribution allocations. Attribution depends on the fitted representation and reference assumptions. It cannot establish which of two correlated features caused the real outcome.

In production, define the consumer: developer debugging, analyst review and an explanation to an affected user require different interfaces. Evaluate stability and usefulness. A perfectly additive explanation can faithfully describe a poor predictor, and a plausible reason code does not validate a decision policy.

### Try it and reason about the result

1. Reconstruct one raw prediction by summing the expected value and signed contributions. Then inspect the separate calibrated probability. Explain why summing contributions in probability space is incorrect here.
2. Compare a feature's contribution before/after an edit with the **total** margin change. Other contributions may change too, so the edited feature's delta alone need not equal the prediction delta.
3. Inspect `correlated_features.json`. Write a conclusion that preserves both observations: predictions matched, explanations differed.
4. Propose a new background population and predict what question it would answer. Recompute it only as a separately labeled extension; the original additivity result used the original training background.

Read [explainability.py](../src/ml_training/explainability.py), especially `setup`, `explain_rows` and `modify_row`. The edit holds history fixed: changing the amount does not rebuild the event's real-world causal history.

## Evidence and reproducibility

```bash
ml-demo report all
```

This verifies complete run checksums and creates `artifacts/index.json` and `artifacts/report.md`. Failed/incomplete runs remain visibly distinct. Each completed run records dependencies, source file hashes, lockfile hash, dataset checksums, seed, model settings, timing and peak process memory. Generated datasets are frozen by checksum; a changed generator requires an explicit dataset version change rather than silently replacing held-out data.

Use `--profile smoke` for tiny 2,000-row checks. Smoke metrics are never promoted into local evaluation evidence. For a clean-environment reproduction, create a second virtual environment from `uv.lock`, install the package, run the tests, then re-evaluate the saved artifacts. Re-training with the same seed can be compared separately; timings can differ even when predictions reproduce.

For an architectural choice, compare the use/avoid criteria and alternatives for [D05](ARCHITECTURE_DECISION_GUIDE.md#d05--predict-from-tabular-features), [D06](ARCHITECTURE_DECISION_GUIDE.md#d06--optimize-inference-execution) and [D07](ARCHITECTURE_DECISION_GUIDE.md#d07--explain-predictions) in the decision guide.
