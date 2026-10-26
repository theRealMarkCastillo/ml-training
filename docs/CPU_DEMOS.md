# Running the CPU demos

These are actual local training and evaluation workflows. D05 and D06 train models; D07 explains the saved D05 models. All datasets are synthetic. D08 has a separate [retrieval walkthrough](RETRIEVAL_DEMO.md). See [PROGRESS.md](PROGRESS.md) for the language-model and hybrid tracks.

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

`doctor` writes a sanitized environment report to `artifacts/environment.json`. `verify` without a run directory runs contract tests; it does not imply that all training tracks have been implemented. `uv.lock` pins the resolved CPU environment. Model packages for later tracks will use separate dependency groups/environments.

## D05: learn a tabular risk function

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

## D06: export and time a request classifier

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

## D07: inspect and change an explanation

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

## Evidence and reproducibility

```bash
ml-demo report all
```

This verifies complete run checksums and creates `artifacts/index.json` and `artifacts/report.md`. Failed/incomplete runs remain visibly distinct. Each completed run records dependencies, source file hashes, lockfile hash, dataset checksums, seed, model settings, timing and peak process memory. Generated datasets are frozen by checksum; a changed generator requires an explicit dataset version change rather than silently replacing held-out data.

Use `--profile smoke` for tiny 2,000-row checks. Smoke metrics are never promoted into local evaluation evidence. For a clean-environment reproduction, create a second virtual environment from `uv.lock`, install the package, run the tests, then re-evaluate the saved artifacts. Re-training with the same seed can be compared separately; timings can differ even when predictions reproduce.
