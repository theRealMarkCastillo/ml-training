# Reproducing the local demo lab

Reproducibility is itself a lesson: can another environment reconstruct the result, and which parts of the original conditions does that actually test? Start with the [learning guide](LEARNING_GUIDE.md) and one component tutorial before attempting the complete evidence bundle.

## Choose the strength of the check

| Check | What you learn | What it does not establish |
| --- | --- | --- |
| Contract tests | Specified data, parser, gate and artifact invariants hold on the tested fixtures | Model quality |
| Visual notebook execution | Charts and small examples run from the versioned evidence snapshot without model artifacts | New training, new model inference or production performance |
| Smoke training | The small workflow can create and reload an artifact | Full-budget effectiveness |
| Artifact hash verification | Recorded files have not changed | Correctness or trustworthy provenance by itself |
| Saved-model replay | The installed environment reproduces recorded inference and metrics | A new training run will converge to the same model |
| Independent retraining | Training variability under the chosen seeds/settings | Transfer to new populations or hardware |
| Target deployment experiment | Behavior under specified real constraints | Reliability outside the evaluated conditions |

The recorded lab completed tests and saved-model replays; neural independent retraining and production deployment remain follow-up work. Reproduction commands below can perform substantial computation. Read the relevant tutorial's workload and hardware notes first.

## Recorded environment and scope

The observed host is an Apple M2 Max with 96 GB RAM, macOS arm64 and Python 3.12.13. The completed checks use a second installation of each relevant environment from its lockfile. They reload saved artifacts and regenerate outputs. They are not independent retraining replications, clean operating-system installs, cold filesystem-cache tests or cross-hardware benchmarks. All required replay and profile checks passed; [PROGRESS.md](project/PROGRESS.md) and [CONFORMANCE.md](project/CONFORMANCE.md) record the evidence.

## Install separate environments

The [visual notebooks](../notebooks/README.md) have an independent, lightweight environment: `uv sync --project environments/notebooks --frozen`. They do not require the model environments or macOS OpenMP below.

For the full lab, install `uv`, Python 3.12 and macOS OpenMP (`brew install libomp`) if absent. Training and UI dependencies are pinned in the root and four environment lockfiles; notebooks have a fifth environment lockfile. Do not combine their current Transformers requirements into one environment. [`configs/environments.json`](../configs/environments.json) lists each environment, its purpose and which lessons use it.

The bootstrap script installs any subset from their lockfiles, and `ml-demo list` shows which are installed:

```bash
python3 scripts/bootstrap.py                  # all six environments
python3 scripts/bootstrap.py cpu transformers # only what D05-D08 need
python3 scripts/bootstrap.py --repro          # the second .venv-repro installation used by replay checks
```

It runs the same `uv sync` commands as the manual equivalent below:

```bash
uv sync --frozen
uv sync --project environments/transformers --frozen
uv sync --project environments/mlx --frozen
uv sync --project environments/rlvr --frozen
uv sync --project environments/ui --frozen

UV_PROJECT_ENVIRONMENT=.venv-repro uv sync --frozen
UV_PROJECT_ENVIRONMENT=.venv-repro uv sync --project environments/transformers --frozen
UV_PROJECT_ENVIRONMENT=.venv-repro uv sync --project environments/mlx --frozen
UV_PROJECT_ENVIRONMENT=.venv-repro uv sync --project environments/rlvr --frozen
```

The relative `UV_PROJECT_ENVIRONMENT` is interpreted relative to each selected project. Confirm that the separate directories exist before running validators. The UI uses its own environment; model subprocesses use their corresponding task environment. Metal operations require GPU access; a restrictive process sandbox can make a usable host appear unsupported.

Download the exact model snapshots in [SLM_DEMOS.md](SLM_DEMOS.md) and [RETRIEVAL_DEMO.md](RETRIEVAL_DEMO.md), matching `configs/models.json`. `ML_DEMO_MODEL_CACHE` can override the Hugging Face cache root. After downloading:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
.venv/bin/ml-demo doctor
.venv-repro/bin/ml-demo verify all --profile smoke
```

Offline flags prevent normal Hugging Face downloads but are not a network-isolation proof. D04 has a separate OS-enforced test below.

## Training from source versus replaying artifacts

A new checkout contains source, contracts and pinned environment definitions. Generated datasets, teacher cache, model artifacts and virtual environments are excluded from Git. Run the individual walkthroughs in dependency order to create them: D05/D06/D07, D08, D01/D02/D04, then D03 from the completed D02 run. Preserve the actual D01 teacher cache when continuing an interrupted experiment. Each training command prints its new run directory.

After creating all eight seed-41 local runs and the additional CPU runs, explicitly record the component selection. This helper verifies model hashes, task identity, seed, completed status and D02→D03 / D05→D07 source consistency before writing any selectors:

```bash
.venv/bin/python scripts/select_component_runs.py \
  --distillation artifacts/distillation/<run> \
  --schema-dsl artifacts/schema_dsl/<run> \
  --rlvr artifacts/rlvr/<run> \
  --edge artifacts/edge/<run> \
  --tabular artifacts/tabular/<run> \
  --fast-classifier artifacts/fast_classifier/<run> \
  --explainability artifacts/explainability/<run> \
  --retrieval artifacts/retrieval/<run> \
  --tabular-42 artifacts/tabular/<seed-42-run> \
  --tabular-43 artifacts/tabular/<seed-43-run> \
  --fast-classifier-42 artifacts/fast_classifier/<seed-42-run> \
  --fast-classifier-43 artifacts/fast_classifier/<seed-43-run>
```

To bind the newest evaluated seed-41 run of each component instead of naming them, let the CLI choose. It takes D02 from the D03 run's recorded source and D05 from the D07 run's source, includes seed-42/43 CPU runs when present, then applies the same checks:

```bash
.venv/bin/ml-demo prepare hybrid --auto-select --source-run artifacts/rlvr/<run>
```

It prints the selected paths. Inspect them before training the capstone. The four extra CPU flags are optional for a seed-41-only hybrid walkthrough, but required to build a complete final-audit selection from a fresh checkout. Existing selections for other seeds are preserved. The original development machine's local evidence bundle already has these selections under `artifacts/validation/selected_*.json`. Replaying it requires the referenced local artifacts and pinned cached base models. The hybrid's bindings contain absolute paths to this checkout; moving the bundle requires creating a new comparison with selectors for the new location. Do not silently edit an already hashed run's bindings.

## Replay the measured tracks

Use each walkthrough's exact command options. Replay requires exact agreement by default. A mismatch message says whether the saved files changed (checksum failure) or the recomputation differs (usually a different machine or library version) and names the first differing values. On another host, `ml-demo evaluate <demo> --run-dir <run> --tolerance 1e-6` accepts small numeric drift; generated text must still match exactly. These are the validation entry points:

```bash
.venv-repro/bin/python scripts/validate_cpu.py --help
environments/transformers/.venv-repro/bin/python scripts/validate_retrieval.py --help
environments/mlx/.venv-repro/bin/python scripts/validate_slm.py --help
environments/rlvr/.venv-repro/bin/python scripts/validate_rlvr.py --help
```

CPU reproduction verifies all model artifacts, recomputes metrics and explanations, checks ONNX parity and sends real localhost HTTP requests. Retrieval replay checks all seven comparisons, rankings, candidate pools and parameter hashes. SLM replay regenerates every recorded test/challenge output for all variants, including the teacher, classical router, typed parser and quantized device model. RLVR replay regenerates both greedy and four-sample outputs for all four variants, requiring exact raw text and metrics.

```bash
environments/rlvr/.venv-repro/bin/python scripts/validate_rlvr.py \
  artifacts/rlvr/<run> --device cpu --output artifacts/validation/rlvr.json

environments/rlvr/.venv-repro/bin/python scripts/check_rlvr_selection.py

environments/rlvr/.venv-repro/bin/python scripts/profile_rlvr.py \
  artifacts/rlvr/<run> --output artifacts/validation/rlvr-performance.json

sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  environments/mlx/.venv-repro/bin/python scripts/validate_edge_offline.py \
  artifacts/edge/<run> artifacts/validation/edge-offline.json
```

Run timing profiles without competing model work. CPU inference uses four PyTorch threads for RLVR; ONNX uses its own explicit single-thread configuration. Warm measurements and fresh-process startup remain separate. Fresh-process time includes startup/import/loading overhead and uncontrolled OS caches. The RLVR evaluator is much slower than its short training stage because it generates hundreds of candidates for each of four variants.

The final audit also checks the paired native/ONNX batch profile and the small-perturbation explanation study. Run these against the three selected D06 runs and selected D07 run, preserving the original artifacts:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-repro/bin/python scripts/profile_classifier_batches.py \
  artifacts/fast_classifier/<seed-41-run> artifacts/fast_classifier/<seed-42-run> \
  artifacts/fast_classifier/<seed-43-run> --output artifacts/validation/classifier-batches.json

.venv/bin/python scripts/validate_explanation_stability.py \
  artifacts/explainability/<run> --output artifacts/validation/explanation-stability.json
.venv-repro/bin/python scripts/validate_explanation_stability.py \
  artifacts/explainability/<run> --compare artifacts/validation/explanation-stability.json \
  --output artifacts/validation/explanation-stability-reproduced.json
```

The classifier profile uses identical fixed inputs for both backends at batch sizes 1 and 256. The explanation study changes amount by ±1% on the first 100 held-out rows, records missing amounts explicitly and requires an exact second-environment replay. See [CPU_DEMOS.md](CPU_DEMOS.md) for interpretation and measurement boundaries.

## Corrections to recorded runs

Recorded runs are hash-locked and are never edited. Where a later review found a flaw in how a run was measured or compared, a supplemental script re-scores the run's saved outputs, first reproducing the recorded figures exactly, and writes a separate file under `artifacts/validation/`. New runs use the corrected behavior and record its version, so older runs still replay as recorded.

```bash
.venv/bin/python scripts/rescore_typed_slot.py      # D02: parser limited to training wording
.venv/bin/python scripts/rescore_rlvr_hidden.py     # D03: hidden fixtures disjoint from the reward (v2)
OMP_NUM_THREADS=1 .venv/bin/python scripts/profile_onnx_startup.py  # D06: fresh-process startup, minimal scorer
```

The first two are deterministic and run no model; the third is a timing measurement and varies by host. D01 needs no script: its curation log already shows every accepted teacher answer equals the oracle label, and `export_notebook_snapshot.py` records that check. The [results pages](SLM_RESULTS.md) explain what each correction changed.

## Hybrid, interface and final evidence audit

```bash
.venv/bin/python scripts/validate_capstone.py --rlvr-run artifacts/rlvr/<run>
```

That runs the full 320-request comparison, selects its completed artifact and starts clean-worker replay. For an already completed comparison, use `--run artifacts/hybrid/<run>` instead; the replay destination must be new. The raw CLI alternative in [HYBRID_DEMO.md](HYBRID_DEMO.md) accepts an explicit output directory. No training occurs during either replay.

```bash
environments/ui/.venv/bin/python scripts/check_ui.py
environments/ui/.venv/bin/python scripts/check_ui_fresh.py \
  distillation schema_dsl edge retrieval explainability \
  --output artifacts/validation/ui-fresh-components.json
environments/ui/.venv/bin/python scripts/check_ui_fresh.py rlvr hybrid \
  --output artifacts/validation/ui-fresh-final.json
.venv/bin/ml-demo report all
.venv-repro/bin/python scripts/final_audit.py
```

`check_ui.py` checks artifact views. `check_ui_fresh.py` submits real forms that execute local models, rather than substituting outputs. Browser rendering and the fresh D07/D09 flows are checked separately and recorded in `ui-browser.json` with screenshots. `final_audit.py` intentionally fails if selected runs, actual parameter changes, replay evidence, branch coverage, UI checks or required reports are missing. Its pass is an evidence-bundle audit, not a production certification.

The run manifests retain source hashes, data hashes, dependency versions, locks and model revisions. Failed runs remain visible. Source and small fixtures are versioned; the local artifact bundle has not been published or deployed. The narrow synthetic results require new real-data evaluation, additional training seeds and deployment-specific operational checks before any production claim.
