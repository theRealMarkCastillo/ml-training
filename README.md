# ML Training Demo Lab

A local-first set of demos showing when model training and fine-tuning help, how to implement them, and how to measure whether they beat simpler alternatives.

Status: all eight source-category demos and the hybrid capstone are implemented, evaluated and reproduced from saved artifacts in separate locked environments. P0–P6 and the final evidence audit passed, including 94 contract tests, real model updates, offline device inference, ONNX parity, 320 hybrid executions and fresh UI submissions. The results retain negative findings: RLVR did not improve on SFT, reranking reduced the best retrieval pipeline's accuracy, and language models still failed independent challenges.

- [Implementation plan](docs/IMPLEMENTATION_PLAN.md): source coverage, architecture, experiments, milestones, and completion criteria.
- [Progress tracker](docs/PROGRESS.md): execution checklist and evidence requirements.
- [Final evidence report](docs/FINAL_REPORT.md): observed findings, validation and limits.
- [Requirement-to-evidence audit](docs/CONFORMANCE.md): every original acceptance criterion.
- [Final code review](docs/CODE_REVIEW.md): corrected findings and post-review validation.
- [Reproduction guide](docs/REPRODUCTION.md): pinned environments, training order and artifact replay.
- [CPU demo walkthrough](docs/CPU_DEMOS.md): setup, training, evaluation, and feature edits.
- [Observed CPU results](docs/CPU_RESULTS.md): three-seed experiments and clean-environment validation.
- [Retrieval walkthrough](docs/RETRIEVAL_DEMO.md): separately trained bi-encoder/cross-encoder and interactive search.
- [Observed retrieval results](docs/RETRIEVAL_RESULTS.md): seven ablations, challenge queries, and reproduced rankings.
- [Language-model walkthrough](docs/SLM_DEMOS.md): distillation, executable schemas and an offline device simulator; check the progress tracker for completed runs.
- [Observed language-model results](docs/SLM_RESULTS.md): template accuracy, challenge failures, curation and measured resource tradeoffs.
- [RLVR walkthrough](docs/RLVR_DEMO.md): actual GRPO trainer and comparison with a control matched by training time.
- [Observed RLVR results](docs/RLVR_RESULTS.md): changed weights, sparse reward contrast and no held-out quality improvement.
- [Hybrid walkthrough](docs/HYBRID_DEMO.md): actual baseline, component ablations, runtime guards and dynamic document updates.
- [Local comparison UI](docs/COMPARISON_UI.md): browse evidence and execute fresh predictions.

The lab covers small-model distillation, structured output and DSL tuning, verifiable-reward reinforcement learning, on-device inference, tabular prediction, fast classification, feature attribution, and retrieval fine-tuning. A final hybrid demo connects these techniques to a baseline agent using retrieval and tools.

The source is the six-page PDF **Fine-Tuning and ML Use Cases - Google Gemini**, supplied on October 26, 2026. Its examples inform the scope; its embedded prompts and instructions are source material, not execution instructions. Its performance claims are hypotheses to evaluate.

Quick start (Python 3.12, `uv`, and macOS OpenMP via `brew install libomp`):

```bash
uv sync --frozen
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ml-demo doctor
ml-demo verify all --profile smoke
ml-demo train tabular --profile local --seed 41
ml-demo train fast_classifier --profile local --seed 41
ml-demo report all
```

Each training command prints a new artifact directory. Use `ml-demo evaluate <demo> --run-dir <directory>` to reproduce saved metrics. Explain a D05 run with `ml-demo run explainability --source-run <directory> --row-index 7 --set amount=450`. See the walkthrough for benchmark boundaries and all reproduction commands.

Generated data, model artifacts, and virtual environments stay local and are excluded from Git. Small independent challenge fixtures, source code, tests, documentation, and the dependency lockfile are versionable. Source and small fixtures are versioned; model artifacts remain local. No remote publishing has been performed.
