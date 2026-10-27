# ML Training Demo Lab

Learn when to train a model, when a simpler method is enough, and how to tell the difference.

This repository is a hands-on tutorial built around eight ML use cases and a hybrid assistant capstone. You will train small models, inspect their failures, reload their saved artifacts, and compare quality with latency and operational complexity. Each lesson connects the underlying theory to runnable code and asks what would have to change before applying the technique in production.

The experiments include useful negative results. The distilled router reached 100% on generated test tickets but only 6/12 on independent challenges. Reinforcement learning changed weights without improving held-out accuracy. A reranker improved a controlled comparison yet hurt the best complete retrieval pipeline. Learning to interpret these outcomes is part of the curriculum.

## Start here

Read the [learning guide](docs/LEARNING_GUIDE.md) for prerequisites, evaluation vocabulary and the experiment workflow. Then choose a route:

- **Choosing an architecture:** use the [architecture decision guide](docs/ARCHITECTURE_DECISION_GUIDE.md) for use/avoid criteria, alternatives beyond the repo, and a reusable decision record.
- **Learning visually:** open the [nine Jupyter lessons](notebooks/README.md) for recorded-result charts, editable examples and architecture exercises. They run without model downloads and include saved figures for browsing.
- **First ML project:** start with D05, D06 and D07. These CPU lessons teach prediction, calibration, deployment measurement and explanations without downloading language models.
- **Language-model customization:** read the shared LoRA background, then work through D01, D02 and D04. Continue to D03 after producing the D02 checkpoint.
- **Search and assistants:** start with D08. After completing the component lessons, use D09 to investigate whether their gains survive integration.
- **Read before running:** each results document is a worked case study. Recorded results remain readable in a fresh clone even though the large local artifacts are not included.

## The lessons

| Lesson | Question you will investigate | Concepts and experiment | Read the observations |
| --- | --- | --- | --- |
| [D01 · Distillation](docs/SLM_DEMOS.md#d01-distillation) | Can a small model imitate a larger model on a bounded task? | Teacher-label curation, supervised fine-tuning, abstention, amortization | [Language-model results](docs/SLM_RESULTS.md) |
| [D02 · Structured output](docs/SLM_DEMOS.md#d02-strict-syntax-and-executable-semantics) | When is valid JSON still the wrong answer? | Typed query ASTs, executable checks, compositional generalization, parser baseline | [Language-model results](docs/SLM_RESULTS.md) |
| [D03 · Verifiable rewards](docs/RLVR_DEMO.md) | Does reinforcement learning add anything after SFT? | GRPO, reward contrast, hidden evaluation, compute controls, best-of-N | [RLVR results](docs/RLVR_RESULTS.md) |
| [D04 · Local inference](docs/SLM_DEMOS.md#d04-local-model-and-simulated-device) | What do quantization and offline execution actually guarantee? | Adapter fusion, 4-bit export, device-command validation, simulator state | [Language-model results](docs/SLM_RESULTS.md) |
| [D05 · Tabular prediction](docs/CPU_DEMOS.md#d05-learn-a-tabular-risk-function) | Can a useful ranking still produce too many false alarms? | Logistic regression, boosted trees, temporal splits, calibration, drift | [CPU results](docs/CPU_RESULTS.md) |
| [D06 · Fast classification](docs/CPU_DEMOS.md#d06-export-and-time-a-request-classifier) | How much of service latency belongs to the model? | ONNX export, numerical parity, batching, warm versus startup timing | [CPU results](docs/CPU_RESULTS.md) |
| [D07 · Explanations](docs/CPU_DEMOS.md#d07-inspect-and-change-an-explanation) | What does a feature contribution explain? | SHAP, log-odds, background data, correlated features, sensitivity | [CPU results](docs/CPU_RESULTS.md) |
| [D08 · Retrieval](docs/RETRIEVAL_DEMO.md) | Should you tune the retriever, add a reranker, or do both? | Bi-encoders, cross-encoders, hard negatives, candidate recall, ablations | [Retrieval results](docs/RETRIEVAL_RESULTS.md) |
| [D09 · Hybrid capstone](docs/HYBRID_DEMO.md) | Does a better component make a better system? | Routing, fallback, external gates, end-to-end evaluation, changing knowledge | [Hybrid results](docs/HYBRID_RESULTS.md) |

Every tutorial includes concepts, an implementation walkthrough, observations to inspect, production tradeoffs, and exercises with interpretation guidance. The [architecture decision guide](docs/ARCHITECTURE_DECISION_GUIDE.md) helps you choose an approach; the [production tradeoff guide](docs/PRODUCTION_TRADEOFFS.md) develops its operating consequences.

## Run your first experiment

Run commands from the repository root with Python 3.12 and `uv` installed. On macOS, install OpenMP for XGBoost with `brew install libomp`. Start with the CPU environment:

~~~bash
uv sync --frozen
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ml-demo doctor
ml-demo verify all --profile smoke
ml-demo train tabular --profile smoke --seed 41
ml-demo report all
~~~

The verification command runs the contract suite; it does not train every model. The small smoke training run checks the workflow. Its metrics are not the full experiment. Open the new run's `report.md` and compare the dummy, logistic and tree models. Then follow the [CPU tutorial](docs/CPU_DEMOS.md) to run the 50,000-event experiment and explain a prediction.

Training prints a unique directory. Replace placeholders such as `artifacts/tabular/<run>` with that printed path:

~~~bash
ml-demo evaluate tabular --run-dir artifacts/tabular/<run>
ml-demo verify tabular --run-dir artifacts/tabular/<run>
~~~

`evaluate` recomputes saved-model metrics; `verify --run-dir` checks file hashes. They answer different questions. On a different machine, exact recomputation can drift in the last digits; the error says so, and `--tolerance 1e-6` accepts that drift.

The other lessons run in their own locked environments because their dependencies conflict. `ml-demo list` shows which are installed; `python3 scripts/bootstrap.py <name>` installs one. The [reproduction guide](docs/REPRODUCTION.md) explains how to create the separate environments, download pinned models and replay the full lab.

## Explore interactively

The [local comparison UI](docs/COMPARISON_UI.md) turns the saved artifacts into an experiment viewer. Compare baselines, inspect individual failures, try a fresh language-model request, or edit a tabular feature and recompute its explanation.

The UI needs locally generated runs. A fresh clone contains source, tests, small challenge fixtures and dependency locks; it does not include model checkpoints, generated datasets or virtual environments. Links into `artifacts/` in results documents refer to the original local evidence bundle and do not resolve in a fresh clone. Reproduction creates new run directory names; use the paths printed by your commands to inspect your own evidence.

## What the lab establishes

The recorded implementation trained models for seven use cases, recomputed explanations from the saved D05 model for D07, and completed artifact reload and inference replay across all eight tracks. It also ran 320 executions of the ten capstone variants. The post-review validation passed 94 contract tests and seven fresh UI submissions. These records describe the implementation as evaluated on October 26, 2026; they are not a promise that a new machine will reproduce timing or accuracy exactly.

Most data is synthetic. Neural experiments use one training seed; the cheap CPU experiments use three. The observed host was an Apple M2 Max with 96 GB RAM. The MLX lessons require Apple Silicon/Metal in the supplied implementation. CPU lessons provide the smallest starting point; no claim is made that 96 GB is a minimum requirement for every lesson. Physical devices and customer systems are not connected.

Use the [final report](docs/project/FINAL_REPORT.md) as a synthesis of what was learned. The [implementation plan](docs/project/IMPLEMENTATION_PLAN.md), [progress record](docs/project/PROGRESS.md), [conformance map](docs/project/CONFORMANCE.md) and [code-review case study](docs/project/CODE_REVIEW.md) explain how the evidence was built and checked. These project records live in [docs/project](docs/project/README.md), apart from the lessons.

The scope originated in the supplied six-page PDF **Fine-Tuning and ML Use Cases - Google Gemini**. Its examples supplied questions to test. Its embedded prompts were treated as source material, and its illustrative performance claims were not adopted as measured facts.
