# Visual ML lessons

These nine Jupyter notebooks turn the demos into visual experiments. Read the theory, predict what a chart should show, run the cells, change one assumption, and explain what the result supports. Plotting and analysis code is visible in the cells; `lab.py` only loads evidence and supplies shared styling.

Every notebook works with the small checked-in [recorded results snapshot](data/recorded_results.json). It needs no model download, API key, GPU or original `artifacts/` directory. Saved figures are also included in the notebooks, so GitHub and notebook viewers can display the lessons without execution. Running the notebooks recomputes charts and small examples; it does not repeat model training or claim a new model evaluation.

## Choose a lesson

| Notebook | Visual question | Change or inspect |
| --- | --- | --- |
| [D01 · Distillation](D01_distillation.ipynb) | Does template accuracy hide a generalization gap? | Workload volume in a time-only amortization illustration; challenge escalation |
| [D02 · Structured generation](D02_structured_generation.ipynb) | How do syntax, schema and meaning diverge? | An AST operator in the real bounded interpreter |
| [D03 · Verifiable rewards](D03_verifiable_rewards.ipynb) | Did training improve behavior, or did more inference help? | A reward group and its simplified relative advantages |
| [D04 · Local inference](D04_local_inference.ipynb) | What does compression guarantee, and what does a runtime gate miss? | A proposed temperature in the real simulator |
| [D05 · Tabular prediction](D05_tabular_prediction.ipynb) | Can AP improve while false alerts exceed the budget? | Assumed prevalence and daily volume |
| [D06 · Inference optimization](D06_inference_optimization.ipynb) | How much can a faster kernel improve a full request? | The model's assumed share of serial execution time |
| [D07 · Feature attribution](D07_feature_attribution.ipynb) | What does a contribution explain, and in which units? | Before/after contributions for a recorded feature edit |
| [D08 · Retrieval](D08_retrieval.ipynb) | Why can a better reranker make the pipeline worse? | Any held-out query's relevant-document rank |
| [D09 · Hybrid capstone](D09_hybrid_capstone.ipynb) | Do component gains survive composition? | Outcome-specific ablations and aligned risk populations |

For a first ML course, start with D05 → D06 → D07. For language models, use D01 → D02 → D04 → D03. Read D08 before D09. Unlike training, viewing these lessons has no checkpoint dependency order. The [learning guide](../docs/LEARNING_GUIDE.md) defines prerequisite terms and the [architecture guide](../docs/ARCHITECTURE_DECISION_GUIDE.md) helps compare alternatives.

## Run locally

From the repository root, with Python 3.12 and `uv`:

```bash
uv sync --project environments/notebooks --frozen
uv run --project environments/notebooks --frozen jupyter lab --notebook-dir=.
```

Open `notebooks/`, select a lesson, and use **Restart Kernel and Run All Cells**. Choose the Python kernel from `environments/notebooks/.venv` if your editor offers several environments. In VS Code, select that environment as the notebook kernel. The notebook environment is separate from the CPU, MLX, Transformers and RLVR training environments; it does not install or import model frameworks.

Each lesson finds the repository from either the root or `notebooks/`. Keep the repository layout intact: D02 and D04 import the actual interpreter/simulator from `src/`. Change a parameter in the marked example cell and rerun that cell. Restart and run all before drawing conclusions; hidden state from an earlier cell order can otherwise mislead you.

Validation was performed on macOS. D04 reuses repository utilities that import the Unix `resource` module, so that lesson requires macOS/Linux or a compatible Unix environment; native Windows execution is not supported by that imported utility.

Installation needs package access. After installation, the supplied cells only read local files and execute local calculations. No remote model or hosted notebook service is required. Standard Matplotlib figures and editable Python parameters keep the examples portable without widget extensions.

## Read the evidence correctly

Three kinds of material appear in the lessons:

1. **Recorded measurements:** exported from the October 26, 2026 evaluated runs. The loader prints source run IDs, recording dates and seeds. They retain the synthetic-data, sample-size and host limitations in the original [results](../docs/FINAL_REPORT.md).
2. **Derived visualizations:** recomputed from those recorded numbers. Replotting a metric is not new model inference. An ablation remains descriptive evidence from its original population.
3. **Illustrative calculations and examples:** new arithmetic scenarios, small AST tables and simulator commands, explicitly labeled in each lesson. Their parameters are assumptions; they do not become measured results because a cell runs.

The snapshot exports aggregates, all held-out retrieval ranks, one recorded explanation edit, and the duplicate-feature counterexample. It does not contain model weights or complete prediction datasets. In particular, it cannot recompute arbitrary SHAP edits or inference after a prompt change. Follow the linked CLI tutorials for those operations.

The original recorded host was an Apple M2 Max with 96 GB RAM; this is context for those measurements, not a notebook hardware requirement. Neural experiments have one seed; CPU results have three. Simulator success, exported-model parity and explanation additivity answer different questions from production task quality.

## Provenance and your own runs

The snapshot's `sources` map records repository-relative filenames and SHA-256 values for every input read by the exporter. These hashes identify the recorded evidence; a fresh clone does not contain the original large artifacts. They are provenance, not an independent certification of the experiments. The exporter performs no model training or model selection.

To regenerate the original snapshot when the local evidence bundle is present:

```bash
environments/notebooks/.venv/bin/python scripts/export_notebook_snapshot.py \
  --selection-dir artifacts/validation \
  --output artifacts/my-notebook-results.json
```

For your own completed full-lab runs, provide the same selection-file schema in a separate directory. Inspect the selector contracts in [the exporter](../scripts/export_notebook_snapshot.py). Supply the paired classifier profile and hybrid analysis generated for those selected runs, too. Export to a new file; do not overwrite the historical teaching snapshot merely to make a new result match it. Every selected run must have `evaluated` status, and supplemental profiles must match their selected runs. This exporter requires the complete nine-demo evidence set; it does not merge partial new runs with old results.

Set `SNAPSHOT_PATH = ROOT / "artifacts/my-notebook-results.json"` in a lesson's setup cell and restart/run all. A missing or incompatible custom file raises an error, rather than silently substituting historical data. The prose describes the original experiment and must be reassessed when changing the data. Seed choices, variants and sample-count labels in the teaching cells assume the supplied experiment design; adapt them explicitly for a different design.

## Validate and maintain the lessons

```bash
# Validate structure, links, execution order and freshness of saved outputs.
environments/notebooks/.venv/bin/python scripts/check_notebooks.py

# Execute in fresh kernels from temporary copies without artifacts or model weights.
# Write refreshed outputs to the source notebooks after intentional edits.
environments/notebooks/.venv/bin/python scripts/check_notebooks.py --execute --write

# During editing, execute only lessons whose saved outputs are stale.
environments/notebooks/.venv/bin/python scripts/check_notebooks.py --execute --write --stale-only

# Optional only on a machine that has the original evidence bundle:
environments/notebooks/.venv/bin/python scripts/check_notebooks.py --verify-sources

# Regression checks for provenance and educational metric relationships.
environments/notebooks/.venv/bin/python -m pytest tests/test_notebooks.py -q
```

The checker hashes cell sources, the snapshot, shared helpers, imported interpreter/simulator source and the notebook lockfile. It rejects stale output fingerprints, error cells and incomplete execution. It saves a report and extracted PNGs under `artifacts/validation/notebooks/`. This is a consistency check, not proof of correctness: inspect figures and reasoning as well. Jupyter kernels use local sockets; a restricted execution sandbox may need permission to launch them.

After an edit, verify axis units, denominators, cohort alignment and the distinction between measured and illustrative values. Keep plots of honest negative results. Do not select models or tune thresholds by repeatedly checking the published final test.

Implementation references: [Jupyter notebook format](https://nbformat.readthedocs.io/en/latest/format_description.html) and [NBClient execution](https://nbclient.readthedocs.io/en/latest/client.html).
