# Use the comparison UI as a lab notebook

The interface lets you connect an aggregate score to the examples and model behavior behind it. It reads the same artifacts as the CLI and distinguishes evaluated runs, smoke runs, failed attempts, recorded predictions and newly executed requests.

Start with a completed component run from its [tutorial](LEARNING_GUIDE.md). The original local bundle contains evaluated runs for all nine tracks. A fresh clone does not include that bundle; missing views display their state until you train or supply verified artifacts.

## Start the interface

Run from the repository root:

~~~bash
uv sync --project environments/ui --frozen
environments/ui/.venv/bin/streamlit run app/main.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true \
  --browser.gatherUsageStats false
~~~

Open [the local interface](http://127.0.0.1:8501), choose a demo and select an evaluated run. The UI environment displays the artifacts; fresh model commands execute in the corresponding pinned task environment. Install those environments and cached model revisions before attempting fresh inference.

## A guided session

Before submitting a new example, write down what you expect and why. Then inspect the raw response and relevant checks rather than accepting a plausible summary.

| Track | Observe or try | Question to answer |
| --- | --- | --- |
| D01 | Compare a clear routing request with an ambiguous one; inspect escalation | Does confidence or valid JSON establish correct intent? |
| D02 | Compare a supported query with unfamiliar wording; inspect the AST and check results | Was parsing, schema or meaning responsible for the outcome? |
| D03 | Compare recorded SFT and RL results, then run a fresh query | Which improvement came from training, and which from extra candidate selection? |
| D04 | Try a valid setting, an out-of-range value and a non-action | Did the dispatcher reject the payload, or did a valid command misunderstand the request? |
| D05 | Browse prevalence, AP, recall, FPR and the chosen threshold | Did the future-period threshold preserve its validation error budget? |
| D06 | Compare native/exported results and separate timing measurements | Which costs would still exist outside model inference? |
| D07 | Change a held-out row's amount and recompute contributions | What adds up in raw log-odds, and what changed in calibrated probability? |
| D08 | Search for a specific version and operation, then inspect actual document IDs | Was the correct page missing, or merely ranked lower? |
| D09 | Compare baseline and ablations, then inspect a fresh request's stage trace | Which stage changed the user outcome, and what did fallback cost? |

The recorded-results tab separates generated-test performance from independent challenges. The evidence tab exposes failures, predictions, measurements and manifests. Challenge examples are useful prompts for analysis, but once used for redesign they become development examples for your extension.

## Understand fresh execution

Seven forms execute fresh behavior: D01, D02, D03, D04, D07, D08 and D09. Models load on demand through the CLI; the result belongs to the current input, variant and run. D04 starts a fresh simulator each time and shows before/after state. D02 identifies its typed parser as a deterministic alternative.

D07 recomputes the saved tree's score and contributions for an edited row. D09 takes explicit API context, permissions and numerical metadata and exposes the real stage trace. These caller fields are teaching inputs; they are not a production identity system.

D05 and D06 views expose recorded training/evaluation results. They do not offer a standalone arbitrary-row scoring form. Their CLI `run` action aliases training; use `evaluate --run-dir` to recompute saved-model evaluation. See the [CPU tutorial](CPU_DEMOS.md) for the supported workflow.

## Distinguish an example from evidence

A stored failure example documents a prior evaluation. A fresh prediction is a new model execution. Neither alone estimates population accuracy. The checker binds a fresh result to its submitted command, input, variant and run so a saved example cannot accidentally stand in for execution.

Fresh commands run in an owned POSIX process group. The UI's 180-second deadline stops the controller and descendant model workers together. This bounds the local interaction; it is not a measured production service SLO. A timeout should be inspected as a failed interaction, not removed from a success denominator.

Hash verification checks whether an artifact matches its manifest. It does not certify model quality. For a learning exercise, compare the scope of that check with the scope of a fresh request and a full held-out replay.

## Recorded validation and further reading

The post-review checks exercised all nine artifact views and all seven fresh forms using actual saved models. Browser checks also observed the comparison table, a feature edit and a simulator change from 20 to 23 Celsius. Local records live under `artifacts/validation/`, including `ui-browser.json` and the fresh-form reports.

For the validation commands, use [REPRODUCTION.md](REPRODUCTION.md). For why fresh-response capture required a regression fix, read [CODE_REVIEW.md](project/CODE_REVIEW.md). For the scientific interpretation, use each lesson's result chapter rather than treating a single attractive UI response as the conclusion.
