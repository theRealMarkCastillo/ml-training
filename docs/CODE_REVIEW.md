# Final code review

This review is also a case study in evaluating the evaluator. The [tutorials](LEARNING_GUIDE.md) depend on trustworthy evidence, and the highest-priority finding below showed how a UI test could accidentally report a saved example as a fresh response. Read the fixes as general lessons about observable behavior, process ownership and reproducible setup.

Reviewed on 2026-10-26 against initial commit `d6d9ba4` on `main`. The scope was the uncommitted demo lab: source, tests, scripts, configuration and documentation. Generated artifacts, binaries and lockfile contents were excluded from the code audit; installed frozen environments were used for validation. Entire CLI was available but had no checkpoint/session record, so intent came from the implementation plan and this task's recorded requirements.

The review found four actionable issues. All are fixed, with no unresolved finding from this review. The review focused on data/evaluation boundaries, runtime guards, process lifecycle, reproducibility and the correctness of validation evidence.

## Findings and fixes

### `scripts/check_ui_fresh.py` and `app/main.py`

**High: saved evaluation examples could be mistaken for fresh UI output.** The checker searched every JSON widget, including the evidence tab, and retained the last matching object. Four existing check records contained held-out failure examples rather than their submitted request's response. This could also let a no-op form submission pass.

The UI now captures the newly completed CLI command/result separately. Each check clears prior capture, submits the form, and verifies command, run, input and variant through `ui_evidence.py`. The final audit enforces that contract. All seven forms were rerun with real saved models, and the corrected records contain the actual submitted inputs. Prior records are retained under `artifacts/validation/pre-review/`. The hybrid caption also uses the actual scenario count for smoke runs.

### `app/main.py`, `src/ml_training/processes.py` and `src/ml_training/hybrid.py`

**High: a timeout could leave model workers running.** The UI previously timed out only its direct CLI process. A hybrid CLI can have language, retrieval and RL child workers. In addition, a graceful worker shutdown timeout could propagate after terminating the first worker and prevent cleanup of the remaining workers.

Fresh UI commands now run in a separate POSIX process group, which is killed together on timeout/interruption. Worker shutdown escalates from graceful close to terminate and kill, closes handles, and attempts every worker even if one cleanup fails. A real parent/child subprocess regression proves that timeout returns promptly when a descendant holds the output pipes open; focused tests cover escalation and cleanup continuation.

### `scripts/select_component_runs.py`

**Medium: a fresh checkout could not build all final-audit selectors through the documented helper.** The helper wrote seed-41 component selections, while the resource audit required classifier seeds 42 and 43. Existing manually recorded selections concealed the gap.

Explicit optional flags now select tabular/classifier seeds 42 and 43. Every supplied run is validated before selectors are written; wrong seeds cannot partially replace existing selections. The reproduction guide includes all four flags for a complete audit, while retaining seed-41-only use for the hybrid walkthrough. A clean-directory CLI test verifies all seven CPU selectors.

### `src/ml_training/schema_data.py`

**Medium: the typed parser ignored unsupported aggregate prefixes.** Substring matching accepted requests such as “Do not count tickets where status equals open” as a positive count query. It also discarded arbitrary text after a request-reference marker.

The parser now matches the complete supported aggregate phrase and accepts only the bounded generated reference suffix. Negated/extra-action prefixes and unsupported suffixes are rejected. All 1,360 frozen schema inputs/challenges and hybrid scenario/calibration inputs preserve their previous AST or unsupported result; the experiment metrics therefore do not need to be relabeled. This remains a limited command grammar, not a general natural-language semantic verifier.

## Validation after fixes

- **94 contract tests passed**, including 16 new regressions; the focused review subset passed 20 cases.
- **All seven fresh UI forms passed** using actual local models and request-bound capture.
- **All nine UI artifact views passed**; the updated hybrid view also rendered in the browser with 9/9 evaluated tracks and the recorded 19/32 versus 27/32 comparison.
- **All eight final-audit groups passed**, including selected artifact hashes, actual training records, existing clean replay evidence, UI capture, resource profiles and explanation sensitivity.
- **1,360 frozen parser cases were unchanged**, recorded in `artifacts/validation/review-parser-parity.json`.

Training weights, decoding settings, datasets and measurement code were not changed. This review reran affected behavior, fresh inference and the evidence audit; it did not repeat training or the lengthy full sampled RL replay. The earlier full replay records remain explicit historical evidence, and all recorded model artifacts still verify. Original experiment files were preserved.

The synthetic-data, single-seed neural, simulator and production limitations in [FINAL_REPORT.md](FINAL_REPORT.md) still apply. Negative experiment findings are retained rather than treated as implementation defects. Source and small fixtures are included in Git; generated data, checkpoints, virtual environments and local validation artifacts are excluded.

## Lessons to carry into your own demo

A checker needs a failure that it can detect. If the UI did nothing after submission, a correct fresh-inference test would fail; matching some old JSON widget was not enough. Binding the output to the new command and input makes that distinction observable.

A timeout needs an ownership boundary. Ending the controller is insufficient when model workers continue holding resources or pipes. A reproduction helper needs a fresh-directory test, because an existing local selection file can conceal missing setup work. A parser needs rejection cases as well as accepted grammar examples, because substring matches can silently ignore a user's negation.

**Exercise:** for each fix, describe a plausible incorrect implementation that should fail its regression test. If your proposed test would still pass that implementation, tighten the observable contract. See [tests/test_ui_evidence.py](../tests/test_ui_evidence.py), [tests/test_processes.py](../tests/test_processes.py), [tests/test_run_selection.py](../tests/test_run_selection.py) and [tests/test_schema_data.py](../tests/test_schema_data.py).

## Visual notebook review

Reviewed on 2026-10-26 before publishing the nine [visual lessons](../notebooks/README.md). Scope: notebook code and teaching text, snapshot loading/export, execution checks, tests and setup documentation. Entire had no checkpoint/session context for these changes; the review used the working tree and the tutorial requirements in the conversation. No critical or high issues were found. Two medium issues were reproduced and corrected:

| Location | Failure and consequence | Correction and regression evidence |
| --- | --- | --- |
| [Snapshot exporter](../scripts/export_notebook_snapshot.py) | Selectors and profiling commands can name the same run using absolute or relative paths. String comparisons rejected equivalent paths, preventing valid local results from being exported. | Canonical repository-relative paths now identify runs in both classifier profiles and hybrid analysis. Tests cover all four absolute/relative combinations and retain rejection of different runs and changed manifest hashes. |
| [D01 notebook](../notebooks/D01_distillation.ipynb) | The training pipeline can validly record no time-only break-even as `null`. Dividing by that value crashed the teaching calculation; zero also caused division by zero. | The cell explains a missing/nonpositive/nonfinite break-even and skips the savings forecast. Regressions execute the actual notebook cell with missing, zero and negative values. The supplied positive example retains its original plot. |

Post-fix validation passed 94 existing contract tests and 25 notebook tests in their respective locked environments. All nine saved notebooks passed structure, link, execution-order and freshness checks, with 26 figures. D01 was rerun in a fresh kernel after its correction; the other eight retained their previously executed, fingerprint-matching outputs. All 49 source-file hashes matched, and a new export was byte-identical to the checked-in snapshot. These checks do not repeat model training or model inference.

The reusable lessons are to compare resource identity rather than path spelling, and to treat a valid negative result as data the presentation must handle. Neither fix changes model selection, measured outcomes or the historical evidence. After the corrections, no blocking findings remained within this review scope.
