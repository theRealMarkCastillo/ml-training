# How to learn from this lab

The recurring question is: **what evidence would justify adding this model to a system?** A training command that completes is one piece of evidence. A model that beats a baseline on a useful task, stays within an operating budget and fails acceptably provides a stronger case.

You should be comfortable running Python commands and reading dictionaries, functions and JSON. You do not need to derive backpropagation before starting. The CPU lessons introduce evaluation; the language-model and retrieval lessons introduce their own training objectives. For environment details, use [REPRODUCTION.md](REPRODUCTION.md).

For a design decision before running code, read the [architecture decision guide](ARCHITECTURE_DECISION_GUIDE.md). It compares when to use or avoid each approach, introduces alternatives outside the lab, and provides an architecture decision record template.

## Choose a learning path

| Path | Order | What you should be able to explain afterward |
| --- | --- | --- |
| Prediction and serving | [D05 → D06 → D07](CPU_DEMOS.md) | Why ranking, calibrated probability, a decision threshold, service latency and an explanation are different objects |
| Language-model behavior | [D01 → D02 → D04](SLM_DEMOS.md), then [D03](RLVR_DEMO.md) | What SFT changes, how to verify a structured answer, what quantization changes, and when reward feedback has a useful signal |
| Search | [D08](RETRIEVAL_DEMO.md) | Why candidate recall limits reranking, and how to isolate two interacting model improvements |
| Integration | All components, then [D09](HYBRID_DEMO.md) | Why component scores do not add up to system success and how fallback changes cost and error rates |

D07 consumes a saved D05 run. D03 starts from a saved D02 adapter. D09 consumes the component artifacts, so it is the final exercise. You can read any results chapter without installing its model environment.

## One experiment, six steps

~~~mermaid
flowchart TD
    Q["Define a task and baseline"] --> D["Freeze data and splits"]
    D --> T["Train or tune"]
    T --> V["Select using validation"]
    V --> E["Evaluate frozen test and challenges"]
    E --> R["Reload, reproduce, interpret"]
~~~

1. **State a hypothesis.** “A tuned bi-encoder will distinguish API versions better than the base encoder” is testable. “Fine-tuning makes retrieval better” leaves the task and comparison undefined.
2. **Predict the failure modes.** Write examples involving negation, ambiguity, unseen combinations or changed metadata before tuning.
3. **Run a baseline.** A constant score, logistic model, parser or BM25 search establishes what the training effort must beat.
4. **Choose using validation.** A threshold, checkpoint, prompt or pipeline selected after inspecting test errors has used the test set as development data.
5. **Inspect cases, not only averages.** A formatting gain can coexist with an intent failure. A fast rejection can lower latency while making the product worse.
6. **Write a bounded conclusion.** State the dataset, denominator, comparison, hardware, observed effect and the next uncertainty.

For an extension exercise, keep a separate experiment record. Once you use an original challenge failure to redesign the model, that challenge becomes development material. Create a new untouched evaluation set for the next generalization claim.

## What gets learned, and what does not?

| Mechanism | What changes | Example here |
| --- | --- | --- |
| Supervised training | Parameters are updated to predict labeled targets | D05 logistic/tree models; D01 teacher-target SFT |
| Fine-tuning | A pretrained model adapts to a task; it is a form of training | D02/D04 LoRA, D08 encoder training |
| RL with verifiable rewards | Parameters are updated from scored sampled responses | D03 GRPO |
| Retrieval | Documents and their index supply context at inference | D09's response-field update |
| Deterministic validation | Rules accept, reject or execute a bounded payload | D02 AST interpreter, D04 dispatcher |
| Export or quantization | Model representation/execution changes | D06 ONNX, D04 4-bit weights |
| Explanation | Contributions are computed for an existing prediction | D07 TreeSHAP |

D01 uses teacher-generated final answers as SFT labels. It does not match teacher logits. D02 uses supervised canonical answers. D03 samples and scores its own responses. D07 does not train another predictor. These distinctions help you locate the actual learning step in the code.

## Evaluation vocabulary you will use

**Train, calibration, validation and test.** Training fits parameters and preprocessing. D05 uses a separate chronological calibration slice to fit a probability mapping, then a validation slice to choose the model and decision threshold. Final test rows represent a later period. Other lessons have different split structures; do not assume the D05 four-way split is universal.

**Generalization.** A held-out template asks whether the model transfers within the generator's design. Independently authored challenge wording asks a different question. Neither automatically represents real customer traffic. Replacing account IDs creates more rows without necessarily creating more linguistic variety.

**Leakage.** Future information, test labels, test-derived hard negatives or hidden evaluation checks can make an apparent gain unavailable at deployment. Look for where the data enters fitting, calibration and selection, rather than checking only whether train and test files have different names.

**Calibration.** Among many cases assigned probability 0.2, roughly one fifth should be positive if those probabilities are calibrated on that population. Ranking can be useful even when probability estimates are wrong. Calibration can also deteriorate under a population shift. D05 fits a sigmoid mapping; it does not implement the neural temperature-scaling experiments in the [calibration paper](https://arxiv.org/abs/1706.04599). See [scikit-learn's calibration guide](https://scikit-learn.org/stable/modules/calibration.html) for the distinction.

**Ablation.** Replace one component while keeping the rest of the comparison fixed. It tests the component's contribution within that specific system. It does not reveal every interaction or establish that the component is unnecessary everywhere.

## Read the right metric

| Metric | What it asks | What it leaves unanswered |
| --- | --- | --- |
| Accuracy / joint accuracy | Was the required output correct? Joint accuracy here requires multiple fields to be correct together | Which minority class or failure mode suffered? |
| Macro-F1 | How well does each class fare when class scores receive equal weight? | Whether a particular error is affordable |
| Average precision (AP) | Are positive cases ranked near the top across thresholds? | Whether a deployed threshold meets its false-positive budget |
| Recall and false-positive rate | At the chosen threshold, what fraction of positives are found and negatives flagged? | Precision without also knowing prevalence |
| Brier score | How small is the mean squared error between probabilities and binary outcomes? | Discrimination by itself; a low prior can score well on an almost-all-negative cohort |
| Schema validity | Does the output conform to the allowed structure? | Whether it means what the user requested |
| Execution correctness | Did the program match expected behavior on the evaluation fixtures? | Correctness on all possible tables |
| Recall@k / MRR@k | Was the relevant document retrieved, and how early did it appear? | Whether a downstream answer used it correctly |
| Latency percentiles | How long did the specified measured operation take? | A service SLO under different load or hardware |

The code names one field `pr_auc_average_precision`. Its implementation is scikit-learn's average precision, a recall-weighted sum of precisions, rather than trapezoidal integration of a PR curve. The two can differ. [Definition and implementation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

A quick threshold example: with 1,000 events, 20 actual positives, 50% recall and 5% false-positive rate, the model finds 10 positives and flags about 49 negatives. Precision is about 10/59, or 17%. Those are illustrative arithmetic values, not lab measurements. A “5% false-positive rate” does not mean only 5% of alerts are false.

## Treat performance as an experiment too

Separate training time, warm model inference, request processing, startup and concurrent service behavior. Batch throughput divides rows by elapsed batch time; it does not count independently served HTTP requests. Record input/output lengths, batch size, concurrency, threads, warmups and sample count before comparing two numbers.

The lab's small latency samples are useful for finding large local differences. They do not estimate precise production p99 latency. Its fresh processes do not flush operating-system file caches. Shared worker high-water RSS is different from an isolated model's memory requirement.

## Read a run directory

| File | Learning question |
| --- | --- |
| `run.json` | Which data, code, dependencies and artifacts belong to this run? |
| `training.json` or task-specific training records | Did parameters change, and which checkpoint was selected? |
| `selection.json` | What was chosen before final test evaluation? |
| `metrics.json` | How did each comparison perform on the stated denominators? |
| Predictions, challenges and failure files | Which exact cases explain the aggregate scores? |
| `benchmarks.json` or resource records | What operation, device and budget did timing cover? |

Names differ across tracks; each tutorial lists its files. Hash verification detects changed bytes. It does not establish that the dataset is representative or the code is correct. Saved-model replay tests reconstruction of an inference result. Independent retraining tests a stronger, different form of reproducibility.

The CLI has one important teaching trap: `run` is task-specific. For D05/D06, `run` aliases training; use `evaluate --run-dir` to inspect saved-model metrics. For language and retrieval tasks, `run --run-dir ... --query ...` performs fresh inference. D07 creates an explanation from `--source-run`; D09 either builds a comparison or handles a supplied public request. Consult the lesson rather than assuming every verb means the same thing for every track.

## A learning journal you can reuse

After each lesson, write five sentences:

1. The hypothesis and simplest competing implementation were …
2. The decision used this validation data and criterion …
3. On the frozen evaluation, the measured difference was …
4. The most informative failure was … because …
5. Before production, the next experiment should test …

An acceptable conclusion is that the extra model is not justified. D03 and the D08 reranker make that outcome concrete. The [production tradeoff guide](PRODUCTION_TRADEOFFS.md) turns these journal entries into engineering decisions.
