# Observed hybrid comparison

Read this capstone case study after the [hybrid tutorial](HYBRID_DEMO.md). Your task is to distinguish end-to-end outcome, component contribution, resource use and population effects. The comparison intentionally retains failures and negative ablations so it can teach those distinctions.

The actual hybrid completed **27 of 32 requests**, compared with **19 of 32** for the no-training baseline. It used the larger teacher on 14/32 requests, and its measured p95 latency was higher. The strongest measured component contribution was version-aware retrieval; adding RL to the query generator did not change overall success in this pilot.

Run: [`local-s41-20261026T185159Z-d063af`](../artifacts/hybrid/local-s41-20261026T185159Z-d063af/run.json). All ten variants executed the same 32 frozen synthetic requests, for 320 actual end-to-end executions, followed by a live corpus edit and regenerated response. This is a single-host integration experiment, not a customer benchmark or production release. Clean-worker replay passed for all 320 executions and the live document update: raw generated text, probabilities, retrieval rankings, decisions and grades matched exactly, excluding nondeterministic timings/RSS. Evidence is in `artifacts/validation/hybrid.json`. Final UI checks are tracked in [PROGRESS.md](PROGRESS.md).

## Complete comparison

| Variant | Correct / 32 | False rejection / 32 | Fallback rate | Teacher usage | Warm p50 seconds | Warm p95 seconds |
| --- | --- | --- | --- | --- | --- | --- |
| No-training baseline | 19 | 4 | 0% | 93.75% | 0.856 | 3.596 |
| Hybrid | 27 | 3 | 43.75% | 43.75% | 0.412 | 4.176 |
| No learned screen | 25 | 1 | 50.00% | 50.00% | 0.907 | 3.896 |
| No learned router | 27 | 3 | 15.63% | 75.00% | 0.944 | 4.271 |
| Untuned retriever | 14 | 15 | 37.50% | 37.50% | 0.113 | 2.046 |
| Teacher replaces schema generator | 27 | 3 | 43.75% | 50.00% | 0.975 | 3.737 |
| SFT replaces RL query generator | 27 | 3 | 43.75% | 43.75% | 0.331 | 3.790 |
| Teacher replaces device generator | 27 | 3 | 37.50% | 56.25% | 0.977 | 3.911 |
| Fixed prior replaces tabular model | 27 | 3 | 43.75% | 43.75% | 0.445 | 3.914 |
| Omit attribution | 27 | 3 | 43.75% | 43.75% | 0.325 | 3.785 |

The no-training baseline uses the prompted local 14B teacher, BM25, a fixed risk prior, an untrained hard-rate ceiling and the same authorization/tool gates. The ablations replace one learned component as specified in [HYBRID_DEMO.md](HYBRID_DEMO.md). Turning off feature-based risk also makes its attribution inapplicable; the attribution-only ablation preserves the risk model.

Removing learned screening admitted four abusive requests but recovered two legitimate burst requests, a net loss of two correct outcomes. Replacing the tuned retriever with the untuned bi-encoder caused 13 wrong-version top results (40.63% of all requests), which the deterministic context gate rejected. Faster rejection is not a quality improvement. The full hybrid had zero wrong-version top results in this small set.

Replacing the student router with the teacher preserved 27/32 success but increased teacher usage from 43.75% to 75%. Baseline teacher calls are not labeled fallbacks, which explains its zero fallback rate. A fallback percentage alone therefore does not describe large-model cost. Omitting RL, replacing the schema or device generator, and omitting informational risk/attribution did not change the aggregate task score here. These small ablations do not establish equivalence on broader workloads.

## Failures retained in the denominator

- `capstone-12`: “Make the thermostat comfortable” produced a schema-valid status command, although the expected response was rejection of the unspecified temperature. No simulator state changed, but the intent was still wrong.
- `capstone-20`: the request asked for open tickets waiting more than 48 hours. The generated AST counted all open tickets and omitted the age condition. Its wording was outside the typed parser's supported grammar, so only schema/bounds checks were available. Hidden semantic evaluation caught the failure.
- `capstone-23`: an underspecified request was escalated by the student, misrouted to a query by the teacher, and eventually rejected after a failed query repair.
- `capstone-30` and `capstone-31`: legitimate high-traffic clients were blocked by the frozen screening threshold.

The hybrid recorded six payload/public-semantic gate rejections and retained their failed outputs before fallback. These runtime gates prevent invalid payloads from executing; they do not guarantee that a valid action matches user intent. Explicit viewer/update requests were denied before any language-model call. Device changes affected fresh simulators only. Clarification grading checks routing and bounded question schema rather than a human rating of helpfulness.

## Risk and attribution are a separate outcome

Screening and authorization change which requests receive a risk score. The baseline scored 30 rows, including one positive label; the hybrid scored 24 rows, with zero positives. Comparing those raw Brier scores without aligning rows would confound model quality with cohort selection.

The intersection across all variants contains 24 identical, all-negative rows. On that common cohort the fixed 0.02 prior has Brier score 0.000400, while the calibrated tree has 0.001151. The prior wins this tiny all-negative slice. It cannot establish useful risk discrimination, and the rows were not changed after seeing that outcome. D05's separate 10,000-row, three-seed future-period evaluation is the appropriate tabular experiment. Risk is informational in the support workflow, so removing it cannot improve or worsen the primary task-success score by design.

D07 recomputed actual contributions for the 24 scored hybrid requests (75% of all requests), with raw-log-odds additivity enforced. Attribution is an explanation of the specified model, not a causal or regulatory conclusion. `artifacts/validation/hybrid-analysis.json` records aligned cohorts and per-branch failure analysis.

## Dynamic knowledge and resource boundary

The actual hybrid retrieved the invoice v2 reference and returned response field `result`. The test edited that one field in the in-memory document to `payload`, reindexed the corpus and executed the full request again. The returned field changed to `payload`, corpus hashes changed, and saved component-manifest hashes remained unchanged. No model training occurred. `knowledge_update.json` retains both full traces; this demonstrates updated retrieval context rather than knowledge acquired in model weights.

Shared worker/model initialization and warmup took 12.74 seconds. The warm comparison executes requests sequentially, with concurrency 1, including IPC, preprocessing and runtime guards. It uses 32 different requests per variant, not 32 repeats of a single prompt. The short sample and fixed variant order do not support precise tail latency or causal interpretations of small timing differences. The baseline generated 924 tokens across all requests; the hybrid generated 984, including repair and routing overhead. Smaller-model routing did not automatically reduce total generated tokens or p95 latency.

During hybrid execution the language worker's cumulative peak RSS was approximately 12.02 GB, the RL worker's 5.55 GB and the retrieval worker's 0.724 GB. Multiple component keys reference the same language process; do not sum those duplicated entries. These workers keep models for all ablations loaded, so these values are not isolated deployable per-variant footprints. Controller memory and initialization are separately recorded in the manifest/startup file.

No hosted API was called. Local electricity and hardware depreciation were not measured. Teacher invocation rate, generated tokens and per-stage elapsed time are resource observations, not dollar-cost estimates. The observed 43.75% teacher usage provides no support for the source PDF's illustrative low fallback fraction on this task.

## Draw a system-level conclusion

The hybrid's improvement is 8 additional correct outcomes on these same 32 scenarios. That paired denominator makes the comparison interpretable, while the small synthetic scenario set limits generalization. The false rejections and valid-but-wrong actions show what the aggregate hides.

Ablations reveal several different forms of contribution. Retrieval changes task success. Routing reduces teacher usage at the same observed task score. Risk and attribution serve separate informational purposes. RL produces no measured incremental task benefit here. An unchanged aggregate does not prove all individual outputs or downstream purposes are equivalent.

**Exercise:** the untuned-retriever variant is faster and less successful. Explain why selecting it solely on latency would reward early failure. Then compare the baseline's zero fallback rate with its 93.75% teacher usage: a teacher used as the primary model is not counted as a fallback.

**Exercise:** explain why the all-negative common risk cohort makes a low fixed prior look strong under Brier score. The result is informative about this cohort, but cannot establish whether the model ranks actual positives well. Use the separate D05 evaluation for that question.

A production proposal should name the intended user outcome, the stage being justified, its incremental benefit and its observed failure mode. The [production tradeoff guide](PRODUCTION_TRADEOFFS.md) gives a worked next-step decision using these results.
