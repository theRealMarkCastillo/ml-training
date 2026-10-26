# From a teaching demo to a production decision

Production changes the question from “can this model learn?” to “does this system deliver an acceptable outcome under the traffic, failure costs and operating constraints we actually have?” The lab supplies mechanisms and bounded observations. This guide proposes follow-up experiments; none of the production capabilities below is claimed to be implemented by the lab.

Use the [architecture decision guide](ARCHITECTURE_DECISION_GUIDE.md) to shortlist approaches and alternatives. This companion guide develops the cost, data, execution and rollout consequences after that choice. Its summary below is an entry point; the decision guide gives explicit use/avoid criteria for every demo.

## Choose the intervention that matches the failure

| Observed problem | First comparison to make | Why training may help | What you take on |
| --- | --- | --- | --- |
| Repeated stable routing with a large-model bill | Rules/classical router, teacher, distilled student with fallback | A student may learn a narrow decision boundary | Teacher-label quality, coverage shifts, retraining and fallback operations |
| Frequent output-format errors | Typed interface, prompt, constrained decoding, SFT | SFT may improve adherence and task interpretation | Schema migrations and semantic evaluation; valid syntax still needs authorization |
| Correctness can be checked cheaply | SFT versus verifier selection versus RL | Reward feedback may improve sampled behavior | Verifier blind spots, exploration cost and reward saturation |
| Connectivity, privacy or latency prevents cloud inference | Local baseline and target-device model | Local execution can remove a network dependency | Storage, memory, thermal limits, updates and offline failure handling |
| Structured numeric/categorical prediction | Constant prior, logistic regression, boosted trees | Trees can capture useful interactions | Feature freshness, label delay, calibration and threshold ownership |
| Inference overhead dominates | Native versus exported runtime with identical inputs | A deployment runtime may reduce execution overhead | Export compatibility, parity, preprocessing and service bottlenecks |
| A user needs to understand a score | Coefficients and model-specific explanations | Attribution can expose how the model used features | Explanation semantics, correlated inputs and user comprehension |
| Relevant documents rank poorly | Lexical search, embeddings, tuning, reranking separately | Training can distinguish domain-specific near misses | Relevance labels, index refresh, access control and retrieval evaluation |

Constrained neural decoding is a comparison worth exploring for D02, but this repository implements a typed parser baseline and post-generation validation instead. Likewise, D04 is local inference on a workstation with a simulator, not a shipped device integration.

## Price the complete path

A useful planning equation is:

`expected request cost = first-stage cost + fallback_probability × fallback_cost + expected retry cost + serving/operations cost`

If the fallback probability is 0.4, a cheap first stage still invokes the expensive path often. This equation is an accounting model, not a dollar measurement from the repository. D09 observed teacher use on 43.75% of requests and higher p95 latency than its baseline, despite reducing teacher usage.

For distillation, an illustrative break-even count is:

`N = (labeling cost + training cost + evaluation cost) / per-request savings`

All terms must use the same unit and satisfy the same quality requirement. D01's report contains a time-only illustration mixing its measured labeling/training workload with per-request timing. It does not price electricity, engineering, incident handling or model refreshes. In production, remeasure batch size, utilization and fallback frequency with the actual workload before using a financial estimate.

Latency needs its own analysis. Sequential stages and retries add elapsed time, while queueing depends on arrival rate and available capacity. A lower average cost does not guarantee a better tail. Benchmark the whole request path under target concurrency, including slow and failing dependencies, before setting an SLO.

## Data is part of the deployed system

For D05/D06, reproduce feature definitions at the moment a decision is made. A historically correct training table can still contain features that arrive too late online. Track missingness, entity coverage, label delay and changed prevalence. Select thresholds against an explicit false-positive or workload budget, then monitor whether that budget holds later.

For D01/D02/D04, collect representative task wording, including ambiguous requests and explicit non-actions. A teacher-generated target is a proposal, not ground truth. Record which labels were rejected and which classes lost coverage during curation. Test human-authored and future-period requests separately from generator templates.

For D08, document IDs, versions, permissions and index revisions must travel with retrieval results. A high semantic similarity score cannot authorize access or decide that v1 is acceptable when the caller requested v2. Establish a refresh and rollback process for both encoder changes and document changes.

## Keep model output behind a real execution boundary

The lab's AST interpreter and device dispatcher make a useful architectural point: model output is input to application code. That code owns the allowed operations, argument types, resource limits and authorization. D04 also demonstrates the boundary's limit: a schema-valid command can still be an unintended action.

A production device integration would need identity, current-state checks, command idempotency, acknowledgement handling and an appropriate confirmation policy for consequential actions. A query integration would need data permissions and workload budgets in addition to syntax checks. These follow from the proposed use cases; the simulator does not validate them.

D09's explicit caller context is convenient teaching scaffolding. A production service would have to obtain trustworthy identity and permission context from its authentication/authorization system. User-controlled request fields cannot establish those permissions.

## Design an evaluation before a rollout

| Question | Useful follow-up experiment | Which lab limitation motivates it? |
| --- | --- | --- |
| Does the model transfer to our users? | Frozen, representative future-period and independently labeled requests | Synthetic templates and very small challenge sets |
| Is the gain stable? | Independent training seeds and time/entity/document-family holdouts | One neural seed and shared generators |
| Does it fail acceptably? | Slice evaluation for abstention, valid-but-wrong actions and rare costly errors | Routing escalation and device rejection failures |
| Is the comparison fair? | Same input cohort, inference budget, hardware and baseline | Hybrid risk cohorts differ; RL selection uses extra samples |
| Will it meet latency and memory budgets? | Target hardware, realistic load, startup, queueing and failure injection | Concurrency-one local measurements and shared worker RSS |
| Can we operate it? | Shadow evaluation, bounded rollout, observability and rollback rehearsal | No deployed service or customer traffic in this lab |

Keep the previous model and the exact decision policy available during a rollout. Log enough context to connect a bad outcome to a model, data/index version, threshold and downstream action. Logging itself needs data minimization and retention decisions when using customer inputs. These are design tasks for a real application, not hidden features of the demo.

## Explanations need a consumer and a purpose

D07 can help a developer identify a surprising feature dependency. A support agent may instead need a concise, tested reason code; an affected user may need a correction or appeal path. Those are different interfaces. An additive decomposition of raw model log-odds does not explain the calibrated probability in percentage points and does not establish a causal remedy.

For each explanation, specify the model/output space, reference population, intended audience and permitted inference. If a feature is correlated with another, ask how alternative background choices change attribution. Evaluate whether the explanation helps the intended user make a better decision, rather than assuming a plausible chart is sufficient.

## A worked decision: should this lab's hybrid be deployed?

The hybrid solved 27/32 synthetic requests versus the baseline's 19/32. That supports continuing investigation of the integrated approach, especially retrieval. It also produced false rejections, a schema-valid misunderstanding and a missed query condition. Teacher use fell, but p95 latency rose. The tiny common risk cohort contained no positives, so it could not establish useful discrimination.

A defensible next step is a new representative evaluation with an explicit task-success and failure-cost definition. Retain the simpler SFT query baseline because the current RL comparison added no quality gain. Compare the selected bi-encoder with a lexical/hybrid search alternative on real versioned documentation. Measure the complete service under load. The current evidence does not justify declaring the nine-stage composition a preferred production architecture.

## Further study

Use the primary sources linked beside the concepts in the [CPU](CPU_DEMOS.md), [language-model](SLM_DEMOS.md), [RLVR](RLVR_DEMO.md) and [retrieval](RETRIEVAL_DEMO.md) tutorials. Their papers and current library documentation describe mechanisms and options; the repository's lockfiles and saved manifests define what this particular experiment executed.
