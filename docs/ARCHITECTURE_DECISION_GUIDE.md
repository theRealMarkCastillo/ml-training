# Architecture decision guide

Choose an approach by the failure it addresses, the evidence available to improve it, and the operating constraints it must satisfy. The default is the simplest candidate that meets the required outcome. “Use a model” and “train a model” are decisions to justify.

This guide helps you shortlist designs before implementing them. The [tutorials](LEARNING_GUIDE.md) explain how the repository's demos work; the [production tradeoff guide](PRODUCTION_TRADEOFFS.md) covers cost, rollout and ongoing operation.

**Evidence labels:** “In this repo” names implemented comparisons. “Beyond this repo” introduces alternatives in theory; their inclusion is not a claim that they were implemented, benchmarked or proved better here. Use the linked papers and official documentation to study the mechanisms, then evaluate your own workload. The criteria below are engineering judgment informed by those mechanisms and the lab results, not universal numerical thresholds.

## Start with the problem, not the technique

Write down the required outcome, available inputs, costly errors, allowed latency, privacy/connectivity constraints, expected traffic, and how often the task or knowledge changes. Include the people who will handle rejections, ambiguous cases and system failures.

Then identify the dominant failure:

| What is failing? | First candidates to compare | A misleading first response |
| --- | --- | --- |
| The process follows known rules but execution is inconsistent | Ordinary code, a typed form, decision table or state machine | Add a generative model to choose a rule already known exactly |
| Answers lack current or private facts | Direct database/API lookup, retrieval, or selected documents in context | Try to encode every changing fact into weights |
| Correct information is present but the model behaves inconsistently | Prompt/examples, SFT, a smaller task-specific classifier | Keep adding unrelated documents |
| Outputs violate a grammar | Typed input, constrained decoding, external validation; SFT if interpretation also fails | Treat fine-tuning as a guarantee of valid execution |
| Numeric or categorical features predict an outcome | Constant/rule baseline, logistic regression, boosted trees | Serialize every row into a generative prompt without a comparison |
| A correct model is too slow or large | Profile first; then batching, runtime optimization, compression or a smaller model | Train again without locating the bottleneck |
| Users need to understand or change a decision | An interpretable predictor, suitable attribution, or a causal study, depending on the question | Add a plausible explanation and assume the decision is validated |
| Multi-step task success is poor | Trace failures and compare a fixed workflow with bounded model decisions | Add more agents before identifying the failing stage |

These categories overlap. A production assistant might retrieve fresh facts, use an SFT model to interpret a request, and execute through deterministic code. Each part needs its own reason to exist.

~~~mermaid
flowchart TD
    A["Define outcome and constraints"] --> B["Identify the dominant failure"]
    B --> C["Shortlist an approach and a simpler baseline"]
    C --> D["Set quality, failure-cost and operating gates"]
    D --> E["Compare on representative held-out work"]
    E --> F{"Incremental benefit justifies complexity?"}
    F -- "No or unproven" --> G["Keep the simpler option or gather evidence"]
    F -- "Yes, within the tested scope" --> H["Bounded rollout with monitoring and rollback"]
~~~

## Separate decisions that can be combined

| Decision axis | Examples | What this axis does not decide |
| --- | --- | --- |
| Source of knowledge | Database/API, retrieved documents, full documents in context, model weights | How an action is authorized |
| Training signal | Labeled answers, teacher answers, preferences, executable rewards, unlabeled domain text | Where inference runs |
| Parameters updated | Full fine-tuning, LoRA and related parameter-efficient methods | Whether the task needs training at all |
| Numerical representation/runtime | Floating point, low-bit quantization, native runtime, exported or compiled graph | Whether the answer is semantically correct |
| Placement | On-device, local server, hosted service, a cascade between them | Which training objective is best |
| Control flow | Fixed workflow, learned routing, bounded agent loop, human review | Whether every component improves the final outcome |

For example, **LoRA versus RAG is not a complete either/or choice**. LoRA concerns how weights are adapted; retrieval concerns which evidence is supplied at request time. A model can use both. Likewise, ONNX export and local placement do not replace choosing a valid predictor.

## Use or avoid each demonstrated approach

### D01 — Distill a repeated language task

**Use when:** the task is frequent and stable enough to amortize labeling, training and maintenance; representative teacher outputs can be independently checked; and a smaller model can meet the required quality and abstention behavior. Distillation is a candidate for replacing expensive repeated inference, not a reason to accept lower task quality.

**Avoid or defer when:** traffic is too low to repay the fixed work, the intent set changes rapidly, the task requires broad knowledge absent from the inputs, or you cannot detect teacher-label errors. If a small classifier already satisfies the contract, a generative student may add unnecessary decoding and schema failures.

**In this repo:** D01 compares TF-IDF/logistic routing with regex extraction, the prompted base student, the teacher, and the distilled student. It learns from accepted final teacher answers, not teacher logits. D09 adds a student-to-teacher fallback policy.

**Beyond this repo:** a separately trained encoder classifier with an explicit abstention policy; embedding similarity to labeled class prototypes; or rules plus a small learned exception handler. Prototype classification compares an input vector with representative class vectors, which trades generative flexibility for a simpler scoring interface. An encoder classifier produces a label/score directly instead of generating the label token by token. A cascade spends the larger model's work only on selected requests; it must evaluate whether its confidence signal predicts errors, rather than treating token probability as correctness.

**Decision evidence:** class/slice quality, extraction accuracy, ambiguity and escalation behavior, fallback frequency, real request volume, and total refresh/serving cost. D01's 100% generated-test accuracy but 6/12 independent challenges is a reason to test coverage before making a cost argument. [Tutorial and results](SLM_DEMOS.md#d01-distillation).

### D02 — Tune structured generation

**Use when:** users need language flexibility, the output contract is explicit, and representative examples show persistent interpretation or format errors after a prompt baseline. Pair generation with an external executor that validates and authorizes the proposed action.

**Avoid or defer when:** a typed form or bounded parser fits the user interaction; the only problem is syntax that an appropriate decoding constraint can prevent; or no one can specify what correct semantics mean. Fine-tuning is a poor substitute for an unstable or ambiguous API contract.

**In this repo:** base and SFT generation are compared with a deterministic typed-slot parser. All use the bounded AST interpreter. There is no neural grammar-constrained decoder. D09 also demonstrates a bounded teacher-repair attempt.

**Beyond this repo:** a form/query builder; grammar- or schema-constrained decoding; or a typed tool-call interface that returns named operations and arguments for an external executor. Constrained decoding removes inadmissible token choices under its grammar; [PICARD](https://arxiv.org/abs/2109.05093) is one research example. Valid structure still does not establish the intended filter, permission or action. Repair adds latency and must stop rather than loop indefinitely.

**Decision evidence:** separate parsing, schema, execution and intent outcomes; unseen compositions; rejection behavior; schema migration costs; and repair budget. The parser wins the lab's generated grammar but covers only 1/8 independent challenges, illustrating exactness versus language coverage. [Tutorial](SLM_DEMOS.md#d02-strict-syntax-and-executable-semantics).

### D03 — Learn from verifiable rewards

**Use when:** outcomes can be checked with an informative, sufficiently robust verifier; the model sometimes samples successful and unsuccessful candidates; and quality improvement could justify sampling, verification and retraining cost. Compare against a strong SFT starting point.

**Avoid or defer when:** the reward is an unreliable proxy, verifier access leaks evaluation answers, almost every sampled group gets the same reward, or continued SFT/inference-time selection already meets the objective. “We lack labels” alone is not enough: a useful verifier is another substantial source of supervision.

**In this repo:** GRPO, its SFT initialization, a continued-SFT control matched by training wall time, and public-verifier selection among four samples. D03 changed weights without improving held-out greedy behavior.

**Beyond this repo:** rejection sampling followed by SFT on accepted training candidates; or preference training when judgments are comparative rather than executable. DPO learns from chosen/rejected answer pairs without the same online RL sampling loop; it does not turn subjective preference into verified task correctness. [DPO paper](https://arxiv.org/abs/2305.18290).

**Decision evidence:** verifier false accepts/rejects, reward contrast, independent hidden evaluation, training cost and matched inference budgets. Stop if the gain disappears when the SFT baseline gets the same candidate budget. [Tutorial](RLVR_DEMO.md).

### D04 — Run a compact model locally

**Use when:** network independence, request-data handling constraints or measured local latency justify device execution, and a suitable model fits the actual target's memory and compute budget. Quantization is a candidate for reducing representation cost; task adaptation is a separate decision.

**Avoid or defer when:** target-device accuracy, startup, memory or sustained thermal behavior fails the requirement; updates cannot be managed; or a hosted model already meets the constraints with less operational burden. Do not introduce an online fallback into an application whose offline/privacy requirement forbids it.

**In this repo:** an adapted model, a fused 4-bit export, a workstation-level denied-network test, and a deterministic device simulator. There is no physical-device integration.

**Beyond this repo:** a deterministic command interface; a smaller encoder; hosted inference; or a local/remote cascade where policy permits. For compression, compare a smaller architecture or distilled student with quantization; quantization-aware training simulates low-precision effects while adapting weights, while pruning removes or masks selected weights or structures. Actual speed depends on runtime support for the resulting representation. See [QAT](https://pytorch.org/blog/quantization-aware-training/) and [pruning](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html). QLoRA instead addresses training memory by updating adapters through a frozen quantized base; it is distinct from this demo's post-training export. [QLoRA paper](https://arxiv.org/abs/2305.14314).

**Decision evidence:** target-device quality and rejection behavior, measured resident memory, initialization, energy/thermal behavior, update/rollback and command acknowledgement. D04's valid-but-unintended simulator changes show why local execution is not an intent guarantee. [Tutorial](SLM_DEMOS.md#d04-local-model-and-simulated-device).

### D05 — Predict from tabular features

**Use when:** structured features available at decision time predict a well-defined labeled outcome. Compare a constant prior, simple rules and an additive model before accepting the extra interactions and complexity of boosted trees.

**Avoid or defer when:** labels do not represent the intended outcome, features arrive only after the decision, a deterministic calculation answers the question, or the real objective is an intervention effect rather than prediction. A high ranking score does not establish a usable alert threshold.

**In this repo:** supervised dummy/logistic/XGBoost prediction, chronological splits, probability calibration and validation-only threshold selection. Despite the use-case label “anomaly scoring,” this is not an unsupervised anomaly detector.

**Beyond this repo:** scorecards, generalized additive models (GAMs), other tree ensembles, or sequence/time-series models when event history itself is central. The latter model an ordered history rather than relying only on a fixed row of aggregate features; compare the extra state and compute with simpler lagged features. With few outcome labels, isolation forests or one-class methods can flag unusual observations for investigation. They do not directly estimate “probability of a bad outcome,” and unusual does not mean harmful. See the [official outlier/novelty guide](https://scikit-learn.org/stable/modules/outlier_detection.html).

**Decision evidence:** temporal/entity holdouts, prevalence, label delay, calibration, recall and false alerts at the proposed threshold. D05's future FPR exceeded its validation budget despite better AP. That can require policy/data work rather than a larger predictor. [Tutorial](CPU_DEMOS.md#d05-learn-a-tabular-risk-function).

### D06 — Optimize inference execution

**Use when:** profiling shows model execution is a meaningful bottleneck, the chosen runtime supports the computation, and conversion preserves the contract. Treat preprocessing and feature order as part of the model interface.

**Avoid or defer when:** I/O or queueing dominates, native inference already meets the budget, conversion is fragile for required operations, or deployment complexity outweighs the measured benefit. A model-only benchmark cannot justify a service-level claim.

**In this repo:** native versus ONNX inference, probability parity, batch/startup measurement and localhost HTTP. D06 is not a generative-language-model serving benchmark.

**Beyond this repo:** vectorized native code, keeping a model resident, batching, a device-specific runtime, or compilation such as [torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) for a suitable PyTorch model. Compilation has its own startup and graph-compatibility considerations. Exact-result caching is useful only when inputs, model/data versions and permission context define a valid cache key; approximate semantic caching can change answers and needs separate quality evaluation.

**Decision evidence:** parity including near-threshold cases, realistic input shapes, warm and startup latency, concurrency, queueing and memory. Measure a whole-request improvement before paying the long-term export/runtime maintenance cost. [Tutorial](CPU_DEMOS.md#d06-export-and-time-a-request-classifier).

### D07 — Explain predictions

**Use when:** a named audience needs to understand a particular model's feature use, and the explanation method matches the output space and question. TreeSHAP is appropriate for the demonstrated tree decomposition under its explicit background assumptions.

**Avoid or change approach when:** the question is causal, a user needs an actionable remedy that feature edits cannot establish, or the explanation is being used to excuse an inaccurate predictor. Additivity is not a validation of fairness, task quality or a decision policy.

**In this repo:** standardized logistic coefficients, raw-log-odds TreeSHAP, feature edits and a correlated-feature counterexample. The calibrated probability is displayed separately.

**Beyond this repo:** choose an interpretable predictor from the start, such as a GAM/Explainable Boosting Machine; these learn additive feature functions, potentially with selected interactions. [EBM documentation](https://interpret.ml/docs/ebm.html). For global reliance, [permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html) measures a score change after feature shuffling. For average or individual response curves, use [PDP/ICE](https://scikit-learn.org/stable/modules/partial_dependence.html), checking whether correlated inputs create implausible combinations. [LIME](https://arxiv.org/abs/1602.04938) fits a local interpretable surrogate; its fidelity depends on the chosen neighborhood. Counterfactual explanations search for model-input changes that alter an output; feasible and causal remedies require additional assumptions and evidence.

**Decision evidence:** local versus global purpose, output units, reference population, correlation sensitivity, fidelity/stability, and whether the intended audience makes better decisions with the explanation. [Tutorial](CPU_DEMOS.md#d07-inspect-and-change-an-explanation).

### D08 — Improve retrieval and ranking

**Use when:** error inspection shows relevant material is missing from the shortlist or poorly ordered, and relevance labels or a validated untuned alternative can address that failure. Tune the retriever for candidate quality; test a reranker only after checking that candidates contain the needed evidence.

**Avoid or defer when:** the document is absent/stale, chunking lost the needed context, authorization filters are wrong, or generation ignores already-correct evidence. Training a ranker does not repair those upstream/downstream problems. Avoid an added reranker if its incremental quality is negative or too small for its cost.

**In this repo:** BM25, base/tuned bi-encoders, cross-encoders, fixed-pool comparisons and exact vector search. It does not implement lexical/dense fusion or a production approximate index.

**Beyond this repo:** metadata filtering before candidate selection, hybrid lexical/dense retrieval, query rewriting, learned sparse retrieval, or late-interaction models. Compare these with the implemented lexical baseline for exact identifiers. Reciprocal rank fusion combines result positions rather than assuming different raw scores are comparable. [RRF paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). Learned sparse retrieval assigns weights to vocabulary terms, potentially expanding beyond literal matches while retaining inverted-index search; [SPLADE](https://arxiv.org/abs/2107.05720) is one example. ColBERT retains multiple token-level representations and combines them through late interaction, trading a richer matching signal for a different index/compute footprint. [ColBERT paper](https://arxiv.org/abs/2004.12832). Query rewriting can improve vocabulary matching but can also lose an explicit version or negation.

**Decision evidence:** candidate recall, rank quality on fixed pools, exact version/operation errors, corpus size, permission behavior, index freshness and query latency. D08's tuned bi-encoder beat its reranked pipeline, so “add a reranker” is a hypothesis, not a default. [Tutorial](RETRIEVAL_DEMO.md).

### D09 — Compose a workflow or agent

**Use when:** the outcome genuinely needs several capabilities and each stage has a demonstrated role. Prefer explicit control flow for known steps; let a model make only the decisions that require it.

**Avoid or defer when:** a single lookup, classifier or model call suffices; stage outcomes cannot be observed; or the environment lacks the state, permissions and error handling needed to execute actions reliably. Adding specialized models does not make their errors independent or make the aggregate easier to operate.

**In this repo:** a coded workflow with learned routing, bounded repair/fallback, deterministic gates, and separate task/risk/explanation outcomes. D09 does not demonstrate an autonomous multi-agent planning system.

**Beyond this repo:** ordinary application code, a durable state machine or workflow DAG, a single model with bounded tools, human review, or a bounded agent loop when the sequence of steps must adapt at runtime. The distinction between predefined workflows and model-directed agents is discussed in this [engineering guide](https://www.anthropic.com/research/building-effective-agents); it is a design distinction, not a recommendation to adopt a particular framework. Multiple agents should earn their coordination overhead through measured gains on separable work.

**Decision evidence:** complete task outcomes, per-stage traceability, aligned cohorts, retries, permission checks, stop conditions, failure recovery and component ablations. D09 reduced teacher usage while increasing measured p95, and its RL ablation changed no aggregate task score. [Tutorial](HYBRID_DEMO.md).

## Knowledge access: lookup, RAG, long context or training?

First ask whether the answer already exists in an authoritative structured system. A direct query or API call can be more exact and easier to keep current than generating an answer from prose. When the source is documents, retrieval-augmented generation (RAG) supplies selected material to a model; the [original RAG paper](https://arxiv.org/abs/2005.11401) combines parametric and retrieved memory.

| Candidate | Strong reason to compare it | Reason to avoid making it the default |
| --- | --- | --- |
| Direct lookup or deterministic template | Exact fields, current state, explicit source of truth | The task needs synthesis not expressible by the interface |
| Prompt plus a few examples | Behavior is mostly present and the task is still changing | Repeated long examples cost too much or persistent errors remain |
| Selected documents / long context | Relevant material is small enough and assembling it is straightforward | Full context is expensive, includes inaccessible material, or the model fails to use important details |
| RAG | Large/changing corpus, provenance, or selective access is needed | Retrieval is unnecessary for the task, or missing/poor evidence cannot support the answer |
| SFT, possibly alongside retrieval | Stable behavior must improve even when the needed facts are supplied | The real failure is stale/missing facts rather than behavior |
| Domain-adaptive continued pretraining | Specialized language/distribution remains a demonstrated bottleneck and suitable domain text is available | The goal is precise factual refresh, or simpler adaptations have not been tested |

A large context window is a capacity limit, not proof of reliable use of all positions. [Lost in the Middle](https://arxiv.org/abs/2307.03172) documented position-sensitive failures in its evaluated models; treat that as motivation to test the current model on your documents, not a claim that every current model has the same failure.

Continued pretraining uses an unlabeled domain corpus to adapt representations/language modeling; [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964) studies this approach. It differs from teaching a task with labeled SFT examples. Updating either kind of weights does not create a dependable database of current facts or a per-document access-control mechanism.

D09's document-edit experiment demonstrates changing retrieved reference information without changing weights. Its bounded fact extraction is not an implementation of every RAG architecture in the literature.

## Training choice: signal first, parameter method second

| What supervision do you actually have? | Candidate | What to compare or resolve first |
| --- | --- | --- |
| Reliable prompt/answer examples | SFT | Prompt-only and simpler task models; coverage and label consistency |
| A capable teacher plus independent quality checks | Distillation | Teacher errors, filtering bias, student capacity and economic payoff |
| Chosen/rejected answers reflecting a useful preference | DPO or another preference method | Annotator agreement and whether preference tracks the required outcome |
| Executable outcome feedback | RLVR or inference-time verification | Verifier quality, SFT and best-of-N at matched budgets |
| Relevant unlabeled domain text | Continued pretraining | Whether domain language, rather than task policy or knowledge freshness, is the actual bottleneck |
| No trustworthy supervision or evaluation | Data/specification work | Labeling, error discovery, rules or bounded human review before training claims |

After choosing the objective, compare parameter-update methods. **LoRA** reduces the trainable update space; **QLoRA** also uses a quantized frozen base during adaptation; **full fine-tuning** updates more or all model weights and generally carries a larger training/state footprint. More adjustable parameters are not a guaranteed improvement. Evaluate task gain, general-behavior regressions, training memory and serving compatibility.

These methods are not a mandatory escalation ladder. A good prompted model may be enough indefinitely, while a well-labeled numerical task may go directly to logistic regression without any language-model stage.

## Make the decision on constraints and evidence

Use hard requirements to eliminate infeasible candidates before comparing nice-to-have gains. “Offline required” cannot be compensated for by higher average accuracy from an online-only service. Among feasible candidates, compare quality, costly error slices, latency, total cost and operational effort; avoid averaging incompatible units into an unexplained score.

| Constraint or unknown | What to establish before choosing |
| --- | --- |
| Quality and error cost | Required outcomes, abstention policy, class/slice losses and who absorbs false positives |
| Training data | Rights to use it, representativeness, label/reward validity, leakage controls and refresh cadence |
| Latency and throughput | Full-path p95/p99 under target concurrency, queueing, retries and startup |
| Cost and volume | Upfront work plus recurring serving, fallback, refresh and operational costs |
| Freshness and privacy | Source-of-truth update path, permission filtering, placement and data retention |
| Operations | Owner, observability, reproducible release artifacts, rollback and incident path |
| Uncertainty | What remains unknown and the cheapest experiment that could change the decision |

Set acceptance criteria before opening the final evaluation. Use paired requests, aligned populations and the same intended service contract. A richer candidate can justify its cost, but a “no improvement” result should make keeping the baseline an ordinary outcome.

## Worked decisions

### Current, versioned API documentation

**Situation:** support answers must reflect frequently changing API fields. **First comparison:** direct structured lookup where possible, otherwise lexical/dense retrieval with explicit version and permission filters; add generation only for needed synthesis. **Defer:** encoding every field revision through SFT. **Revisit:** tune retrieval if representative relevance labels show that near-duplicate versions are the bottleneck; add reranking only if it improves the full pipeline within budget. D08/D09 motivate this design, but a real corpus evaluation is still required.

### A stable, high-volume intent router

**Situation:** a small intent set drives a known workflow. **First comparison:** rules or an encoder/classical classifier against a prompted model. **Consider distillation:** if the simple baseline misses necessary language variation, a teacher can supply checked labels, and a student meets independently tested abstention requirements. **Defer:** a generative router whose only advantage is template accuracy. **Revisit:** when intents, ambiguity patterns or fallback cost change. D01's challenge result is an example of why validation must represent those concerns.

### A bounded query language with a verifier

**Situation:** requests become a small AST, and executable correctness can be checked on meaningful public fixtures. **First comparison:** typed input/parser versus prompted or SFT generation with external gates. **Consider best-of-N:** if candidates vary in correctness and extra inference fits the budget. **Consider RLVR:** only after demonstrating useful reward contrast and a benefit over SFT at matched budgets. **Defer:** more RL steps solely because training reward is high. D03's null result supplies the counterexample.

### Commands on a disconnected device

**Situation:** the product must work without a network and has a fixed command set. **First comparison:** deterministic UI/grammar or a compact intent model on the actual device. **Consider a quantized language model:** if language flexibility adds enough user value and target-device behavior meets the contract. **Defer:** a cloud cascade that violates offline requirements, or treating a range-valid action as proof of intended action. **Revisit:** when the command set, hardware or update process changes. D04 provides a simulator exercise, not the hardware qualification.

## Write a small architecture decision record

Copy these fields into a project note. Separate measured facts from estimates and unresolved questions.

| Field | What to record |
| --- | --- |
| Decision and status | Proposed, experiment in progress, accepted within a stated scope, or superseded; name the owner and date |
| Context | User outcome, inputs, traffic, update frequency and costly failure modes |
| Hard constraints | Offline/privacy requirements, quality floor, maximum failure rates and operational budgets |
| Options | Simplest baseline, preferred candidate, and at least one credible alternative |
| Evidence | Dataset/split, model/runtime versions, matched budgets, results and confidence limits |
| Choice and rationale | Why this option satisfies the constraints and whether its incremental benefit earns the complexity |
| Rejected/deferred options | Why each loses under current evidence, and what would change that conclusion |
| Consequences | New dependencies, data work, failure paths, owners and ongoing costs |
| Rollout and rollback | Bounded exposure, monitoring, previous version/policy and recovery criteria |
| Revisit trigger | A measurable drift, traffic, cost, schema, hardware or capability change |

For a **lab-scoped** example: choose D08's tuned bi-encoder without a reranker because validation selected it and the recorded complete pipeline outperformed the reranked alternative. Record single-seed synthetic scope and exact-search corpus size. List hybrid lexical/dense fusion as untested rather than rejected by experiment. Reopen the decision on a new representative corpus or changed retrieval budget. Production acceptance remains a separate decision.

## Exercise: argue against your preferred design

Choose a use case and draft the record above. Then propose the strongest simpler alternative and identify the one experiment most likely to make you switch. If the only justification is that the repository demonstrates your preferred technique, the decision is incomplete.

Use [the production guide](PRODUCTION_TRADEOFFS.md) to develop operational consequences and [the final report](FINAL_REPORT.md) to see how the lab's positive and negative findings constrain conclusions. The alternatives linked here are starting points for comparison, not promises of superiority on your application.
