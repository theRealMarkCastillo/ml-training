# D03: executable-reward GRPO

Status: the full 50-step CPU pilot completed actual GRPO updates, checkpoint reload, hidden evaluation and a matched-time SFT control. It did not improve held-out accuracy over SFT; see [observed results](RLVR_RESULTS.md). Clean-environment exact-output replay and the separate latency/startup profile passed; [PROGRESS.md](PROGRESS.md) records the overall lab status.

Use CPU for the full local profile. The MPS compatibility probe passed, but a longer run emitted Metal command-buffer errors without raising an exception; that run was rejected. The CPU retry preserves the fixed dataset, step budget and hyperparameters. Use the isolated environment because current TRL uses a different Transformers version from the retrieval environment:

```bash
uv sync --project environments/rlvr --frozen
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
environments/rlvr/.venv/bin/python scripts/probe_grpo.py --device cpu
environments/rlvr/.venv/bin/ml-demo train rlvr --profile smoke --seed 41 \
  --source-run artifacts/schema_dsl/<completed-run> --device cpu
environments/rlvr/.venv/bin/ml-demo train rlvr --profile local --seed 41 \
  --source-run artifacts/schema_dsl/<completed-run> --device cpu
```

The CPU probe is a tiny letter-choice task, not query training. Its zero mean scalar loss can coexist with nonzero gradients because group-normalized positive and negative advantages cancel in the value of the on-policy objective. Parameter hashes and nonzero gradient norms establish that updates actually occurred. Do not infer task success from the probe.

The real runner uses the supported [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer), pinned to TRL 1.8.0. It fuses a verified D02 adapter, records validation generation before/after fusion, and loads that same saved checkpoint for both the RL policy and the continued-SFT control. Floating-point fusion/backend changes are evaluated explicitly rather than assuming identical generations.

There are 500 RL training tasks, 64 reserved validation tasks and 96 final test tasks. Canonical ASTs are disjoint across these partitions and from every D02 source example/challenge. Test phrasing and a 24-example composition slice are held out. The eight D02 independent challenge cases are also excluded from RL training.

The reward is 1 only when a schema-valid candidate passes all public executable fixtures; otherwise it is 0. There is no format bonus, empty-table partial credit or hidden-case access. Hidden evaluation changes the randomized fixture seed and includes boundary witnesses. The AST interpreter bounds source length, nodes, depth and rows and offers no arbitrary-code execution. This bounds work structurally instead of executing untrusted code with a timeout.

The local profile uses 50 actual GRPO optimizer steps, four candidates per prompt, temperature 1, maximum completion length 160, rank-8 LoRA over the last four query/value projection layers, learning rate 2e-5, no KL term and no weight decay. The checkpoint is the final fixed-budget step, with no test-driven selection. Raw groups, rewards, variance, generated token counts, optimizer logs and before/after parameter hashes are retained. A run with no parameter change fails the completion check.

The SFT control starts at the identical checkpoint and initial adapter parameters. It receives canonical training answers and runs for the measured GRPO training wall time on the same device, stopping after the first update crossing that budget. This matches host time, not FLOPs, steps or token counts; those differences are reported. GRPO pays for sampling and verification within its budget.

Final comparisons include base, SFT, RLVR and continued SFT. Each gets greedy pass@1 and four stochastic candidates at the same temperature, seed and token limit. Best-of-four uses only public checks to select the first passing response, then grades it on hidden fixtures. If none passes public checks, the first output is graded and remains in the denominator. The report exposes publicly passing outputs that fail hidden fixtures.

```bash
environments/rlvr/.venv/bin/ml-demo evaluate rlvr --run-dir artifacts/rlvr/<run> --device cpu
environments/rlvr/.venv/bin/ml-demo run rlvr --run-dir artifacts/rlvr/<run> --device cpu \
  --query "Count tickets where age_hours is greater than 24." --variant rlvr
```

This is a bounded synthetic experiment. It does not establish general reasoning, useful self-correction, production readiness or superiority over SFT. The independent challenge is small. Runtime validation establishes grammar validity; semantic correctness still requires task specifications and suitable fixture coverage.
