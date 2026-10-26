# D03: executable-reward GRPO

Status: the full 50-step CPU pilot completed actual GRPO updates, checkpoint reload, hidden evaluation and a matched-time SFT control. It did not improve held-out accuracy over SFT; see [observed results](RLVR_RESULTS.md). Clean-environment exact-output replay and the separate latency/startup profile passed; [PROGRESS.md](PROGRESS.md) records the overall lab status.

## The learning question

Can executable feedback improve a model that already learned the output format through supervised fine-tuning? By the end of this lesson, you should be able to distinguish “parameters changed,” “training reward was high,” and “held-out behavior improved.” They are separate claims.

Complete [D02](SLM_DEMOS.md#d02-strict-syntax-and-executable-semantics) first. You need its saved SFT artifact and the typed interpreter. This lesson is a bounded query experiment, not a general reasoning curriculum.

## The concept: learn from relative reward

In SFT, each prompt comes with an answer to imitate. In reinforcement learning with verifiable rewards (RLVR), the policy samples candidate answers and a checker assigns rewards. Here reward is binary: the candidate must satisfy the grammar and pass all public execution fixtures.

GRPO compares candidates from the same prompt using relative reward. A simplified advantage is `A_i = (r_i - mean(group_rewards)) / (std(group_rewards) + epsilon)`. It supplies a direction for increasing the likelihood of relatively better samples, within the trainer's policy optimization objective. GRPO was introduced in [DeepSeekMath](https://arxiv.org/abs/2402.03300); the [TRL guide](https://huggingface.co/docs/trl/grpo_trainer) describes its implementation and configuration variants.

The exact loss depends on the trainer version and settings. This lab pins its environment and disables the KL term. The equation above explains the reward signal, not every term of the installed trainer.

Consider three illustrative four-candidate groups:

| Rewards | What the relative signal can say |
| --- | --- |
| `[1, 1, 1, 1]` | All sampled candidates passed; reward does not distinguish among them |
| `[0, 0, 0, 0]` | All failed; reward gives no relative direction toward a better candidate |
| `[1, 0, 1, 0]` | Passing samples have higher relative advantage than failing samples |

Very easy and very hard prompts can both provide weak contrast under binary rewards. A high mean reward can therefore coexist with little useful group-relative learning.

~~~mermaid
flowchart TD
    P["Prompt from training split"] --> S["Sample four candidates"]
    S --> V["Public grammar and execution verifier"]
    V --> G["Relative reward within group"]
    G --> U["Update LoRA policy"]
    U --> E["Separate held-out and hidden evaluation"]
~~~

## A verifier is a specification with blind spots

A count query can accidentally agree with the target on an empty table. Random tables and threshold-boundary witnesses reduce that blind spot, but agreement on finite fixtures is still not proof of equivalence for all inputs. Public fixtures supply training/inference feedback; hidden fixtures grade the selected answer later.

This separation also matters for **best-of-N inference**. Generating four candidates and selecting a publicly passing one spends extra inference work. It is not the same intervention as improving the trained policy. Selecting whichever answer passes hidden tests would leak evaluation information unavailable in the deployed decision.

## Run the experiment

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


## What was learned, and production tradeoffs

The [recorded pilot](RLVR_RESULTS.md) performed 50 GRPO updates and changed adapter hashes. Yet only 2/50 sampled groups contained different rewards, and greedy outputs matched the SFT baseline on all 96 test requests. Continued SFT trained for approximately the same wall time also did not establish a useful greedy gain. These observations make a strong lesson about reward saturation after a good SFT initialization; they do not prove that RLVR cannot help other tasks.

Public verifier selection increased SFT/RL success from 89/96 greedy to 94/96 with four candidates. The independent challenge remained 5/8. Report that gain with its inference budget; attributing it to RL training would be incorrect.

Production RLVR is most plausible when task feedback is meaningful, resistant to easy shortcuts, and cheap enough to evaluate many samples. Verifier engineering, sampling and regression evaluation can dominate the effort. If supervised targets are readily available, continued SFT is an essential comparison. If candidates are already good, inference-time checking may be the simpler first experiment. This lab's measurements support retaining SFT for the demonstrated task.

## Try it and reason about the result

1. Inspect `grpo_training.json` and the raw reward groups. Count constant groups, then compare that count with held-out quality. Explain why a 0.99 mean reward was not evidence of improvement here.
2. Compare SFT, RLVR and continued SFT at greedy decoding, then compare all three with four-candidate selection. Keep training intervention and inference budget separate.
3. Read the public-only candidate-selection test. Construct a candidate that passes a weak public fixture but fails a hidden witness; the selector must not use the hidden result to replace its choice.
4. Propose a more discriminating **training** curriculum containing some solvable and some failing samples. Keep the final test untouched. Do not claim that increasing the step count alone will fix absent reward contrast.

Read [rlvr.py](../src/ml_training/rlvr.py) for reward, control training and evaluation; [rlvr_data.py](../src/ml_training/rlvr_data.py) for disjoint task construction; and [check_rlvr_selection.py](../scripts/check_rlvr_selection.py) for the selection boundary. Current upstream documentation may evolve; the lockfile and saved training record define this experiment.

For an architectural choice, compare the use/avoid criteria and alternatives for [D03](ARCHITECTURE_DECISION_GUIDE.md#d03--learn-from-verifiable-rewards) in the decision guide.
