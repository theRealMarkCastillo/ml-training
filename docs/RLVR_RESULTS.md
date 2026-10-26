# Observed executable-reward training results

The fixed 50-step CPU GRPO pilot completed actual optimizer updates but did not improve held-out accuracy over its SFT initialization. Run: [`local-s41-20261026T180904Z-d20988`](../artifacts/rlvr/local-s41-20261026T180904Z-d20988/run.json). The experiment used one seed on Apple M2 Max with 96 GB RAM. These are synthetic task results, not evidence of general reasoning or self-correction.

## Quality and inference budgets

| Variant | Greedy hidden pass@1, 96 tasks | Sampled hidden pass@1 | Public-verifier selected at 4 | Independent challenge greedy / selected at 4 |
| --- | --- | --- | --- | --- |
| Untuned base | 0% | 0% | 0% | 0/8 / 0/8 |
| Fused SFT initialization | 92.71% (89/96) | 88.54% (85/96) | 97.92% (94/96) | 5/8 / 5/8 |
| SFT + GRPO | 92.71% (89/96) | 88.54% (85/96) | 97.92% (94/96) | 5/8 / 5/8 |
| Continued SFT, matched training time | 92.71% (89/96) | 89.58% (86/96) | 97.92% (94/96) | 5/8 / 5/8 |

All variants used maximum completion length 160. Greedy generation is distinct from temperature-1 sampling with four candidates. Selection chooses the first output passing every public executable fixture, or the first output if none passes; hidden checks then grade that choice. No failed request is removed from the denominator. The empirical “any candidate passes hidden checks” statistic is descriptive and is not used to select a response.

The 24 novel-composition tasks scored 17/24 greedy and 22/24 with verifier selection for all three SFT-derived variants. The seven greedy SFT failures all fell in this slice. SFT and RL produced exactly the same raw greedy text for all 96 test requests. The continued-SFT model reached the same aggregate best-of-four score with different failures: SFT/RL missed `rl-test-77` and `rl-test-80`; continued SFT missed `rl-test-79` and `rl-test-94`.

The untuned base produced no schema-valid candidate in this evaluation. SFT and RL generated schema-valid greedy outputs on 90/96 test tasks, of which 89 were semantically correct. Their sampled outputs were schema-valid in 347/384 cases. Continued SFT produced 91/96 schema-valid greedy outputs and 349/384 schema-valid samples. Syntax and semantics remain separate criteria.

Across all recorded test and challenge candidates, no candidate passed the public fixtures and failed the hidden fixtures. That is an observed count of zero in this bounded evaluation, not proof that the verifier has no exploitable blind spots. A dedicated contract check constructs a public-only passing candidate and confirms that best-of-N cannot secretly replace it using hidden results.

## Actual training evidence and the negative result

| Measure | GRPO | Continued SFT control |
| --- | --- | --- |
| Starting fused checkpoint | Same D02 artifact | Same D02 artifact |
| Initial new adapter SHA-256 | `0898e99c260f…` | `0898e99c260f…` |
| Optimizer updates | 50 | 290 |
| Measured training seconds | 255.518 | 255.693 |
| Generated / supervised completion tokens | 7,379 generated | 43,345 supervised |
| Final adapter differs from initialization | Yes | Yes |
| Final adapter SHA-256 | `01cb62d5cc1a…` | `89399f9fc603…` |

Both adapters were saved and reloaded before final evaluation, with parameter hashes checked. GRPO uses supported TRL 1.8.0, rank-8 LoRA on the last four query/value projection layers, AdamW at 2e-5 with linear decay, four completions per prompt, binary public execution rewards, no KL term and no weight decay. The SFT control uses the same initial adapter, constant 2e-5 learning rate and batch 4, stopping after the first update that crosses the measured GRPO wall-time budget. This matches host/device training time, not FLOPs, optimizer steps or tokens. SFT receives canonical answers; RL receives binary verification feedback.

Only **2 of 50 sampled reward groups had nonconstant rewards**; 48 were constant. The mean public reward was 0.99, with 50 distinct sampled prompts from a frozen pool of 500. The strong SFT initialization and easy sampled training tasks left little within-group reward contrast. The changed adapter establishes actual training, while the unchanged held-out scores establish that this particular run did not demonstrate a useful gain. Zero group variance removes the relative reward advantage; this observation is not a universal statement about all GRPO loss terms or configurations.

The initial MPS attempt was rejected after PyTorch emitted unraised Metal command-buffer errors. Its files remain under `local-s41-20261026T180522Z-541d1a` with failed status. The CPU retry kept the frozen datasets, hyperparameters and 50-step budget. The accepted process peaked at approximately 6.70 GB RSS. Its complete artifact directory is about 1.00 GB, dominated by the shared fused SFT checkpoint. The 42.6-minute process duration includes conversion, training, control training and the much larger greedy/sampled final evaluation; it is not the RL training duration.

The shared checkpoint is a fused transfer of the MLX D02 model into the PyTorch environment. Fusion/precision/backend differences can change generations. D03's SFT baseline is therefore evaluated directly in the same backend as RL, rather than treating D02's earlier MLX scores as an interchangeable control.

## Reproduction and recommendation

`artifacts/validation/rlvr-slices.json` contains post-evaluation novelty, syntax, verifier-blind-spot and paired-output analysis. `scripts/summarize_rlvr.py` only reads saved evidence; it never selects or changes a checkpoint. The separate locked-environment replay regenerated exact raw greedy and sampled outputs and matching metrics for all four variants, all 96 test tasks and all eight challenges. It took 2,024.5 seconds; [replay evidence](../artifacts/validation/rlvr.json) records the result.

The dedicated resource profile then ran without another model workload. Warm measurements used the same first 12 validation requests, one excluded warmup, batch/concurrency 1, four CPU threads, float32 and greedy generation capped at 160 tokens. Prompt tokenization and generation are included; model loading and semantic verification are excluded. Actual output lengths ranged from 21–60 tokens for base and 23–45 for each SFT-derived variant.

| Variant | Warm p50 seconds | Warm p95 seconds | Warm p99 seconds | Fresh process seconds | Fresh process peak RSS, decimal GB |
| --- | --- | --- | --- | --- | --- |
| Base | 2.102 | 2.604 | 2.756 | 6.233 | 3.437 |
| SFT | 1.938 | 2.115 | 2.116 | 5.554 | 3.438 |
| RLVR | 1.909 | 2.132 | 2.134 | 5.584 | 3.437 |
| Continued SFT | 1.919 | 2.132 | 2.137 | 5.526 | 3.437 |

[Raw resource evidence](../artifacts/validation/rlvr-performance.json) includes outputs, token counts and the environment. Each fresh-process number is one launch-to-exit observation including imports, recursive artifact checksums, model loading and one generation; filesystem caches are uncontrolled. Twelve warm samples do not establish precise tail latency, and the small SFT/RL timing differences do not establish a performance benefit. These CPU measurements are not interchangeable with D02's earlier MLX timings or with four-candidate inference cost.

Keep the SFT checkpoint as the simpler choice for this task. Verifier selection improved generated-test success within its four-sample budget, but added inference work and did not improve the eight independent challenges. The evidence does not justify the extra RL training stage for quality, nor unattended use on arbitrary queries. D09 includes the RL artifact to demonstrate the integration and its ablation, without implying it is superior. No hosted model API was used and local electricity/depreciation costs were not measured.
