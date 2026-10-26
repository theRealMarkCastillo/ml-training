# Retrieval experiment results

Observed on 2026-10-26, using Apple M2 Max MPS, float32, one training seed (41). D08 has actual trained bi-encoder and cross-encoder artifacts. [PROGRESS.md](PROGRESS.md) records the overall lab status.

## Main finding

**Validation selected the tuned bi-encoder alone.** Its held-out top-result accuracy was 99.3%, compared with 37.5% for the base bi-encoder and 41.7% for BM25. Adding the tuned reranker reduced accuracy to 88.9%; extra model stages were not automatically beneficial.

On the identical base candidate pool, tuning the reranker improved MRR@10 from 0.8854 to 0.9201. That isolates a genuine positive reranker effect, while the full-pipeline comparison shows why it was not selected for this task.

## Held-out evaluation

The corpus has 240 synthetic pages. Training uses 840 queries across 28 resource families; validation uses 108 queries across six different families; test uses 144 queries across another six families. Query phrasing families are disjoint too. Only training documents and queries participate in negative mining.

| Variant | Top-1 | Recall@10 | MRR@10 | nDCG@10 | Wrong version |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.4167 | 1.0000 | 0.6644 | 0.7491 | 0.1667 |
| base_bi | 0.3750 | 0.9861 | 0.5711 | 0.6718 | 0.4306 |
| tuned_bi | 0.9931 | 1.0000 | 0.9965 | 0.9974 | 0.0000 |
| tuned_bi_base_cross | 0.8611 | 1.0000 | 0.9306 | 0.9487 | 0.1319 |
| tuned_bi_tuned_cross | 0.8889 | 1.0000 | 0.9444 | 0.9590 | 0.1111 |
| fixed_base_pool_base_cross | 0.8125 | 0.9861 | 0.8854 | 0.9111 | 0.1806 |
| fixed_base_pool_tuned_cross | 0.8542 | 0.9861 | 0.9201 | 0.9374 | 0.1458 |

The base candidate pool contains the correct page for 98.6% of test queries; its rerankers cannot exceed that Recall@10. The tuned pool contains it for all test queries. Wrong-version errors are reported separately from wrong resource or operation.

## Independent challenge queries

| Variant | Top-1 correct / 12 | MRR@10 |
| --- | --- | --- |
| bm25 | 4 / 12 | 0.5722 |
| base_bi | 5 / 12 | 0.6028 |
| tuned_bi | 12 / 12 | 1.0000 |
| tuned_bi_tuned_cross | 11 / 12 | 0.9583 |

These separately authored cases include negation, aliases, read-versus-update distinctions and competing version references. Twelve cases are useful failure probes, not a broad reliability estimate.

## Training and reload evidence

- bi_encoder: 106 optimizer steps executed over two epochs; validation selected epoch 1 at step 53. Training/evaluation/checkpoint-selection elapsed 21.0 seconds. Saved parameter hashes differ from the base checkpoint and reproduce after reload.
- cross_encoder: 158 optimizer steps executed over two epochs; validation selected epoch 1 at step 79. Training/evaluation/checkpoint-selection elapsed 26.5 seconds. Saved parameter hashes differ from the base checkpoint and reproduce after reload.

Both checkpoints use genuine gradient updates; neither is a pretrained-only wrapper. Every model/tokenizer file is hashed in the run manifest. A separate environment installed from the frozen transformer lock reproduced all seven comparisons, all top-ten rankings, candidate pools, and saved parameter hashes.

## Timing boundaries

Warm query embedding plus exact search over precomputed corpus vectors: p95 6.21 ms. Reranking ten query-document pairs: p95 14.16 ms. These use fixed queries, concurrency 1, five warmups, and 30 samples; corpus embedding construction is excluded.

Fresh-process first inference took 3.27 seconds for the bi-encoder and 3.50 seconds for the cross-encoder. Those single-sample checks include Python startup, imports, checkpoint/data loading and inference; OS caches are uncontrolled.

## Representative failures

- base_bi: for `automation-rule-v1-read-q1`, returned `automation-rule-v3-update` ahead of `automation-rule-v1-read` (correct page rank 5).
- tuned_bi: for `audit-export-v3-update-q2`, returned `audit-export-v3-read` ahead of `audit-export-v3-update` (correct page rank 2).
- tuned_bi_tuned_cross: for `automation-rule-v2-update-q0`, returned `automation-rule-v1-update` ahead of `automation-rule-v2-update` (correct page rank 2).

## Evidence and limits

- [Full run report](../artifacts/retrieval/local-s41-20261026T165108Z-bcab85/report.md).
- [Run manifest and hashes](../artifacts/retrieval/local-s41-20261026T165108Z-bcab85/run.json).
- [All held-out predictions](../artifacts/retrieval/local-s41-20261026T165108Z-bcab85/predictions.json).
- [Clean-environment reproduction](../artifacts/validation/retrieval.json).
- [Walkthrough and commands](RETRIEVAL_DEMO.md).

This is a single-seed experiment on generated API documentation with only six held-out resource families. The data generator shares a simple API contract across families, so the result demonstrates learning that narrow contract; it does not establish effectiveness on arbitrary enterprise search. No model was retuned after test results were opened. The other tracks and the comparison interface have their own evidence; [PROGRESS.md](PROGRESS.md) records their current status.
