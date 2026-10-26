# D08: Version-aware retrieval and reranking

Visual companions: [D08 notebook](../notebooks/D08_retrieval.ipynb). These plot the recorded evidence and include editable examples without loading models; see the [notebook guide](../notebooks/README.md).

This demo fine-tunes two real transformer models independently: a bi-encoder for vector retrieval and a cross-encoder for reranking. It compares both with BM25 and their untuned checkpoints. The latest observed experiment is summarized in [RETRIEVAL_RESULTS.md](RETRIEVAL_RESULTS.md).

## The learning question

When search returns the wrong API version, is the problem candidate retrieval, candidate ranking, or both? By the end, you should be able to choose a comparison that isolates each stage and explain why a “better reranker” may still make the full system worse.

## The concept: encode separately, then optionally score jointly

A **bi-encoder** maps the query and each document into vectors independently. Document vectors can be computed ahead of time; a request needs one query encoding and a similarity search. A **cross-encoder** reads each query-document pair together and assigns a relevance score. It can model their interaction but must perform work for each pair, so it is normally applied to a shortlist. The [Sentence Transformers retrieval/reranking guide](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) describes this architecture.

~~~mermaid
flowchart LR
    D["Documents"] --> I["Precomputed document vectors"]
    Q["Query"] --> B["Query vector"]
    I --> K["Top ten candidates"]
    B --> K
    K --> C["Optional cross-encoder scores"]
    C --> R["Final ranking"]
~~~

The lab uses exact vector search over a small corpus, not an approximate-nearest-neighbor service. BM25 supplies a lexical baseline. In this dataset, “v2 read” versus “v3 update” can matter more than broad topical similarity.

### What each training objective teaches

The bi-encoder gets one positive and two hard negatives for each query. A hard negative is a wrong document that the base model ranks highly, often a plausible near miss. With similarity logits `s_j = cosine(query, document_j) / 0.05`, its per-query loss is `-log(exp(s_positive) / sum_j exp(s_j))`. This encourages the correct page to outrank the selected alternatives. It is the actual three-document objective in [retrieval_models.py](../src/ml_training/retrieval_models.py), not an all-batch-negative objective.

The cross-encoder trains on labeled pairs using binary cross-entropy with raw logits. Its scores order candidates; they are not established calibrated probabilities of relevance. Negative mining uses training queries and documents only.

### Why candidate recall is a ceiling

If the relevant page is missing from a ten-document candidate set, reranking those ten cannot recover it. If it is present at rank 4, that query contributes 1 to Recall@10 and 1/4 to MRR@10. Moving it to rank 1 improves MRR and top-1 without changing recall. This lab has one labeled relevant document per query; nDCG uses that binary relevance definition too.

## Setup and model revisions

Use the separate transformer environment so the CPU environment remains reproducible:

```bash
UV_CACHE_DIR="$PWD/.cache/uv" uv sync --project environments/transformers --frozen
source environments/transformers/.venv/bin/activate
```

The exact checkpoints are pinned in `configs/models.json`:

| Role | Model | Revision |
| --- | --- | --- |
| Bi-encoder | sentence-transformers/all-MiniLM-L6-v2 | 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 |
| Cross-encoder | cross-encoder/ms-marco-MiniLM-L6-v2 | 233902d25c440f23af6f7d6e94d2946bac0bee0a |

Both are approximately 22.7M parameters. Their official model cards declare Apache 2.0 licenses: [bi-encoder](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [cross-encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2). Training uses the documented [Sentence Transformer](https://sbert.net/docs/sentence_transformer/training_overview.html) and [cross-encoder](https://sbert.net/docs/cross_encoder/training_overview.html) model APIs with explicit PyTorch optimizer loops.

The loader expects these snapshots in the standard Hugging Face cache. Set `ML_DEMO_MODEL_CACHE` to another cache root if necessary. It fails explicitly when files are missing and does not silently download a different model. On a machine without the snapshots, download the exact revisions once before enabling offline mode:

```bash
hf download sentence-transformers/all-MiniLM-L6-v2 --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
hf download cross-encoder/ms-marco-MiniLM-L6-v2 --revision 233902d25c440f23af6f7d6e94d2946bac0bee0a
```

After both downloads complete, enable offline loading for the experiment:

~~~bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
~~~

The observed run used `--device mps` on Apple Silicon. The process needs access to Metal; in a restrictive sandbox, MPS may report unavailable even when the host supports it. `--device cpu` is supported but timings and floating-point tie behavior may differ. Pin the device for exact ranking reproduction.

## Train and evaluate

```bash
ml-demo prepare retrieval --profile local --seed 41
ml-demo train retrieval --profile local --seed 41 --device mps
ml-demo evaluate retrieval --run-dir artifacts/retrieval/<run> --device mps
ml-demo verify retrieval --run-dir artifacts/retrieval/<run>
```

Training prints a unique run directory and progress every 20 optimizer steps. `--profile smoke` uses a smaller training/evaluation workload on a 36-page corpus with the same model architectures and labels the resulting run as smoke evidence. It does not replace the local experiment.

The full corpus has 240 pages: 40 resource families, three API versions, and two operations. There are 840 training queries, 108 validation queries and 144 test queries, covering 28/6/6 disjoint resource families. Query template families are also disjoint. The corpus is synthetic and deliberately contains near-duplicate version and read/update pages.

The bi-encoder is trained with a per-query contrastive loss over one positive and two hard negatives, mined by the base encoder from training documents only. Other queries' positives are not automatically treated as negatives. The cross-encoder is trained separately with binary cross-entropy over 2,520 labeled pairs. Both use AdamW, learning rate 2e-5, float32, two epochs, and gradient clipping. Token limits are 160 for the bi-encoder and 192 for the cross-encoder.

Each model's best trained checkpoint is selected by validation MRR@10. Ties preserve the earlier epoch. A second choice selects the full retrieval pipeline on validation MRR@10; ties prefer the simpler listed pipeline. Only then are test and independent challenge queries evaluated. Training manifests distinguish total optimizer steps executed from the selected checkpoint's epoch/history.

## Inspect real retrieval

```bash
ml-demo run retrieval --run-dir artifacts/retrieval/<run> --device mps \
  --query "Inspect the v2 thermostat settings without changing them."
```

The command loads the selected pipeline and returns five actual corpus pages. Compare it with a baseline using `--variant bm25`, `--variant base_bi`, or `--variant tuned_bi_tuned_cross`. The result names the variant and artifact directory. No stored answer is replayed.

## Read the evidence

- `bi_encoder/` and `cross_encoder/` contain saved weights and tokenizer/configuration files.
- `*_training.json` record losses, optimizer steps, model revisions, input hashes, before/after parameter hashes, changed weights and checkpoint selection.
- `training_negatives.json` identifies every query, positive document and mined negative pair.
- `selection.json` records the validation choice before test evaluation.
- `metrics.json` covers seven comparisons, including two rerankers on the identical base candidate pool.
- `predictions.json` retains every query's top ten document IDs, correct-document rank, and version/resource/operation errors.
- `candidate_pools.npz` freezes the candidate ordering used by each reranker.
- `challenge.json` evaluates 12 separately authored queries containing negation, aliases and competing version references.
- `benchmarks.json` separates single-query vector search from reranking ten pairs. Corpus embedding construction is outside warm query timing.
- `run.json` recursively hashes every checkpoint and output file. A modified weight file invalidates verification.

Recall@10 is the candidate-recall ceiling for reranking ten hits. Reranking cannot recover an absent positive. Read the fixed-pool comparison to isolate reranker changes from improvements to first-stage retrieval. Wrong-version rate counts any top result with a different explicit version; it is not a synonym for overall failure rate.

## Clean-environment reproduction

```bash
UV_PROJECT_ENVIRONMENT=.venv-repro UV_CACHE_DIR="$PWD/.cache/uv" \
  uv sync --project environments/transformers --frozen
environments/transformers/.venv-repro/bin/python scripts/validate_retrieval.py \
  artifacts/retrieval/<run> --device mps
```

This runs contract tests, checks saved parameter hashes, reconstructs all seven rankings/metrics, and measures first inference in separate fresh Python processes. It writes `artifacts/validation/retrieval.json`. Cold-process measurements include imports and checkpoint loading; they do not flush OS file caches.

The demo is a single-seed pilot on six held-out synthetic resource families. It is not evidence of performance on real enterprise documentation, and the independently authored challenge set is too small for a broad reliability claim.


## What was learned, and production tradeoffs

The [measured experiment](RETRIEVAL_RESULTS.md) selected the tuned bi-encoder alone: 99.3% held-out top-1 accuracy. Adding the tuned reranker reduced it to 88.9%. On the **same base candidate pool**, however, reranker tuning improved MRR@10 from 0.8854 to 0.9201. Both observations are true: one isolates a reranker improvement, while the other compares complete pipelines.

Tune retrieval when relevance failures reflect domain distinctions that labeled pairs can teach. Add reranking when its incremental quality is useful at the allowed candidate count and latency. A lexical or hybrid search baseline can remain strong for exact identifiers and versions; hybrid lexical/dense fusion is an extension to compare, not an implemented variant here.

Production introduces corpus size, approximate search, index refresh, permission filtering, stale content and broader query distributions. A 240-page exact-search experiment does not measure those effects. Changing an encoder also requires rebuilding document vectors in its space. Changing documents requires reindexing their content; neither should silently mix incompatible embedding versions.

## Try it and reason about the result

1. Search for a read operation, an update operation and a competing-version request. Inspect document IDs as well as text similarity.
2. In `candidate_pools.npz` and `predictions.json`, identify a wrong top result where the correct document is still in the shortlist. Contrast it with a missing-candidate failure.
3. Compare the two `fixed_base_pool` variants before comparing the tuned full pipelines. Explain which comparison isolates the cross-encoder and which changes the upstream candidate source.
4. Propose a larger-corpus evaluation with a fixed query set and labeled near-duplicate versions. Measure retrieval recall before deciding whether to spend more on reranking. No numerical gain is promised.

Follow [retrieval_data.py](../src/ml_training/retrieval_data.py) for family splits, [retrieval_models.py](../src/ml_training/retrieval_models.py) for the objectives, [retrieval_metrics.py](../src/ml_training/retrieval_metrics.py) for ranking definitions, and [retrieval.py](../src/ml_training/retrieval.py) for validation selection and complete comparisons.

For an architectural choice, compare the use/avoid criteria and alternatives for [D08](ARCHITECTURE_DECISION_GUIDE.md#d08--improve-retrieval-and-ranking) in the decision guide.
