"""Retrieval metrics and training-negative contracts, independent of torch."""

import numpy as np


def ranking(scores):
    """Stable ties preserve corpus order for reproducible rank comparisons."""
    return np.argsort(-np.asarray(scores), axis=1, kind="stable")


def metrics(queries, corpus, ranks):
    doc_position = {d["doc_id"]: i for i, d in enumerate(corpus)}
    values = []
    for query, order in zip(queries, ranks, strict=True):
        target = doc_position[query["positive_id"]]
        places = np.flatnonzero(np.asarray(order) == target)
        rank = int(places[0]) + 1 if len(places) else len(corpus) + 1
        top = corpus[int(order[0])]
        values.append({"query_id": query["query_id"], "family": query["family"], "rank": rank,
                       "recall_at_5": float(rank <= 5), "recall_at_10": float(rank <= 10),
                       "mrr_at_10": 1 / rank if rank <= 10 else 0.,
                       "ndcg_at_10": 1 / np.log2(rank + 1) if rank <= 10 else 0.,
                       "top1_accuracy": float(rank == 1),
                       "wrong_version_rate": float(top["version"] != query["version"]),
                       "wrong_resource_rate": float(top["family"] != query["family"]),
                       "wrong_operation_rate": float(top["operation"] != query["operation"])})
    keys = [k for k in values[0] if k not in ["query_id", "family", "rank"]]
    aggregate = {key: float(np.mean([v[key] for v in values])) for key in keys}
    aggregate["queries"] = len(queries)
    return aggregate, values


def family_bootstrap(rows, seed):
    families = sorted({r["family"] for r in rows})
    rng = np.random.default_rng(seed)
    means = [np.mean([r["mrr_at_10"] for r in rows if r["family"] == f]) for f in families]
    samples = [np.mean(rng.choice(means, len(means), replace=True)) for _ in range(500)]
    return {"metric": "mrr_at_10", "lower": float(np.percentile(samples, 2.5)),
            "upper": float(np.percentile(samples, 97.5)), "families": len(families),
            "method": "500 equal-family bootstrap samples; small synthetic family set limits inference"}


def negatives_from_scores(queries, documents, scores):
    if any(q["split"] != "train" for q in queries) or any(d["split"] != "train" for d in documents):
        raise ValueError("Negative mining must use training queries and documents only")
    result = []
    for query, order in zip(queries, ranking(scores), strict=True):
        candidates = [documents[int(i)]["doc_id"] for i in order if documents[int(i)]["doc_id"] != query["positive_id"]][:2]
        if len(candidates) != 2 or query["positive_id"] in candidates:
            raise ValueError("Invalid hard-negative set")
        result.append({"query_id": query["query_id"], "positive_id": query["positive_id"], "negative_ids": candidates})
    return result
