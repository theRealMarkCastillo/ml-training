import numpy as np
import pytest

from ml_training.retrieval_metrics import metrics, negatives_from_scores


def examples():
    docs = [{"doc_id": f"doc{i}", "family": "example", "version": f"v{i}", "operation": "read", "split": "train"}
            for i in range(1, 4)]
    queries = [{"query_id": "q1", "family": "example", "version": "v2", "operation": "read", "positive_id": "doc2", "split": "train"}]
    return docs, queries


def test_rank_metric_distinguishes_version_error_from_recall():
    docs, queries = examples()
    value, _ = metrics(queries, docs, np.array([[0, 1, 2]]))
    assert value["recall_at_5"] == 1
    assert value["mrr_at_10"] == .5
    assert value["wrong_version_rate"] == 1
    assert value["wrong_resource_rate"] == 0


def test_negative_mining_excludes_positive_even_if_highest_score():
    docs, queries = examples()
    result = negatives_from_scores(queries, docs, [[.4, .9, .3]])
    assert result[0]["negative_ids"] == ["doc1", "doc3"]


def test_negative_mining_rejects_evaluation_documents():
    docs, queries = examples()
    docs[0]["split"] = "test"
    with pytest.raises(ValueError, match="training"):
        negatives_from_scores(queries, docs, [[.4, .9, .3]])
