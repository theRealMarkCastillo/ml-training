from copy import deepcopy

import pytest

from ml_training.retrieval_data import audit, build


def test_retrieval_families_and_templates_are_disjoint():
    corpus, queries = build("local")
    result = audit(corpus, queries)
    assert len(corpus) == 240
    assert sum(r["queries"] for r in result.values()) == 1092
    assert len(result["train"]["families"]) == 28
    assert len(result["test"]["families"]) == 6


def test_wrong_version_relevance_is_rejected():
    corpus, queries = build("local")
    queries[0]["positive_id"] = queries[0]["positive_id"].replace("v1", "v2")
    with pytest.raises(ValueError, match="relevance"):
        audit(corpus, queries)


def test_cross_partition_resource_leak_is_rejected():
    corpus, queries = build("local")
    train = next(q for q in queries if q["split"] == "train")
    test = next(q for q in queries if q["split"] == "test")
    test["family"] = train["family"]
    with pytest.raises(ValueError, match="leakage"):
        audit(corpus, queries)
