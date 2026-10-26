"""Assert that best-of-N selection cannot use hidden verifier results."""

from unittest.mock import patch

from ml_training.rlvr import group_metrics

def verifier(text, row, hidden=False):
    passed = text == "correct" or text == "public_only" and not hidden
    return {"passed": passed, "valid": True, "cases": 3, "cases_passed": 3 if passed else 0}

row = {"id": "challenge", "user": "Count a hidden boundary", "expected": {}}
with patch("ml_training.rlvr.data.verify", verifier):
    group = [{"text": "public_only", "generation_tokens": 1}, {"text": "correct", "generation_tokens": 1}]
    metrics, records = group_metrics([row], [group])
    assert records[0]["selected_index"] == 0
    assert metrics["verifier_selected_success_at_n"] == 0.
    assert metrics["empirical_any_success_at_n"] == 1.
    assert metrics["public_pass_hidden_failure_rate"] == 1.
    group = [{"text": "wrong", "generation_tokens": 1}] * 4
    metrics, records = group_metrics([row], [group])
    assert metrics["rows"] == 1 and metrics["verifier_selected_success_at_n"] == 0.
    assert records[0]["selected_index"] == 0
print("RLVR selection checks passed: no hidden-oracle choice or denominator exclusion")
