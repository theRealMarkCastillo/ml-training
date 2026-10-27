import copy

from ml_training.hybrid import execution_content, grade
from ml_training.hybrid_data import build


def test_replay_ignores_measurement_noise_but_preserves_model_content():
    record = {
        "elapsed_seconds": 1.0,
        "trace": [{"seconds": 0.1, "output": {"text": "answer", "sequence_confidence": 0.9}}],
    }
    changed = copy.deepcopy(record)
    changed["elapsed_seconds"] = 5.0
    changed["trace"][0]["seconds"] = 2.0
    assert execution_content(record) == execution_content(changed)
    changed["trace"][0]["output"]["text"] = "different answer"
    assert execution_content(record) != execution_content(changed)


def test_query_scoring_uses_hidden_semantics_including_wrong_aggregates():
    scenario = build()[16]
    correct = {"outcome": "query", "result": {"ast": scenario["expected"]["value"]}, "trace": []}
    assert grade(scenario, correct)["correct"]
    wrong = copy.deepcopy(correct)
    wrong["result"]["ast"]["aggregate"] = {"op": "ids"}
    assert not grade(scenario, wrong)["correct"]


def test_guard_reject_is_failure_for_legitimate_request_but_not_false_rejection_for_unsafe_one():
    rejected = {"outcome": "rejected", "trace": []}
    assert grade(build()[8], rejected)["false_rejection"]
    assert not grade(build()[11], rejected)["false_rejection"]
    assert not grade(build()[11], rejected)["correct"]  # Still in the all-request success denominator.
