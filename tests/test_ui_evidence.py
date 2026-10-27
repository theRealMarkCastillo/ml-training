import copy

import pytest

from ml_training.ui_evidence import validate_submission


def execution():
    return {
        "command": [
            "ml-demo",
            "run",
            "distillation",
            "--run-dir",
            "/runs/local-41",
            "--query",
            "A new request",
            "--variant",
            "tuned_student",
        ],
        "result": {"variant": "tuned_student", "input": "A new request", "output": {"text": "model output"}},
    }


def test_fresh_submission_requires_a_new_captured_execution():
    with pytest.raises(AssertionError, match="No new CLI execution"):
        validate_submission(None, "distillation", "A new request", "tuned_student", "local-41")


def test_saved_evaluation_example_cannot_pass_as_a_fresh_result():
    record = execution()
    record["result"].update(split="test", id="test-0", expected={})
    with pytest.raises(AssertionError, match="saved evaluation"):
        validate_submission(record, "distillation", "A new request", "tuned_student", "local-41")


@pytest.mark.parametrize("field,value", [("input", "Previous request"), ("variant", "teacher")])
def test_fresh_result_must_match_the_submitted_input_and_variant(field, value):
    record = execution()
    record["result"][field] = value
    with pytest.raises(AssertionError):
        validate_submission(record, "distillation", "A new request", "tuned_student", "local-41")


def test_correct_fresh_response_is_retained_exactly():
    record = execution()
    original = copy.deepcopy(record)
    result = validate_submission(record, "distillation", "A new request", "tuned_student", "local-41")
    assert result == original["result"] and record == original


def test_different_run_cannot_supply_fresh_evidence():
    with pytest.raises(AssertionError):
        validate_submission(execution(), "distillation", "A new request", "tuned_student", "other-run")
