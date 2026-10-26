"""Bind fresh-UI evidence to the submitted CLI command and request."""

from pathlib import Path


def validate_submission(record, demo, submitted, variant, run_name):
    if not isinstance(record, dict):
        raise AssertionError("No new CLI execution was captured for this submission")
    command, result = record["command"], record["result"]
    assert command[1:3] == ["run", demo], "Captured a different demo command"
    assert Path(command[command.index("--run-dir") + 1]).name == run_name
    if demo == "hybrid":
        assert result["execution"] == "fresh_local_models"
        assert result["request"] == submitted
        assert result["response"]["id"] == submitted["id"]
        assert result["response"]["variant"] == variant
        assert result["response"]["trace"]
    elif demo == "explainability":
        assert result["execution"] == "fresh_saved_model"
        assert result["row_index"] == submitted["row_index"]
        assert result["updates"] == {"amount":submitted["amount"]}
    else:
        assert command[command.index("--query") + 1] == submitted
        assert result["variant"] == variant
        assert result["query" if demo == "retrieval" else "input"] == submitted
        assert "split" not in result, "A saved evaluation example is not a fresh response"
        if demo == "retrieval":
            assert result["top_five"]
        else:
            assert isinstance(result["output"]["text"], str)
    return result
