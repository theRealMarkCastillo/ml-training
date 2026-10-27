import json
import pytest
from ml_training.hybrid_policy import calibrate, checked_device, checked_query, document_guard
from ml_training.hybrid_data import build


def test_validation_threshold_can_reject_every_prediction():
    rows = [{"intent": "billing"}, {"intent": "access"}]
    predictions = [{"parsed": {"intent": "incident"}, "sequence_confidence": 1.0}] * 2
    result = calibrate(rows, predictions)
    assert result["coverage"] == 0 and result["threshold"] > 1


def test_schema_valid_write_cannot_bypass_authorization():
    request = build()[9]["public"]
    with pytest.raises(PermissionError):
        checked_device('{"action":"set_temperature","args":{"celsius":22}}', request)


def test_public_query_spec_rejects_boundary_semantic_error():
    wrong = {"filter": {"field": "age_hours", "op": "ge", "value": 24}, "aggregate": {"op": "count"}}
    with pytest.raises(ValueError, match="public specification"):
        checked_query(json.dumps(wrong), "Count tickets where age_hours is greater than 24.")


def test_wrong_document_version_rejected():
    request = build()[0]["public"]
    assert not document_guard({"version": "v2", "family": request["resource"], "operation": "read"}, request)
