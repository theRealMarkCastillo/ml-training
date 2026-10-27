import pytest

from ml_training.routing_data import audit, build, parse_output, score


def test_routing_splits_preserve_families_and_account_domains():
    rows = build("local")
    report = audit(rows)
    assert report["train"]["rows"] == 2016
    assert report["validation"]["rows"] == 72
    assert report["test"]["rows"] == 144
    parts = [
        {r["expected"]["account_id"] for r in rows if r["split"] == s} - {None} for s in ["train", "validation", "test"]
    ]
    assert not parts[0] & parts[1] and not parts[0] & parts[2]


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"intent":"billing","account_id":null}\n```',
        '{"intent":"billing","intent":"access","account_id":null}',
        '{"intent":"billing","account_id":true}',
        '{"intent":"unknown","account_id":null}',
        '{"intent":"billing","account_id":null,"extra":1}',
    ],
)
def test_router_rejects_invalid_schema(text):
    with pytest.raises(ValueError):
        parse_output(text)


def test_invalid_predictions_are_in_accuracy_denominator():
    rows = [{"expected": {"intent": "billing", "account_id": None}}] * 2
    outputs = [{"text": '{"intent":"billing","account_id":null}'}, {"text": "not JSON"}]
    result = score(rows, outputs)
    assert result["joint_accuracy"] == 0.5
    assert result["schema_validity"] == 0.5
