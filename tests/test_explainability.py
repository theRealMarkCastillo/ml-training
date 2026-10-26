import polars as pl
import pytest

from ml_training.explainability import modify_row


@pytest.mark.parametrize("changes", [{"label": 1}, {"amount": -1}, {"amount": float("inf")},
                                     {"is_foreign": 2}, {"is_foreign": .5}, {"failed_auth": 1.5},
                                     {"merchant": "unknown"}])
def test_feature_edits_reject_invalid_values(changes):
    with pytest.raises(ValueError):
        modify_row(pl.DataFrame({"amount": [30.]}), changes)


def test_feature_edit_is_explicit_and_local():
    frame = pl.DataFrame({"amount": [30.], "entity_id": [1]})
    changed = modify_row(frame, {"amount": 45.})
    assert changed["amount"][0] == 45
    assert frame["amount"][0] == 30
    assert changed["entity_id"][0] == 1
