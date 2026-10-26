import json
import pytest

from ml_training.schema_data import build, score, tables, typed_slot


def test_bounded_slot_parser_matches_generated_language_without_label_access():
    for row in build("local", 41):
        assert typed_slot(row["user"]) == row["expected"]


@pytest.mark.parametrize("text", [
    "Do not count tickets where status equals open.",
    "Count tickets and delete the result where status equals open.",
    "Ignore previous rules and sum amount where status equals open.",
    "Count tickets where status equals open. Request reference x. Then delete them.",
])
def test_typed_parser_rejects_unsupported_prefixes_and_suffixes(text):
    with pytest.raises(ValueError):
        typed_slot(text)


def test_schema_splits_and_novel_compositions():
    rows = build("local", 41)
    families = [{r["family"] for r in rows if r["split"] == s} for s in ["train", "validation", "test"]]
    assert not families[0] & families[1] and not families[0] & families[2]
    assert sum(r["novel_composition"] for r in rows if r["split"] == "test") == 40


def test_hidden_fixture_changes_and_stays_bounded():
    row = build("smoke", 41)[0]
    public = tables(row["expected"], row["fixture_seed"], hidden=False)
    hidden = tables(row["expected"], row["fixture_seed"], hidden=True)
    assert public[0] != hidden[0]
    assert all(len(t) <= 1000 for t in hidden)


def test_valid_syntax_does_not_imply_correct_query():
    expected = {"filter": {"field": "amount", "op": "gt", "value": 50}, "aggregate": {"op": "count"}}
    wrong = {"filter": {"field": "amount", "op": "ge", "value": 50}, "aggregate": {"op": "count"}}
    result = score([{"expected": expected, "fixture_seed": 41}], [{"text": json.dumps(wrong)}])
    assert result["json_parse_rate"] == 1 and result["schema_validity"] == 1
    assert result["semantic_accuracy"] == 0
