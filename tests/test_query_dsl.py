import json

import pytest

from ml_training.query_dsl import InvalidQuery, equivalent, execute, parse

AST = {"filter": {"field": "age_hours", "op": "gt", "value": 12}, "aggregate": {"op": "count"}}


@pytest.mark.parametrize(
    "text",
    [
        '{"filter":{},"filter":{},"aggregate":{"op":"count"}}',
        '{"filter":{"field":"amount","op":"gt","value":NaN},"aggregate":{"op":"count"}}',
        json.dumps({"filter": {"field": "__import__", "op": "eq", "value": "os"}, "aggregate": {"op": "count"}}),
        json.dumps({**AST, "extra": "ignored?"}),
        json.dumps({"filter": {"field": "amount", "op": "gt", "value": True}, "aggregate": {"op": "count"}}),
        json.dumps({"filter": {"field": "amount", "op": "gt", "value": 10**1000}, "aggregate": {"op": "count"}}),
    ],
)
def test_untrusted_ast_is_strictly_validated(text):
    with pytest.raises(InvalidQuery):
        parse(text)


def test_semantic_verifier_rejects_valid_but_wrong_boundary():
    wrong = {"filter": {"field": "age_hours", "op": "ge", "value": 12}, "aggregate": {"op": "count"}}
    tables = [[{"id": "a", "age_hours": 12}, {"id": "b", "age_hours": 13}]]
    result = equivalent(json.dumps(wrong), AST, tables)
    assert result["valid"] and not result["passed"]


def test_empty_table_alone_can_hide_an_error_but_second_fixture_exposes_it():
    wrong = {"filter": {"field": "age_hours", "op": "lt", "value": 12}, "aggregate": {"op": "count"}}
    result = equivalent(json.dumps(wrong), AST, [[], [{"id": "a", "age_hours": 13}]])
    assert result["cases_passed"] == 1
    assert not result["passed"]


def test_bounded_depth_and_table_size():
    nested = AST["filter"]
    for _ in range(4):
        nested = {"all": [nested]}
    with pytest.raises(InvalidQuery, match="complexity"):
        parse(json.dumps({"filter": nested, "aggregate": {"op": "count"}}))
    with pytest.raises(InvalidQuery, match="row count"):
        execute(AST, [{}] * 1001)


def test_empty_mean_is_null_not_nan():
    assert execute({**AST, "aggregate": {"op": "mean", "field": "amount"}}, []) is None
