"""D02/D03 foundation: a bounded typed AST interpreter, never eval or SQL."""

from __future__ import annotations

import json
import math

FIELDS = {"priority": str, "status": str, "team": str, "age_hours": (int, float), "amount": (int, float)}
ENUMS = {
    "priority": {"low", "normal", "urgent"},
    "status": {"open", "pending", "closed"},
    "team": {"billing", "support", "security"},
}
MAX_ROWS = 1000
MAX_DEPTH = 3
MAX_NODES = 15


class InvalidQuery(ValueError):
    pass


def _exact(obj, keys):
    if not isinstance(obj, dict) or set(obj) != set(keys):
        raise InvalidQuery(f"Expected exact keys: {sorted(keys)}")


def parse(text):
    if not isinstance(text, str) or len(text) > 4096:
        raise InvalidQuery("Query must be text of at most 4096 characters")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise InvalidQuery("Duplicate JSON key")
            result[key] = value
        return result

    try:
        ast = json.loads(
            text,
            object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(InvalidQuery("Nonfinite number")),
        )
    except (ValueError, RecursionError) as error:
        raise InvalidQuery(str(error)) from error
    validate(ast)
    return ast


def validate(ast):
    _exact(ast, {"filter", "aggregate"})
    nodes = 0

    def visit(node, depth):
        nonlocal nodes
        nodes += 1
        if depth > MAX_DEPTH or nodes > MAX_NODES:
            raise InvalidQuery("Filter exceeds bounded complexity")
        if not isinstance(node, dict):
            raise InvalidQuery("Filter must be an object")
        if "all" in node or "any" in node:
            key = "all" if "all" in node else "any"
            _exact(node, {key})
            if not isinstance(node[key], list) or not 1 <= len(node[key]) <= 3:
                raise InvalidQuery("Logical groups need 1-3 children")
            for child in node[key]:
                visit(child, depth + 1)
        else:
            _exact(node, {"field", "op", "value"})
            field, op, value = node["field"], node["op"], node["value"]
            if not isinstance(field, str) or field not in FIELDS or not isinstance(op, str):
                raise InvalidQuery("Unknown field or operation")
            if isinstance(value, bool) or not isinstance(value, FIELDS[field]):
                raise InvalidQuery("Wrong value type")
            if field in ENUMS:
                if value not in ENUMS[field] or op not in {"eq", "ne"}:
                    raise InvalidQuery("Invalid categorical predicate")
            elif op not in {"eq", "ne", "gt", "ge", "lt", "le"} or not 0 <= value <= 100000 or not math.isfinite(value):
                raise InvalidQuery("Invalid numerical predicate")

    visit(ast["filter"], 1)
    aggregate = ast["aggregate"]
    if not isinstance(aggregate, dict):
        raise InvalidQuery("Aggregate must be an object")
    op = aggregate.get("op")
    if op in ["count", "ids"]:
        _exact(aggregate, {"op"})
    elif op in ["sum", "mean"]:
        _exact(aggregate, {"op", "field"})
        if aggregate["field"] not in ["age_hours", "amount"]:
            raise InvalidQuery("Only numeric fields can be aggregated")
    else:
        raise InvalidQuery("Unknown aggregate")


def execute(ast, rows):
    validate(ast)
    if not isinstance(rows, list) or len(rows) > MAX_ROWS:
        raise InvalidQuery("Table exceeds bounded row count")

    def matches(node, row):
        if "all" in node:
            return all(matches(child, row) for child in node["all"])
        if "any" in node:
            return any(matches(child, row) for child in node["any"])
        left, right = row[node["field"]], node["value"]
        op = node["op"]
        if op == "eq":
            return left == right
        if op == "ne":
            return left != right
        if op == "gt":
            return left > right
        if op == "ge":
            return left >= right
        if op == "lt":
            return left < right
        return left <= right

    selected = [row for row in rows if matches(ast["filter"], row)]
    operation = ast["aggregate"]["op"]
    if operation == "ids":
        return sorted(row["id"] for row in selected)
    if operation == "count":
        return len(selected)
    values = [row[ast["aggregate"]["field"]] for row in selected]
    if operation == "sum":
        return sum(values)
    return sum(values) / len(values) if values else None


def equivalent(candidate_text, expected_ast, tables):
    """Multiple independent fixtures reduce empty-result and constant-answer exploits."""
    if not tables:
        raise ValueError("Verifier requires at least one fixture table")
    try:
        candidate = parse(candidate_text)
    except InvalidQuery:
        return {"valid": False, "passed": False, "cases_passed": 0, "cases": len(tables)}
    actual = [execute(candidate, rows) for rows in tables]
    expected = [execute(expected_ast, rows) for rows in tables]
    correct = [a == e for a, e in zip(actual, expected, strict=True)]
    return {"valid": True, "passed": all(correct), "cases_passed": sum(correct), "cases": len(tables)}
