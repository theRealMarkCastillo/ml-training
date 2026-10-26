import json

from ml_training import rlvr_data, schema_data


def test_rl_tasks_disjoint_from_sft_and_each_other():
    sft = schema_data.build("local", 41)
    rows = rlvr_data.build("local", 41, sft)
    seen = {rlvr_data.ast_key(r["expected"]) for r in sft}
    for row in rows:
        key = rlvr_data.ast_key(row["expected"])
        assert key not in seen
        seen.add(key)
    assert len(rows) == 660
    assert sum(r["novel_composition"] for r in rows if r["split"] == "test") == 24


def test_reward_only_credits_all_public_cases():
    row = {"expected": {"filter": {"field": "age_hours", "op": "gt", "value": 12}, "aggregate": {"op": "count"}}, "fixture_seed": 411}
    assert rlvr_data.reward(json.dumps(row["expected"]), row) == 1.
    wrong = {**row["expected"], "filter": {"field": "age_hours", "op": "ge", "value": 12}}
    assert rlvr_data.reward(json.dumps(wrong), row) == 0.
    assert rlvr_data.reward("[]", row) == 0.
    assert rlvr_data.reward("```json\n" + json.dumps(row["expected"]) + "\n```", row) == 0.


def test_reward_public_tables_exclude_hidden_random_rows(monkeypatch):
    flags = []
    original = schema_data.tables
    def observed(ast, seed, hidden=True):
        flags.append(hidden)
        return original(ast, seed, hidden)
    monkeypatch.setattr(schema_data, "tables", observed)
    row = schema_data.build("smoke", 41)[0]
    rlvr_data.reward(row["answer"], row)
    assert flags == [False]
