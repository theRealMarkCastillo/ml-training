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
    row = {
        "expected": {"filter": {"field": "age_hours", "op": "gt", "value": 12}, "aggregate": {"op": "count"}},
        "fixture_seed": 411,
    }
    assert rlvr_data.reward(json.dumps(row["expected"]), row) == 1.0
    wrong = {**row["expected"], "filter": {"field": "age_hours", "op": "ge", "value": 12}}
    assert rlvr_data.reward(json.dumps(wrong), row) == 0.0
    assert rlvr_data.reward("[]", row) == 0.0
    assert rlvr_data.reward("```json\n" + json.dumps(row["expected"]) + "\n```", row) == 0.0


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


def test_hidden_v2_fixtures_share_no_table_with_public_reward():
    for row in schema_data.build("local", 41)[:200]:
        public = schema_data.tables(row["expected"], row["fixture_seed"], hidden=False)
        hidden = schema_data.hidden_tables(row["expected"], row["fixture_seed"])
        assert not any(table in public for table in hidden)
        assert all(0 < len(t) <= 1000 for t in hidden)


def test_hidden_v1_still_grades_recorded_runs_with_original_fixtures():
    row = schema_data.build("smoke", 41)[0]
    original = schema_data.tables(row["expected"], row["fixture_seed"], hidden=True)
    seen = {}

    def capture(text, expected, fixtures):
        seen["fixtures"] = fixtures
        return {"passed": True}

    import ml_training.rlvr_data as module

    module_equivalent = module.equivalent
    try:
        module.equivalent = capture
        rlvr_data.verify(row["answer"], row, hidden=True, hidden_version="v1")
        assert seen["fixtures"] == original
        rlvr_data.verify(row["answer"], row, hidden=True)
        assert seen["fixtures"] == schema_data.hidden_tables(row["expected"], row["fixture_seed"])
    finally:
        module.equivalent = module_equivalent


def test_hidden_v2_catches_threshold_shift_that_public_witnesses_miss():
    import json

    row = {
        "expected": {"filter": {"field": "amount", "op": "gt", "value": 50}, "aggregate": {"op": "ids"}},
        "fixture_seed": 41,
    }
    # Treats 52 as above the threshold too: identical on 49-51 and on 0/110.
    shifted = {
        "filter": {"any": [{"field": "amount", "op": "gt", "value": 52}, {"field": "amount", "op": "eq", "value": 51}]},
        "aggregate": {"op": "ids"},
    }
    from ml_training.query_dsl import equivalent

    public_witness = schema_data.tables(row["expected"], 41, hidden=False)[1]
    assert equivalent(json.dumps(shifted), row["expected"], [public_witness])["passed"]
    assert not rlvr_data.verify(json.dumps(shifted), row, hidden=True)["passed"]


def test_hidden_v2_is_not_weaker_at_range_extremes():
    import json

    row = {
        "expected": {
            "filter": {
                "any": [
                    {"field": "status", "op": "eq", "value": "closed"},
                    {"field": "team", "op": "ne", "value": "support"},
                ]
            },
            "aggregate": {"op": "count"},
        },
        "fixture_seed": 41,
    }
    # A recorded D03 output: an extra clause that only matters when amount is 0.
    extra = json.loads(json.dumps(row["expected"]))
    extra["filter"]["any"].append({"field": "amount", "op": "le", "value": 0})
    assert not rlvr_data.verify(json.dumps(extra), row, hidden=True)["passed"]
