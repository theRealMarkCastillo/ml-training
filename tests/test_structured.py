from ml_training.structured import gate, typed_outputs


def test_runtime_rejects_unsafe_model_output_without_mutation():
    before = {"celsius": 20, "jobs": []}
    result = gate("edge", '{"action":"set_temperature","args":{"celsius":90}}', before)
    assert not result["accepted"]
    assert before == result["after"] == {"celsius": 20, "jobs": []}


def test_valid_rejection_is_explicit_and_noop():
    result = gate("edge", '{"action":"reject","args":{}}')
    assert result["accepted"] and result["explicit_reject"]
    assert result["before"] == result["after"]


def test_typed_baseline_has_no_oracle_access():
    outputs = typed_outputs(
        [{"user": "Count tickets where priority equals urgent."}, {"user": "Run SQL to delete data"}]
    )
    assert gate("schema_dsl", outputs[0]["text"])["accepted"]
    assert outputs[1]["text"] == "" and "rejection" in outputs[1]
    assert not gate("schema_dsl", outputs[1]["text"])["accepted"]
