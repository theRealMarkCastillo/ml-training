from copy import deepcopy

import pytest

from ml_training.device_data import build, dispatch, parse_output


@pytest.mark.parametrize("command", [
    {"action": "set_temperature", "args": {"celsius": 100}},
    {"action": "set_temperature", "args": {"celsius": True}},
    {"action": "set_temperature", "args": {"celsius": float("nan")}},
    {"action": "schedule", "args": {"hour": 24}},
    {"action": "schedule", "args": {"hour": 2.5}},
    {"action": "shell", "args": {"command": "anything"}},
    {"action": "status", "args": {"extra": 1}},
])
def test_invalid_commands_cannot_change_device_state(command):
    state = {"celsius": 20, "jobs": []}
    original = deepcopy(state)
    with pytest.raises(ValueError):
        dispatch(command, state)
    assert state == original


def test_valid_command_creates_new_state_and_preserves_input():
    state = {"celsius": 20, "jobs": []}
    result = dispatch({"action": "schedule", "args": {"hour": 9}}, state)
    assert result["jobs"] == [{"hour": 9}]
    assert state["jobs"] == []


def test_device_templates_and_temperature_values_are_held_out():
    rows = build("local")
    families = [{r["family"] for r in rows if r["split"] == split} for split in ["train", "validation", "test"]]
    assert not families[0] & families[1] and not families[0] & families[2]
    train_values = {r["expected"]["args"]["celsius"] for r in rows if r["split"] == "train" and r["expected"]["action"] == "set_temperature"}
    assert not {22, 26} & train_values


def test_duplicate_command_keys_are_rejected():
    with pytest.raises(ValueError):
        parse_output('{"action":"status","action":"schedule","args":{}}')
