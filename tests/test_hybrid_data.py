import pytest
from ml_training.hybrid_data import build, calibration, validate_public


def test_oracles_cannot_enter_runtime_request():
    rows = build()
    assert len(rows)==32 and len({r['public']['id'] for r in rows})==32
    for row in rows:
        validate_public(row['public'])
        with pytest.raises(ValueError, match='oracle'):
            validate_public(row['public'] | {'expected':row['expected']})


def test_calibration_separate_and_all_branches_present():
    rows = build()
    assert not {r['public']['text'] for r in rows} & {r['text'] for r in calibration()}
    assert {r['expected']['outcome'] for r in rows} == {'reference','device','query','clarify','blocked','denied'}
    assert {r['public']['transaction_row'] for r in rows} == set(range(32))
