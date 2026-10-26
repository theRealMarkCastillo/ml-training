import pytest

from ml_training.common import complete_run, digest, verify_run, write_json


def test_tampered_saved_model_cannot_pass_checksum_check(tmp_path):
    model = tmp_path / "model.joblib"
    model.write_bytes(b"original model")
    write_json(tmp_path / "run.json", {"status": "evaluated", "files": {"model.joblib": {"sha256": digest(model)}}})
    verify_run(tmp_path)
    model.write_bytes(b"changed model")
    with pytest.raises(ValueError, match="checksum"):
        verify_run(tmp_path)


def test_incomplete_run_does_not_count_as_verified(tmp_path):
    write_json(tmp_path / "run.json", {"status": "started"})
    with pytest.raises(ValueError, match="incomplete"):
        verify_run(tmp_path)


def test_nested_transformer_weights_are_in_manifest(tmp_path):
    nested = tmp_path / "bi_encoder" / "model.safetensors"
    nested.parent.mkdir()
    nested.write_bytes(b"saved checkpoint")
    write_json(tmp_path / "run.json", {"profile": "local"})
    complete_run(tmp_path)
    verify_run(tmp_path)
    nested.write_bytes(b"modified checkpoint")
    with pytest.raises(ValueError, match="checksum"):
        verify_run(tmp_path)
