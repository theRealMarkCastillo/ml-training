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


def test_reproduction_mismatch_names_drift_not_tampering():
    from ml_training.common import ReproductionMismatch, require_reproduced
    saved = {"ap": 0.5, "rows": 10, "texts": ["a", "b"]}
    require_reproduced({"ap": 0.5, "rows": 10, "texts": ["a", "b"]}, saved, "metrics")
    with pytest.raises(ReproductionMismatch, match=r"checksums passed.*result\.ap: 0\.5000001 != saved 0\.5"):
        require_reproduced({**saved, "ap": 0.5000001}, saved, "metrics")
    require_reproduced({**saved, "ap": 0.5000001}, saved, "metrics", tolerance=1e-6)
    with pytest.raises(ReproductionMismatch, match=r"result\.texts\[1\]"):
        require_reproduced({**saved, "texts": ["a", "c"]}, saved, "text", tolerance=1)


def test_failed_run_is_marked_instead_of_left_started(tmp_path):
    from ml_training.common import mark_failed, read_json
    write_json(tmp_path / "run.json", {"status": "started"})
    mark_failed(tmp_path, RuntimeError("out of memory"))
    assert read_json(tmp_path / "run.json") == {"status": "failed", "error": "RuntimeError: out of memory"}


def test_interrupted_prepare_explains_recovery(tmp_path):
    from ml_training.common import new_dataset_folder
    new_dataset_folder(tmp_path / "data")
    with pytest.raises(FileExistsError, match="interrupted.*Delete that folder"):
        new_dataset_folder(tmp_path / "data")


def test_missing_environment_names_the_install_command(tmp_path, monkeypatch):
    from ml_training import common
    (tmp_path / "configs").mkdir()
    write_json(tmp_path / "configs" / "environments.json",
               {"environments": {"mlx": {"project": "environments/mlx", "purpose": "language models"}}})
    monkeypatch.setattr(common, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match=r"scripts/bootstrap\.py --repro mlx"):
        common.environment_executable("mlx", clean=True)
    python = tmp_path / "environments/mlx/.venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    assert common.environment_executable("mlx") == python


def test_every_demo_maps_to_a_configured_environment():
    from ml_training.cli import DEMOS, HANDLERS
    from ml_training.common import environments
    config = environments()
    assert set(config["demos"]) == set(DEMOS) == set(HANDLERS)
    assert set(config["demos"].values()) <= set(config["environments"])
