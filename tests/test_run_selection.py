import sys

import pytest

from ml_training.common import digest, read_json, write_json
from scripts import select_component_runs as selection


def runs(root):
    def create(demo, seed):
        path = root / "artifacts" / demo / f"local-{seed}"
        path.mkdir(parents=True)
        (path / "model.bin").write_bytes(b"fixture weights")
        write_json(path / "run.json",{"demo":demo,"profile":"local","status":"evaluated","seed":seed,
                                     "files":{"model.bin":{"sha256":digest(path / "model.bin")}}})
        return path
    paths = {demo:create(demo,41) for demo in selection.DEMOS}
    for demo,source in [("rlvr","schema_dsl"),("explainability","tabular")]:
        record = read_json(paths[demo] / "run.json")
        write_json(paths[demo] / "run.json",record | {"source_manifest_sha256":digest(paths[source] / "run.json")})
    extra = {key:create(*value) for key,value in selection.EXTRA_CPU.items()}
    return paths,extra


def test_clean_checkout_cli_records_all_three_cpu_seeds(tmp_path,monkeypatch):
    monkeypatch.setattr(selection,"ROOT",tmp_path)
    paths, extra = runs(tmp_path)
    args = ["select_component_runs.py"]
    for key,path in paths.items() | extra.items():
        args.extend(["--"+key.replace("_","-"),str(path)])
    monkeypatch.setattr(sys,"argv",args)
    selection.main()
    cpu = read_json(tmp_path / "artifacts/validation/selected_cpu_runs.json")
    assert set(cpu) == {"tabular-41","fast_classifier-41","explainability-41",*extra}
    assert all((tmp_path / path / "run.json").is_file() for path in cpu.values())


def test_wrong_replication_seed_cannot_partially_replace_selectors(tmp_path,monkeypatch):
    monkeypatch.setattr(selection,"ROOT",tmp_path)
    paths,extra = runs(tmp_path)
    extra["fast_classifier-42"] = extra["fast_classifier-43"]
    with pytest.raises(ValueError,match="seed-42"):
        selection.select_runs(paths,extra)
    assert not (tmp_path / "artifacts/validation").exists()
