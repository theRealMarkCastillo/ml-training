import sys

import pytest

from ml_training.common import digest, read_json, write_json
from ml_training import run_selection as selection


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


def test_auto_select_binds_newest_consistent_runs(tmp_path,monkeypatch):
    monkeypatch.setattr(selection,"ROOT",tmp_path)
    def create(demo, stamp, **extra):
        path = tmp_path / "artifacts" / demo / f"local-s41-{stamp}-abc123"
        path.mkdir(parents=True)
        (path / "model.bin").write_bytes(demo.encode())
        write_json(path / "run.json",{"demo":demo,"profile":"local","status":"evaluated","seed":41,
                                     "files":{"model.bin":{"sha256":digest(path / "model.bin")}}} | extra)
        return path
    paths = {demo:create(demo,"20260101T000000Z") for demo in selection.DEMOS if demo not in ["rlvr","explainability"]}
    newer_tabular = create("tabular","20260201T000000Z")
    create("explainability","20260101T000000Z",source_run=str(paths["tabular"].relative_to(tmp_path)),
           source_manifest_sha256=digest(paths["tabular"] / "run.json"))
    rlvr = create("rlvr","20260101T000000Z",source_run=str(paths["schema_dsl"]),
                  source_manifest_sha256=digest(paths["schema_dsl"] / "run.json"))
    chosen = selection.auto_select()
    # The explanation's own source wins over a newer, unrelated tabular run.
    assert chosen["tabular"] == str(paths["tabular"].relative_to(tmp_path)) != str(newer_tabular.relative_to(tmp_path))
    assert chosen["rlvr"] == str(rlvr.relative_to(tmp_path))
    assert read_json(tmp_path / "artifacts/validation/selected_cpu_runs.json")["tabular-41"] == chosen["tabular"]
