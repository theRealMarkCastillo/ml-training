"""Record which component runs the D09 capstone binds, explicitly or automatically.

The capstone reads these selectors from artifacts/validation/. Use the
command-line form to name every run, or `ml-demo prepare hybrid --auto-select`
to bind the latest consistent evaluated runs in this checkout.
"""

import argparse
from pathlib import Path

from ml_training.common import ROOT, digest, read_json, verify_run, write_json

DEMOS = ["distillation","schema_dsl","rlvr","edge","tabular","fast_classifier","explainability","retrieval"]
EXTRA_CPU = {f"{demo}-{seed}":(demo,seed) for demo in ["tabular","fast_classifier"] for seed in [42,43]}


def select_runs(paths, additional_cpu=None):
    paths = {demo:Path(path).resolve() for demo,path in paths.items()}
    if set(paths) != set(DEMOS):
        raise ValueError("Select all eight demo components")
    additional_cpu = {key:Path(path).resolve() for key,path in (additional_cpu or {}).items()}
    if set(additional_cpu) - set(EXTRA_CPU):
        raise ValueError("Additional CPU runs must be tabular/classifier seeds 42 or 43")
    def checked(path, demo, seed):
        verify_run(path)
        record = read_json(path / "run.json")
        if record["demo"]!=demo or record["profile"]!="local" or record["status"]!="evaluated" or record["seed"]!=seed:
            raise ValueError(f"Expected an evaluated local seed-{seed} {demo} run: {path}")
        path.relative_to(ROOT)
        return record
    info = {demo:checked(path,demo,41) for demo,path in paths.items()}
    for key,path in additional_cpu.items():
        checked(path,*EXTRA_CPU[key])
    if info["rlvr"]["source_manifest_sha256"]!=digest(paths["schema_dsl"] / "run.json"):
        raise ValueError("RLVR must have started from the selected schema/DSL checkpoint")
    if info["explainability"]["source_manifest_sha256"]!=digest(paths["tabular"] / "run.json"):
        raise ValueError("Explanation source must match the selected tabular run")
    destination = ROOT / "artifacts/validation"
    cpu_path, structured_path = destination / "selected_cpu_runs.json", destination / "selected_structured_runs.json"
    cpu = read_json(cpu_path) if cpu_path.exists() else {}
    structured = read_json(structured_path) if structured_path.exists() else {}
    for demo in ["tabular","fast_classifier","explainability"]:
        cpu[f"{demo}-41"] = str(paths[demo].relative_to(ROOT))
    cpu.update({key:str(path.relative_to(ROOT)) for key,path in additional_cpu.items()})
    for demo in ["schema_dsl","edge"]:
        structured.setdefault(demo,{})["local"] = str(paths[demo].relative_to(ROOT))
    for demo in ["distillation","retrieval","rlvr"]:
        write_json(destination / f"selected_{demo}_run.json",{"run":str(paths[demo].relative_to(ROOT))})
    write_json(cpu_path,cpu)
    write_json(structured_path,structured)


def latest_runs(demo, seed=41):
    """Evaluated local runs for one demo, newest first (run names embed a UTC timestamp)."""
    found = []
    for path in sorted((ROOT / "artifacts" / demo).glob(f"local-s{seed}-*"), key=lambda p: p.name.split("-")[2], reverse=True):
        try:
            record = read_json(path / "run.json")
        except (OSError, ValueError):
            continue
        if record.get("status") == "evaluated" and record.get("seed") == seed:
            found.append(path)
    return found


def auto_select(rlvr_run=None):
    """Choose the newest evaluated seed-41 run per component, keeping D02->D03 and D05->D07 consistent."""
    def newest(demo, accept=lambda path: True):
        for path in latest_runs(demo):
            if accept(path):
                return path
        raise FileNotFoundError(f"No evaluated local seed-41 {demo} run satisfies the capstone's requirements. "
                                f"Train it first; see docs/REPRODUCTION.md for the dependency order.")
    paths = {}
    paths["rlvr"] = Path(rlvr_run).resolve() if rlvr_run else newest("rlvr")
    source = Path(read_json(paths["rlvr"] / "run.json")["source_run"])
    paths["schema_dsl"] = (source if source.is_absolute() else ROOT / source).resolve()
    paths["explainability"] = newest("explainability", lambda path: (ROOT / read_json(path / "run.json")["source_run"]).is_dir())
    paths["tabular"] = (ROOT / read_json(paths["explainability"] / "run.json")["source_run"]).resolve()
    for demo in ["distillation", "edge", "fast_classifier", "retrieval"]:
        paths[demo] = newest(demo)
    additional = {}
    for key, (demo, seed) in EXTRA_CPU.items():
        if candidates := latest_runs(demo, seed):
            additional[key] = candidates[0]
    select_runs(paths, additional)
    return {demo: str(path.relative_to(ROOT)) for demo, path in {**paths, **additional}.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for demo in DEMOS:
        parser.add_argument("--"+demo.replace("_","-"),type=Path,required=True)
    for key in EXTRA_CPU:
        parser.add_argument("--"+key.replace("_","-"),type=Path,
                            help="Include for the complete three-seed final audit")
    args = parser.parse_args()
    paths = {demo:getattr(args,demo) for demo in DEMOS}
    extra = {key:getattr(args,key.replace("-","_")) for key in EXTRA_CPU}
    select_runs(paths,{key:path for key,path in extra.items() if path is not None})
    print("Selected and verified all explicit component and CPU replication runs")


if __name__ == "__main__":
    main()
