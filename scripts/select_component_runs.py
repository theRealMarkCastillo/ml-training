"""Record explicit hybrid components and optional additional CPU replication runs."""

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
