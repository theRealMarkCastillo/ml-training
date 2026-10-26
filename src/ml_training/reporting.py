"""Generate an evidence index without turning missing runs into passes."""

from ml_training.common import ROOT, read_json, verify_run, write_json


def report_all():
    runs = []
    for path in sorted((ROOT / "artifacts").glob("*/*/run.json")):
        manifest = read_json(path)
        record = {"demo": manifest["demo"], "profile": manifest["profile"],
                  "seed": manifest["seed"], "status": manifest["status"],
                  "path": str(path.parent.relative_to(ROOT))}
        if manifest["status"] in ["smoke", "evaluated"]:
            verify_run(path.parent)
            record["checksums_verified"] = True
            metrics = path.parent / "metrics.json"
            if metrics.exists():
                record["test_metrics"] = read_json(metrics)
        runs.append(record)
    write_json(ROOT / "artifacts" / "index.json", runs)
    lines = ["# ML Training Demo Lab evidence index", "",
             "Smoke runs validate wiring only. Evaluated runs have real artifacts; category completion still requires the plan's full acceptance criteria.", "",
             "| Demo | Profile | Seed | Evidence status | Run |", "| --- | --- | --- | --- | --- |"]
    for run in runs:
        lines.append(f"| {run['demo']} | {run['profile']} | {run['seed']} | {run['status']} | [{run['path']}]({ROOT / run['path'] / 'run.json'}) |")
    (ROOT / "artifacts" / "report.md").write_text("\n".join(lines) + "\n")
    return ROOT / "artifacts" / "report.md"
