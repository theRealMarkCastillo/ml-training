"""Export small, traceable teaching data from the selected evaluated local runs.

No model loading, training, or selection occurs here. Run from the repository root.
The exporter is deterministic for unchanged input files; large/raw artifacts stay local.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export_snapshot(selection_dir):
    sources = {}

    def relative(path):
        # Selectors and profiling CLIs accept either absolute or relative paths.
        # Compare their canonical repository-relative identities, not spelling.
        return (ROOT / path).resolve().relative_to(ROOT.resolve()).as_posix()

    def read(path):
        # Keep the portable export free of host-specific absolute paths.
        name = relative(path)
        content = (ROOT / name).read_bytes()
        sources[name] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    def selected(name):
        return read(selection_dir / name)

    def correction(name, run_path):
        """Supplemental re-analysis of a selected run; absent for exports without it."""
        if not (selection_dir / name).is_file():
            return None
        value = read(selection_dir / name)
        if relative(value["run"]) != relative(run_path):
            raise ValueError(f"{name} does not describe the selected run {relative(run_path)}")
        return value

    def run(path, files):
        path = Path(path)
        manifest = read(path / "run.json")
        if manifest["status"] != "evaluated":
            raise ValueError(f"Requires evaluated run: {path}")
        return {
            "run": relative(path),
            "recorded_at": manifest["started_at"],
            "seed": manifest["seed"],
            "environment": {k: manifest["environment"].get(k) for k in ["python", "platform", "architecture"]},
            **{f.removesuffix(".json"): read(path / f) for f in files},
        }

    demos = {}
    p = selected("selected_distillation_run.json")["run"]
    demos["D01"] = run(p, ["metrics.json", "challenge.json", "costs.json"])
    outputs = Path(p) / "teacher_curation/outputs.jsonl"
    if (ROOT / outputs).is_file():
        content = (ROOT / outputs).read_bytes()
        sources[relative(outputs)] = hashlib.sha256(content).hexdigest()
        records = [json.loads(line) for line in content.decode().splitlines()]
        # The teacher answer is kept only if it equals the oracle label, so curation
        # filters examples; it never changes a training target.
        demos["D01"]["curation"] = {
            "examples": len(records),
            "accepted": sum(r["accepted"] for r in records),
            "accepted_equal_oracle": all(r["parsed"] == r["expected"] for r in records if r["accepted"]),
            "rejected_by_template": dict(Counter(r["id"].rsplit("-", 1)[0] for r in records if not r["accepted"])),
        }
    structured = selected("selected_structured_runs.json")
    demos["D02"] = run(structured["schema_dsl"]["local"], ["metrics.json", "challenge.json"])
    if rescored := correction("d02-typed-slot-training-wording.json", structured["schema_dsl"]["local"]):
        demos["D02"]["training_wording_baseline"] = {
            "metrics": {split: rescored["metrics"]["training"][split] for split in ["validation", "test", "challenge"]},
            "recorded_selection": rescored["recorded_selection"],
            "selection_with_training_wording_baseline": rescored["selection_with_training_wording_baseline"],
        }
    p = selected("selected_rlvr_run.json")["run"]
    demos["D03"] = run(p, ["metrics.json", "challenge.json"])
    training = read(Path(p) / "grpo_training.json")
    demos["D03"]["training"] = {
        k: training[k]
        for k in [
            "optimizer_steps",
            "nonconstant_reward_groups",
            "constant_reward_groups",
            "parameters_changed",
            "elapsed_training_seconds",
        ]
    }
    if rescored := correction("rlvr-hidden-v2.json", p):
        demos["D03"]["hidden_v2"] = rescored["metrics"]["v2"]
    p = structured["edge"]["local"]
    demos["D04"] = run(p, ["metrics.json", "challenge.json", "selection.json", "quantization.json"])
    predictions = read(Path(p) / "predictions.json")
    demos["D04"]["rejection_mutations"] = {}
    for variant in demos["D04"]["metrics"]:
        rows = [r for r in predictions if r["variant"] == variant and r["expected"]["action"] == "reject"]
        demos["D04"]["rejection_mutations"][variant] = {
            "requests": len(rows),
            "mutations": sum(r["runtime"]["before"] != r["runtime"]["after"] for r in rows),
        }
    cpu = selected("selected_cpu_runs.json")
    demos["D05"] = {
        "runs": [
            run(path, ["metrics.json", "selection.json"]) for name, path in cpu.items() if name.startswith("tabular-")
        ]
    }
    demos["D06"] = {
        "runs": [run(path, ["benchmarks.json"]) for name, path in cpu.items() if name.startswith("fast_classifier-")]
    }
    profile = read(selection_dir / "classifier-batches.json")
    for row in profile["runs"]:
        row["run"] = relative(row["run"])
    demos["D06"]["paired_profiles"] = profile["runs"]
    # Do not mix profiles from a previous model with newly selected runs.
    if {r["run"] for r in profile["runs"]} != {r["run"] for r in demos["D06"]["runs"]}:
        raise ValueError("Paired classifier profiles do not match selected runs")
    for row in profile["runs"]:
        if row["source_manifest_sha256"] != sources[row["run"] + "/run.json"]:
            raise ValueError("Paired classifier source manifest changed")
    if (selection_dir / "onnx-fresh-process.json").is_file():
        startup = read(selection_dir / "onnx-fresh-process.json")
        if {relative(r["run"]) for r in startup["runs"]} != {r["run"] for r in demos["D06"]["runs"]}:
            raise ValueError("Fresh-process profile does not match selected runs")
        demos["D06"]["fresh_process_remeasured"] = [
            {
                "run": relative(r["run"]),
                **{k: r[k]["median_seconds"] for k in ["minimal_scorer", "with_training_imports"]},
            }
            for r in startup["runs"]
        ]
    demos["D07"] = run(cpu["explainability-41"], ["feature_edit.json", "correlated_features.json"])
    p = selected("selected_retrieval_run.json")["run"]
    demos["D08"] = run(p, ["metrics.json"])
    # All held-out ranks, not a cherry-picked success/failure subset.
    demos["D08"]["ranks"] = [
        {k: row[k] for k in ["query_id", "variant", "rank"]} for row in read(Path(p) / "predictions.json")
    ]
    p = selected("selected_hybrid_run.json")["run"]
    demos["D09"] = run(p, ["metrics.json"])
    for metrics in demos["D09"]["metrics"].values():
        metrics.pop("worker_peak_rss_bytes_max_by_component", None)
    predictions = read(Path(p) / "predictions.json")
    demos["D09"]["outcomes"] = {}
    for variant in demos["D09"]["metrics"]:
        rows = [r for r in predictions if r["variant"] == variant]
        total = Counter(r["grade"]["expected"]["outcome"] for r in rows)
        correct = Counter(r["grade"]["expected"]["outcome"] for r in rows if r["grade"]["correct"])
        demos["D09"]["outcomes"][variant] = {k: {"total": v, "correct": correct[k]} for k, v in total.items()}
    analysis = read(selection_dir / "hybrid-analysis.json")
    if relative(analysis["run"]) != relative(p):
        raise ValueError("Hybrid analysis does not match selected run")
    demos["D09"]["common_risk_cohort"] = analysis["common_risk_cohort"]
    return {
        "schema_version": 1,
        "evidence_kind": "recorded_local_experiments",
        "scope": "Recorded local educational experiments. Inspect run dates, seeds and source "
        "manifests before comparing results. No production or physical-device validation.",
        "sources": sources,
        "demos": demos,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, default=ROOT / "artifacts/validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = export_snapshot(args.selection_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    print(f"Exported {len(snapshot['demos'])} lessons from {len(snapshot['sources'])} source files to {args.output}")
