"""Score the D02 typed-slot baseline with training wording only, for a recorded run.

The recorded baseline's grammar included the validation and test wording, which the
trained models never saw. The parser is deterministic, so no model runs: this script
first reproduces the recorded typed_slot metrics with all wordings, then scores the
training-wording parser and re-applies the run's validation selection rule. The
run's own files are not modified.
"""

import argparse
from pathlib import Path

from ml_training import schema_data
from ml_training.common import ROOT, digest, read_json, require_reproduced, verify_run, write_json
from ml_training.structured import typed_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", help="Default: the selected D02 run")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/validation/d02-typed-slot-training-wording.json"
    )
    args = parser.parse_args()
    if args.run is None:
        args.run = ROOT / read_json(ROOT / "artifacts/validation/selected_structured_runs.json")["schema_dsl"]["local"]
    run = args.run.resolve()
    verify_run(run)
    info = read_json(run / "run.json")
    if info["demo"] != "schema_dsl" or info.get("typed_slot_wordings"):
        raise SystemExit(f"{run.name} is not a D02 run recorded with the all-wordings baseline")
    rows, challenge = schema_data.load(ROOT / info["dataset_path"])
    parts = {split: [r for r in rows if r["split"] == split] for split in ["validation", "test"]} | {
        "challenge": challenge
    }
    selection = read_json(run / "selection.json")
    recorded = {
        "validation": selection["validation"]["typed_slot"],
        "test": read_json(run / "metrics.json")["typed_slot"],
        "challenge": read_json(run / "challenge.json")["typed_slot"],
    }
    scores = {wordings: {} for wordings in ["all", "training"]}
    for wordings in scores:
        for split, part in parts.items():
            outputs = typed_outputs(part, wordings)
            scores[wordings][split] = schema_data.score(part, outputs) | {
                "parsed": sum(bool(o["text"]) for o in outputs)
            }
    for split, values in recorded.items():
        actual = {k: v for k, v in scores["all"][split].items() if k != "parsed"}
        require_reproduced(actual, values, f"All-wordings typed_slot {split} metrics")
    timings = read_json(run / "benchmarks.json")
    validation = dict(selection["validation"]) | {"typed_slot": scores["training"]["validation"]}
    candidates = [name for name in validation if name in timings]
    # The run's rule; typed_slot's recorded p95 stands in for the training-wording parser.
    reselected = min(candidates, key=lambda name: (-validation[name]["joint_accuracy"], timings[name]["p95_seconds"]))
    write_json(
        args.output,
        {
            "run": str(run.relative_to(ROOT)),
            "source_manifest_sha256": digest(run / "run.json"),
            "method": "Deterministic parser re-scored on the saved dataset; all-wordings scores reproduced the recorded metrics first.",
            "training_styles": list(schema_data.TRAINING_STYLES),
            "metrics": scores,
            "recorded_selection": selection["selected_variant"],
            "selection_with_training_wording_baseline": reselected,
        },
    )
    print(f"{'split':10} {'rows':>4} {'all-wordings acc':>16} {'training-wording acc':>20} {'parsed (training)':>17}")
    for split in parts:
        a, t = scores["all"][split], scores["training"][split]
        print(f"{split:10} {a['rows']:4d} {a['joint_accuracy']:16.4f} {t['joint_accuracy']:20.4f} {t['parsed']:17d}")
    print(f"Recorded selection: {selection['selected_variant']}; with the training-wording baseline: {reselected}")
    print(f"Wrote {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
