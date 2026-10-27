"""Re-grade a recorded D03 run's saved outputs with the v2 hidden fixtures.

No model runs: the saved greedy and four-sample outputs are verified again. The
script first rebuilds the recorded (v1) metrics exactly, proving the re-scoring is
faithful, then grades the same outputs on fixtures disjoint from the public reward.
The run's own files are not modified.
"""

import argparse
from pathlib import Path

from ml_training import rlvr_data
from ml_training.common import ROOT, digest, read_json, require_reproduced, verify_run, write_json


def rescore(run, version):
    info = read_json(run / "run.json")
    rows, challenge = rlvr_data.load(ROOT / info["dataset_path"])
    splits = {"test": [r for r in rows if r["split"] == "test"], "challenge": challenge}
    predictions = read_json(run / "predictions.json")
    results = {"test": {}, "challenge": {}}
    for variant, modes_by_split in predictions.items():
        for split, part in splits.items():
            modes = modes_by_split[split]
            groups = {mode: [r["outputs"] for r in modes[mode]] for mode in ["greedy", "sampled"]}
            if [r["id"] for r in modes["sampled"]] != [r["id"] for r in part]:
                raise ValueError(f"Saved {variant}/{split} outputs do not align with the dataset")
            greedy, _ = rlvr_data.group_metrics(part, groups["greedy"], version)
            sampled, _ = rlvr_data.group_metrics(part, groups["sampled"], version)
            results[split][variant] = {
                "greedy_pass_at_1": greedy["sampled_pass_at_1"],
                **sampled,
                "greedy_mean_generation_tokens": greedy["mean_generation_tokens"],
            }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", help="Default: the selected D03 run")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/validation/rlvr-hidden-v2.json")
    args = parser.parse_args()
    run = (args.run or ROOT / read_json(ROOT / "artifacts/validation/selected_rlvr_run.json")["run"]).resolve()
    verify_run(run)
    recorded_version = read_json(run / "run.json").get("hidden_fixtures", "v1")
    if recorded_version != "v1":
        raise SystemExit(f"{run.name} was already graded with {recorded_version}; nothing to re-score")
    v1 = rescore(run, "v1")
    require_reproduced(v1["test"], read_json(run / "metrics.json"), "Re-scored v1 test metrics")
    require_reproduced(v1["challenge"], read_json(run / "challenge.json"), "Re-scored v1 challenge metrics")
    v2 = rescore(run, "v2")
    write_json(
        args.output,
        {
            "run": str(run.relative_to(ROOT)),
            "source_manifest_sha256": digest(run / "run.json"),
            "method": "Saved outputs re-verified; no generation. v1 rebuilt the recorded metrics exactly before v2 grading.",
            "v1": "tables(hidden=True): one new random table; boundary-witness and empty tables shared with the reward",
            "v2": "hidden_tables(): three random tables and offset boundary witnesses, none shared with the reward",
            "metrics": {"v1": v1, "v2": v2},
        },
    )
    print(
        f"{'variant':12} {'split':9} {'greedy v1':>9} {'greedy v2':>9} {'sel@4 v1':>8} {'sel@4 v2':>8} {'exploit v1':>10} {'exploit v2':>10}"
    )
    for split in ["test", "challenge"]:
        for variant in v1[split]:
            a, b = v1[split][variant], v2[split][variant]
            print(
                f"{variant:12} {split:9} {a['greedy_pass_at_1']:9.4f} {b['greedy_pass_at_1']:9.4f} "
                f"{a['verifier_selected_success_at_n']:8.4f} {b['verifier_selected_success_at_n']:8.4f} "
                f"{a['public_pass_hidden_failure_rate']:10.4f} {b['public_pass_hidden_failure_rate']:10.4f}"
            )
    print(f"Wrote {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
