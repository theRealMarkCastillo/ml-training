"""Post-evaluation descriptive slices; never selects or changes a checkpoint."""

import argparse
from pathlib import Path

from ml_training.common import ROOT, read_json, verify_run, write_json

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
verify_run(args.run)
info = read_json(args.run / "run.json")
rows = read_json(ROOT / info["dataset_path"] / "rows.json")
novel = {row["id"] for row in rows if row["split"] == "test" and row["novel_composition"]}
assert len(novel) == 24
predictions = read_json(args.run / "predictions.json")
result = {}
for variant, splits in predictions.items():
    summary = {}
    for split, modes in splits.items():
        for mode, records in modes.items():
            all_pairs = [
                (public, hidden)
                for row in records
                for public, hidden in zip(row["public_checks"], row["hidden_checks"], strict=True)
            ]
            chosen = [
                (r["public_checks"][r["selected_index"]], r["hidden_checks"][r["selected_index"]]) for r in records
            ]
            subset = [r for r in records if r["id"] in novel] if split == "test" else []
            summary[f"{split}_{mode}"] = {
                "requests": len(records),
                "generated_candidates": len(all_pairs),
                "candidate_schema_valid": sum(h["valid"] for p, h in all_pairs),
                "candidate_public_pass_hidden_failure": sum(p["passed"] and not h["passed"] for p, h in all_pairs),
                "selected_public_pass_hidden_failure": sum(p["passed"] and not h["passed"] for p, h in chosen),
                "novel_requests": len(subset),
                "novel_selected_hidden_pass": sum(r["hidden_checks"][r["selected_index"]]["passed"] for r in subset),
            }
    result[variant] = summary
sft, rl = predictions["sft"]["test"]["greedy"], predictions["rlvr"]["test"]["greedy"]
paired = {
    "same_raw_text": sum(a["outputs"][0]["text"] == b["outputs"][0]["text"] for a, b in zip(sft, rl, strict=True)),
    "sft_only_pass": sum(
        a["hidden_checks"][0]["passed"] and not b["hidden_checks"][0]["passed"] for a, b in zip(sft, rl, strict=True)
    ),
    "rl_only_pass": sum(
        b["hidden_checks"][0]["passed"] and not a["hidden_checks"][0]["passed"] for a, b in zip(sft, rl, strict=True)
    ),
}
write_json(
    args.output,
    {
        "run": str(args.run),
        "slices": result,
        "paired_sft_rl_greedy_test": paired,
        "scope": "Descriptive final-test analysis only. No checkpoint, prompt or threshold was changed.",
    },
)
print(args.output)
