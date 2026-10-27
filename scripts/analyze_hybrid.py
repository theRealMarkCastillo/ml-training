"""Read final capstone traces for aligned risk cohorts and branch/failure analysis."""

import argparse
from pathlib import Path

from ml_training.common import read_json, verify_run, write_json

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
verify_run(args.run)
rows = read_json(args.run / "predictions.json")
variants = read_json(args.run / "run.json")["variants"]
risk = {
    variant: {
        row["id"]: {"label": row["risk_label"], "probability": stage["output"]["probability"]}
        for row in rows
        if row["variant"] == variant
        for stage in row["trace"]
        if stage["stage"] == "risk"
    }
    for variant in variants
}
common = set.intersection(*(set(records) for records in risk.values()))
aligned = {
    variant: {
        "rows": len(common),
        "positive_labels": sum(records[key]["label"] for key in common),
        "brier": sum((records[key]["probability"] - records[key]["label"]) ** 2 for key in common) / len(common),
    }
    for variant, records in risk.items()
}
summary = {}
for variant in variants:
    selected = [r for r in rows if r["variant"] == variant]
    categories = sorted({r["grade"]["expected"]["outcome"] for r in selected})
    summary[variant] = {
        "by_expected_outcome": {
            category: {
                "requests": sum(r["grade"]["expected"]["outcome"] == category for r in selected),
                "correct": sum(
                    r["grade"]["correct"] and r["grade"]["expected"]["outcome"] == category for r in selected
                ),
            }
            for category in categories
        },
        "stages": sorted({s["stage"] for r in selected for s in r["trace"]}),
        "failed_requests": [
            {
                "id": r["id"],
                "expected": r["grade"]["expected"],
                "observed": r["outcome"],
                "result": r["result"],
                "reason": r["reason"],
                "fallbacks": r["fallbacks"],
            }
            for r in selected
            if not r["grade"]["correct"]
        ],
    }
write_json(
    args.output,
    {
        "run": str(args.run),
        "variant_analysis": summary,
        "common_risk_cohort": {
            "ids": sorted(common),
            "variants": aligned,
            "limitation": "Screening/authorization changes which rows receive risk scores. This intersection aligns comparisons; zero or very few positive labels cannot establish useful risk discrimination.",
        },
        "scope": "Post-evaluation description only; no model, request, routing threshold or outcome was changed.",
    },
)
print(args.output)
