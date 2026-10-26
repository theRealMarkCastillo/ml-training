"""Measure D07 response to +/-1% amount edits on fixed held-out rows."""

import argparse
from pathlib import Path

import numpy as np

from ml_training.common import ROOT, digest, read_json, verify_run, write_json
from ml_training.explainability import explain_rows, modify_row, setup

parser = argparse.ArgumentParser()
parser.add_argument("run",type=Path)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--compare",type=Path,help="Require exact reproduction of a previous result")
args = parser.parse_args()
verify_run(args.run)
info = read_json(args.run / "run.json")
source = ROOT / info["source_run"]
assert digest(source / "run.json")==info["source_manifest_sha256"]
_,parts,model,explainer,names,background = setup(source)
rows = parts["test"].head(100)
records,skipped = [],[]
for index in range(rows.height):
    row = rows[index:index+1]
    amount = row["amount"][0]
    if amount is None:
        skipped.append(int(row["event_id"][0]))
        continue  # Missingness is not a numeric amount; do not silently impute an edit.
    before,bv = explain_rows(model,explainer,names,row)
    for factor in [.99,1.01]:
        changed = modify_row(row,{"amount":float(amount)*factor})
        after,av = explain_rows(model,explainer,names,changed)
        probability_change = after[0]["calibrated_probability"]-before[0]["calibrated_probability"]
        attribution_change = av[0]-bv[0]
        margin_change = after[0]["raw_log_odds"]-before[0]["raw_log_odds"]
        delta_residual = float(abs(attribution_change.sum()-margin_change))
        assert delta_residual <= 2e-5
        records.append({"event_id":int(row["event_id"][0]),"factor":factor,"amount_before":float(amount),
                        "amount_after":float(amount)*factor,"before":before[0],"after":after[0],
                        "calibrated_probability_change":probability_change,
                        "attribution_l1_change":float(np.abs(attribution_change).sum()),
                        "top_absolute_feature_changed":int(np.argmax(np.abs(av[0])))!=int(np.argmax(np.abs(bv[0]))),
                        "margin_delta_reconstruction_error":delta_residual})
assert len(records)==2*(100-len(skipped))
changes = np.abs([r["calibrated_probability_change"] for r in records])
l1 = np.array([r["attribution_l1_change"] for r in records])
result = {"run":str(args.run),"source_manifest_sha256":digest(source / "run.json"),
          "selection":"First 100 held-out rows, selected before checking their response to edits; no labels used",
          "rows_considered":100,"rows_measured":100-len(skipped),"missing_amount_rows_skipped":skipped,
          "numeric_edits":len(records),"factors":[.99,1.01],"background_event_ids":background,
          "summary":{"probability_unchanged_fraction":float(np.mean(changes==0)),
                     "absolute_probability_change_median":float(np.median(changes)),
                     "absolute_probability_change_p95":float(np.percentile(changes,95)),
                     "absolute_probability_change_max":float(changes.max()),
                     "attribution_l1_median":float(np.median(l1)),"attribution_l1_p95":float(np.percentile(l1,95)),
                     "attribution_l1_max":float(l1.max()),
                     "top_feature_change_fraction":float(np.mean([r["top_absolute_feature_changed"] for r in records])),
                     "max_margin_delta_reconstruction_error":max(r["margin_delta_reconstruction_error"] for r in records)},
          "records":records,"scope":"Conditional amount edits; history and other inputs fixed. Measures local stability of raw-log-odds attributions and calibrated predictions, not causal effects or general robustness."}
if args.compare:
    assert result==read_json(args.compare),"Stability results did not reproduce exactly"
write_json(args.output,result)
print({"status":"reproduced" if args.compare else "measured",**result["summary"]})
