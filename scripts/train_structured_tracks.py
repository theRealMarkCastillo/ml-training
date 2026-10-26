"""Sequential D02/D04 smoke checks and actual local experiments."""

from pathlib import Path
from ml_training.common import ROOT, read_json, verify_run, write_json
from ml_training.structured import train

ledger = ROOT / "artifacts/validation/selected_structured_runs.json"
completed = read_json(ledger) if ledger.exists() else {}
for demo in ["schema_dsl", "edge"]:
    for profile in ["smoke", "local"]:
        prior = completed.get(demo, {}).get(profile)
        if prior:
            verify_run(Path(prior))
            continue
        run = train(demo, profile, 41)
        completed.setdefault(demo, {})[profile] = str(run)
        write_json(ledger, completed)
print(completed)
