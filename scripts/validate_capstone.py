"""Run the full integration comparison, then replay in separately installed environments."""

import argparse
from pathlib import Path
import subprocess
import time

from ml_training.common import ROOT, environment_executable, read_json, verify_run, write_json
from ml_training.hybrid import run as run_capstone

parser = argparse.ArgumentParser()
parser.add_argument("--rlvr-run",type=Path)
parser.add_argument("--run",type=Path,help="Replay an already completed capstone instead of rerunning its experiment")
args = parser.parse_args()
if args.run:
    run = args.run.resolve()
else:
    if not args.rlvr_run:
        parser.error("Supply --rlvr-run for a new comparison, or --run for an existing one")
    run = run_capstone(args.rlvr_run,profile="local",seed=41)
verify_run(run)
write_json(ROOT / "artifacts/validation/selected_hybrid_run.json",{"run":str(run.relative_to(ROOT)),"status":"evaluated","reproduction_pending":True})
destination = ROOT / "artifacts/validation" / f"hybrid-replay-{run.name}"
started = time.perf_counter()
subprocess.run([str(environment_executable("cpu", "ml-demo", clean=True)),"evaluate","hybrid","--run-dir",str(run),
                "--output-dir",str(destination),"--clean-workers"],check=True,cwd=ROOT)
result = read_json(destination / "result.json")
write_json(ROOT / "artifacts/validation/hybrid.json",{"result":result,"elapsed_seconds":time.perf_counter()-started,
    "replay_directory":str(destination.relative_to(ROOT)),"environment_note":"Root .venv-repro and separate MLX, retrieval and RLVR .venv-repro workers, installed from pinned locks"})
write_json(ROOT / "artifacts/validation/selected_hybrid_run.json",{"run":str(run.relative_to(ROOT)),"status":"evaluated","reproduction_pending":False})
print(f"Capstone comparison and clean replay passed: {run}",flush=True)
