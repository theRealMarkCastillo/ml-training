"""Reproduce selected SLM experiments in a separately installed environment."""

import argparse
from pathlib import Path
import time
import subprocess
import sys

from ml_training.common import ROOT, environment, read_json, write_json

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
results = []
for run in args.runs:
    start = time.perf_counter()
    demo = read_json(run / "run.json")["demo"]
    if demo == "distillation":
        from ml_training.distillation import reevaluate
    elif demo in ["schema_dsl", "edge"]:
        from ml_training.structured import reevaluate
    else:
        raise ValueError(f"Unexpected SLM demo: {demo}")
    reproduction = reevaluate(run) | {"elapsed_seconds": time.perf_counter()-start}
    cold = {}
    for variant in read_json(run / "metrics.json"):
        output = args.output.parent / "cold_slm" / run.name / f"{variant}.json"
        launched = time.perf_counter()
        subprocess.run([sys.executable, str(ROOT / "scripts/cold_slm.py"), str(run), variant, str(output)],
                       check=True, timeout=180, cwd=ROOT)
        cold[variant] = read_json(output) | {"process_launch_to_exit_seconds": time.perf_counter()-launched}
    results.append(reproduction | {"cold_processes": cold})
    write_json(args.output, {"environment": environment(), "completed_runs": results})
    print(f"Reproduced {demo}: {run}", flush=True)
