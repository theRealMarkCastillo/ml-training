"""Fresh-environment RLVR checkpoint, greedy/sample-output, and metric replay."""

import argparse
from pathlib import Path
import time

from ml_training.common import environment, write_json
from ml_training.rlvr import reevaluate

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="auto")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
start = time.perf_counter()
result = reevaluate(args.run, args.device)
write_json(
    args.output, {"environment": environment(), "result": result, "elapsed_seconds": time.perf_counter() - start}
)
print(args.output)
