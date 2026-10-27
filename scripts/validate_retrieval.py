"""Clean-environment checkpoint, ranking and cold-inference reproduction."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from ml_training.common import ROOT, digest, environment, write_json
from ml_training.retrieval import reevaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    args = parser.parse_args()
    run = args.run.resolve()
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, text=True, capture_output=True)
    if tests.returncode:
        print(tests.stdout)
        print(tests.stderr)
        raise SystemExit(tests.returncode)
    result = {
        "python_executable": sys.executable,
        "environment": environment(),
        "tests": {"returncode": tests.returncode, "stdout": tests.stdout},
        "reevaluation": reevaluate(run, args.device),
        "source_manifest_sha256": digest(run / "run.json"),
        "fresh_process": {},
    }
    for kind in ["bi_encoder", "cross_encoder"]:
        start = time.perf_counter()
        child = subprocess.run(
            [sys.executable, "-m", "ml_training.retrieval_cold", str(run), kind, args.device],
            text=True,
            capture_output=True,
            check=True,
            timeout=60,
        )
        result["fresh_process"][kind] = {
            "seconds": time.perf_counter() - start,
            "output": json.loads(child.stdout),
            "scope": "Fresh Python process, imports, checkpoint/data load and first inference; OS cache uncontrolled",
            "samples": 1,
        }
    path = ROOT / "artifacts" / "validation" / "retrieval.json"
    write_json(path, result)
    print(tests.stdout.strip())
    print(json.dumps(result["reevaluation"], indent=2))
    print(f"Evidence: {path}")


if __name__ == "__main__":
    main()
