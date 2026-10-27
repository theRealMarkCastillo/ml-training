"""Re-measure fresh-process ONNX startup for the selected D06 runs.

Recorded D06 runs timed a child process that imported the training module, so their
fresh-process figures include scikit-learn, skl2onnx and polars imports. This script
times the minimal scorer used now and, for comparison, the old import path, without
modifying any run. Run from the root environment with no competing model work.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from ml_training.common import ROOT, cold_process, digest, environment, read_json, verify_run, write_json

TRAINING_IMPORTS = (
    "import sys, joblib, json; import ml_training.fast_classifier as f; "
    "x = joblib.load(sys.argv[2]); print(json.dumps(f.probabilities(f.session(sys.argv[1]), x).tolist()))"
)


def old_path(model, inputs):
    start = time.perf_counter()
    child = subprocess.run(
        [sys.executable, "-c", TRAINING_IMPORTS, str(model), str(inputs)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return time.perf_counter() - start, json.loads(child.stdout)


def summary(values):
    return {
        "median_seconds": float(np.median(values)),
        "min_seconds": float(min(values)),
        "max_seconds": float(max(values)),
        "samples": len(values),
        "values": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/validation/onnx-fresh-process.json")
    args = parser.parse_args()
    cpu = read_json(ROOT / "artifacts/validation/selected_cpu_runs.json")
    results = []
    for key in sorted(k for k in cpu if k.startswith("fast_classifier-")):
        run = (ROOT / cpu[key]).resolve()
        verify_run(run)
        model, inputs = run / "model.onnx", run / "benchmark_input.joblib"
        minimal, legacy = [], []
        for _ in range(args.samples):  # Interleaved so drift affects both paths alike.
            fresh = cold_process(model, inputs, "onnx")
            minimal.append(fresh["process_start_to_exit_seconds"])
            seconds, prediction = old_path(model, inputs)
            legacy.append(seconds)
            if prediction != fresh["prediction"]:
                raise AssertionError("Old and minimal scorers disagree")
        recorded = read_json(run / "benchmarks.json")["fresh_process_first_inference"]["process_start_to_exit_seconds"]
        results.append(
            {
                "run": str(run.relative_to(ROOT)),
                "source_manifest_sha256": digest(run / "run.json"),
                "recorded_seconds": recorded,
                "minimal_scorer": summary(minimal),
                "with_training_imports": summary(legacy),
            }
        )
        print(
            f"{key}: recorded {recorded:.3f}s | minimal median {np.median(minimal):.3f}s | "
            f"training imports median {np.median(legacy):.3f}s"
        )
    write_json(
        args.output,
        {
            "scope": "Fresh Python process, imports, ONNX session creation, input load and first inference; OS file cache uncontrolled",
            "minimal_scorer": "python -m ml_training.cold_infer onnx (onnxruntime, numpy, joblib)",
            "with_training_imports": "Same work after importing ml_training.fast_classifier, as recorded runs did",
            "environment": environment(),
            "runs": results,
        },
    )
    print(f"Wrote {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
