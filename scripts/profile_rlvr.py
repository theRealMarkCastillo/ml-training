"""Measure D03 warm batch-1 latency and fresh-process startup separately."""

import argparse
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import torch

from ml_training import rlvr_models as models
from ml_training import rlvr_data
from ml_training.common import ROOT, environment, read_json, verify_run, write_json

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--cold-variant", choices=["base", "sft", "rlvr", "sft_control"])
args = parser.parse_args()
torch.set_num_threads(4)
launched = time.perf_counter()
verify_run(args.run)
info = read_json(args.run / "run.json")
rows, _ = rlvr_data.load(ROOT / info["dataset_path"])
validation = [row for row in rows if row["split"] == "validation"][:12]
assert len(validation) == 12


def load(variant):
    source = models.original_path() if variant == "base" else args.run / "shared_sft"
    adapter = (
        args.run / "rl_adapter"
        if variant == "rlvr"
        else args.run / "sft_control_adapter"
        if variant == "sft_control"
        else None
    )
    return models.load(source, "cpu", adapter=adapter)


if args.cold_variant:
    model, tokenizer = load(args.cold_variant)
    output = models.generate(model, tokenizer, [validation[0]])[0][0]
    write_json(
        args.output,
        {
            "variant": args.cold_variant,
            "seconds_after_imports": time.perf_counter() - launched,
            "process_peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "actual_output": output,
        },
    )
else:
    results = {}
    for variant in ["base", "sft", "rlvr", "sft_control"]:
        model, tokenizer = load(variant)
        models.generate(model, tokenizer, [validation[0]])  # One excluded warmup.
        observations = []
        for row in validation:
            start = time.perf_counter()
            output = models.generate(model, tokenizer, [row])[0][0]
            observations.append({"id": row["id"], "seconds": time.perf_counter() - start, **output})
        times = [row["seconds"] for row in observations]
        results[variant] = {
            "p50_seconds": float(np.percentile(times, 50)),
            "p95_seconds": float(np.percentile(times, 95)),
            "p99_seconds": float(np.percentile(times, 99)),
            "samples": 12,
            "warmups": 1,
            "batch_size": 1,
            "concurrency": 1,
            "cpu_threads": 4,
            "device": "cpu",
            "outputs": observations,
        }
        del model, tokenizer
        models.release()
        cold_path = args.output.parent / "cold_rlvr" / f"{variant}.json"
        start = time.perf_counter()
        subprocess.run(
            [sys.executable, __file__, str(args.run), "--output", str(cold_path), "--cold-variant", variant],
            check=True,
            timeout=180,
        )
        results[variant]["cold"] = {
            **read_json(cold_path),
            "process_launch_to_exit_seconds": time.perf_counter() - start,
            "samples": 1,
            "scope": "Fresh interpreter, imports, recursive artifact checksums, model load, one generation; OS file cache uncontrolled",
        }
        write_json(
            args.output,
            {
                "environment": environment(),
                "run": str(args.run),
                "variants": results,
                "warm_scope": "Tokenizer, prompt processing and greedy generation; excludes model load and semantic verifier. Twelve samples do not establish precise tail latency.",
            },
        )
        print(f"Profiled D03 {variant}", flush=True)
