"""Run under an OS network-denial policy; perform actual fresh model inference."""

import argparse
import errno
import os
from pathlib import Path
import resource
import socket
import time

started = time.perf_counter()
parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--variant", choices=["base", "tuned", "quantized_tuned"], default="quantized_tuned")
args = parser.parse_args()

# A refused connection would only prove that the port is closed. Require the OS
# to reject socket creation/connect, even to loopback, with a permission error.
denial = None
try:
    with socket.socket() as connection:
        connection.settimeout(2)
        connection.connect(("127.0.0.1", 9))
except OSError as error:
    if error.errno not in [errno.EPERM, errno.EACCES]:
        raise AssertionError(f"Networking is not proven denied: {error}") from error
    denial = {"errno": error.errno, "message": str(error), "probe": "TCP connect to loopback:9"}
if denial is None:
    raise AssertionError("Network connection unexpectedly succeeded")
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

from ml_training.common import ROOT, read_json, verify_run, write_json
from ml_training.device_data import SYSTEM, score
from ml_training.mlx_models import infer, infer_rows
from ml_training.structured import gate, load_variant

verify_run(args.run)
model, tokenizer = load_variant(args.run, args.variant)
first = infer(model, tokenizer, SYSTEM, "Set the thermostat to 22 degrees.", max_tokens=64)
first_inference_seconds = time.perf_counter() - started
challenge = read_json(ROOT / read_json(args.run / "run.json")["dataset_path"] / "challenge.json")
outputs = infer_rows(model, tokenizer, challenge, max_tokens=64)
rows = [{"input": row["user"], "expected": row["expected"], "output": output,
         "runtime": gate("edge", output["text"])} for row, output in zip(challenge, outputs, strict=True)]
write_json(args.output, {"status": "actual_offline_inference", "run": str(args.run.resolve()), "variant": args.variant,
                         "network_denial_observed": denial, "fresh_process_first_inference_seconds": first_inference_seconds,
                         "cold_scope": "Fresh interpreter through imports, checksums, model loading, and first generation; OS file cache uncontrolled",
                         "first_prediction": first, "first_runtime": gate("edge", first["text"]),
                         "challenge_metrics": score(challenge, outputs), "challenge_predictions": rows,
                         "process_peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         "simulator_only": True, "elapsed_seconds": time.perf_counter() - started})
print(args.output)
