"""Fresh-process first prediction and peak memory, called by validate_slm.py."""

import argparse
from pathlib import Path
import resource
import time

start = time.perf_counter()
parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("variant")
parser.add_argument("output", type=Path)
args = parser.parse_args()

from ml_training.common import ROOT, read_json, write_json
info = read_json(args.run / "run.json")
rows = read_json(ROOT / info["dataset_path"] / "rows.json")
query = next(row["user"] for row in rows if row["split"] == "validation")
if info["demo"] == "distillation":
    from ml_training.distillation import infer_query
else:
    from ml_training.structured import infer_query
result = infer_query(args.run, query, args.variant)
write_json(args.output, {"variant": args.variant, "script_start_to_first_prediction_seconds": time.perf_counter()-start,
                         "process_peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         "prediction": result, "samples": 1,
                         "scope": "Fresh Python process, imports, artifact checksums, model loading and first generation; OS file cache uncontrolled"})
