"""Supplement completed D06 runs with paired native/ONNX batch-size profiles."""

import argparse
from pathlib import Path

import joblib
import numpy as np

from ml_training.common import ROOT, benchmark, digest, environment, read_json, verify_run, write_json
from ml_training.fast_classifier import probabilities, session

parser = argparse.ArgumentParser()
parser.add_argument("runs",nargs="+",type=Path)
parser.add_argument("--output",type=Path,required=True)
args = parser.parse_args()
results = []
for run in args.runs:
    verify_run(run)
    info = read_json(run / "run.json")
    manifest_sha = digest(run / "run.json")
    folder = ROOT / info["dataset_path"]
    data = read_json(folder / "manifest.json")
    assert digest(folder / "manifest.json")==info["dataset_sha256"]
    assert digest(folder / "requests.npz")==data["sha256"]
    x = np.load(folder / "requests.npz")["x"]
    start = int(.8*len(x))
    model = joblib.load(run / f"{info['selected_model']}.joblib")
    runtime = session(run / "model.onnx")
    pairs = {}
    for batch_size in [1,256]:
        inputs = x[start:start+batch_size]
        native = model.predict_proba(inputs)[:,1]
        exported = probabilities(runtime,inputs)
        parity = float(np.max(np.abs(native-exported)))
        assert parity <= 1e-5
        backend_results = {}
        for backend,fn in [("native",lambda:model.predict_proba(inputs)),("onnx",lambda:probabilities(runtime,inputs))]:
            timing = benchmark(fn,warmups=20,samples=200)
            timing.update(batch_size=batch_size,rows_per_second=batch_size*1000/timing["mean"])
            backend_results[backend] = timing
        pairs[str(batch_size)] = {"parity_max_abs_error":parity,"backends":backend_results}
    assert manifest_sha==digest(run / "run.json")
    results.append({"run":str(run),"source_manifest_sha256":manifest_sha,"selected_model":info["selected_model"],
                    "seed":info["seed"],"batch_profiles":pairs})
    write_json(args.output,{"status":"measured","environment":environment(),"runs":results,
        "scope":"Warm repeated fixed inputs from the first held-out rows, identical inputs for both backends; concurrency 1, 20 warmups and 200 samples. Preprocessing is included. This is model throughput, not HTTP throughput. No model training or source-run mutation."})
    print(f"Profiled native and ONNX batches: {run}",flush=True)
