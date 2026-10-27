"""D06: trained request classifier, ONNX parity, CPU and HTTP benchmarks."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from ml_training.common import ROOT, benchmark, cold_process, complete_run, digest, new_dataset_folder, mark_failed, new_run, read_json, require_reproduced, verify_run, write_json
from ml_training.metrics import binary_metrics, threshold_at_fpr
from ml_training.onnx_runtime import probabilities, session  # noqa: F401  (re-exported for callers)

FEATURES = ["requests_per_minute", "failure_ratio", "payload_kb", "distinct_paths",
            "account_age_days", "is_foreign", "path_entropy", "prior_rate"]


def generate(n, seed):
    rng = np.random.default_rng(seed)
    drift = np.arange(n) >= int(n * .8)
    rate = rng.lognormal(2 + .3 * drift, 1, n)
    failure = rng.beta(.6, 4, n)
    payload = rng.lognormal(2, 1.2, n)
    paths = rng.poisson(4, n)
    age = rng.uniform(0, 1000, n)
    foreign = rng.binomial(1, .15, n)
    entropy = rng.uniform(0, 4, n)
    prior = rate * rng.uniform(.3, 1.3, n)
    logit = (-4 + .025 * rate + 4 * failure + 1.1 * foreign * (paths > 5)
             + .8 * (age < 30) + .25 * drift * entropy)
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    x = np.column_stack([rate, failure, payload, paths, age, foreign, entropy, prior]).astype(np.float32)
    return x, y


def prepare(profile, seed):
    folder = ROOT / "data" / "generated" / f"requests-v1-{profile}-s{seed}"
    if (folder / "manifest.json").exists():
        info = read_json(folder / "manifest.json")
        if digest(folder / "requests.npz") != info["sha256"]:
            raise ValueError("Frozen request dataset changed")
        return folder
    new_dataset_folder(folder)
    x, y = generate(2000 if profile == "smoke" else 50_000, seed)
    np.savez_compressed(folder / "requests.npz", x=x, y=y)
    write_json(folder / "manifest.json", {"synthetic": True, "seed": seed, "rows": len(y),
                                         "features": FEATURES, "sha256": digest(folder / "requests.npz"),
                                         "split": "first 60% train; next 20% validation; final 20% future drift test"})
    return folder


def validate_request(payload):
    if not isinstance(payload, dict) or set(payload) != {"features"}:
        raise ValueError("Expected only a features key")
    rows = payload["features"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 256:
        raise ValueError("Expected 1 to 256 feature rows")
    for row in rows:
        if not isinstance(row, list) or len(row) != len(FEATURES):
            raise ValueError("Wrong feature shape")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in row):
            raise ValueError("Features must be numbers")
    x = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Features must be finite")
    return x


def handler(runtime):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            try:
                if self.path != "/predict":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 65536:
                    raise ValueError("Invalid body size")
                x = validate_request(json.loads(self.rfile.read(length)))
                body = json.dumps({"probabilities": probabilities(runtime, x).tolist()}).encode()
                self.send_response(200)
            except (ValueError, TypeError, OverflowError) as error:
                body = json.dumps({"error": str(error)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    return Handler


def benchmark_http(runtime, x):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(runtime))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = json.dumps({"features": x[:1].tolist()}).encode()
        url = f"http://127.0.0.1:{server.server_port}/predict"
        def request():
            with urlopen(Request(url, data=body, headers={"Content-Type": "application/json"}), timeout=5) as response:
                return json.loads(response.read())["probabilities"][0]
        error = abs(request() - float(probabilities(runtime, x[:1])[0]))
        if error > 1e-7:
            raise AssertionError("HTTP prediction differs")
        result = benchmark(request, samples=100)
        result["scope"] = "loopback HTTP, new connection per request, JSON serialization and validation, batch 1"
        return result
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def train(profile, seed):
    folder = prepare(profile, seed)
    dataset = np.load(folder / "requests.npz")
    x, y = dataset["x"], dataset["y"]
    a, b = int(.6 * len(y)), int(.8 * len(y))
    run = new_run("fast_classifier", profile, seed)
    try:
        models = {"logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed)),
                  "tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=40, random_state=seed)}
        validation = {}
        timings = {}
        for name, model in models.items():
            start = time.perf_counter()
            model.fit(x[:a], y[:a])
            timings[name] = time.perf_counter() - start
            p = model.predict_proba(x[a:b])[:, 1]
            validation[name] = binary_metrics(y[a:b], p, threshold_at_fpr(y[a:b], p))
            joblib.dump(model, run / f"{name}.joblib")
            reloaded = joblib.load(run / f"{name}.joblib")
            np.testing.assert_allclose(reloaded.predict_proba(x[a:b]), model.predict_proba(x[a:b]), rtol=0, atol=1e-12)
        winner = max(validation, key=lambda name: validation[name]["pr_auc_average_precision"])
        write_json(run / "selection.json", {"selected_model": winner, "criterion": "validation average precision", "validation": validation})
        model = joblib.load(run / f"{winner}.joblib")
        converted = convert_sklearn(model, initial_types=[("features", FloatTensorType([None, len(FEATURES)]))],
                                    options={id(model): {"zipmap": False}}, target_opset=18)
        (run / "model.onnx").write_bytes(converted.SerializeToString())
        start = time.perf_counter()
        runtime = session(run / "model.onnx")
        startup = time.perf_counter() - start
        native, exported = model.predict_proba(x[b:])[:, 1], probabilities(runtime, x[b:])
        parity = float(np.max(np.abs(native - exported)))
        if parity > 1e-5:
            raise AssertionError(f"ONNX parity exceeds 1e-5: {parity}")
        predictions = {"row_id": np.arange(b, len(y)), "label": y[b:], "onnx_probability": exported}
        reports = {}
        for name, candidate in models.items():
            p = candidate.predict_proba(x[b:])[:, 1]
            predictions[f"{name}_probability"] = p
            reports[name] = binary_metrics(y[b:], p, validation[name]["threshold"])
        pl.DataFrame(predictions).write_parquet(run / "predictions.parquet")
        predicted = native >= validation[winner]["threshold"]
        failures = np.flatnonzero(predicted != y[b:])[:30]
        write_json(run / "failure_examples.json", [{"row_id": int(b + i), "label": int(y[b+i]),
                                                    "probability": float(native[i]), "features": x[b+i].tolist()}
                                                   for i in failures])
        challenge = read_json(ROOT / "data" / "fixtures" / "classifier_challenge.json")
        challenge_x = validate_request({"features": challenge["rows"]})
        write_json(run / "challenge.json", {"description": challenge["description"],
                                            "native": model.predict_proba(challenge_x)[:, 1].tolist(),
                                            "onnx": probabilities(runtime, challenge_x).tolist()})
        joblib.dump(x[b:b+1], run / "benchmark_input.joblib")
        benchmarks = {"native_batch_1": benchmark(lambda: model.predict_proba(x[b:b+1])),
                      "native_batch_256": benchmark(lambda: model.predict_proba(x[b:b+256])),
                      "onnx_batch_1": benchmark(lambda: probabilities(runtime, x[b:b+1])),
                      "onnx_batch_256": benchmark(lambda: probabilities(runtime, x[b:b+256])),
                      "session_load_seconds": startup,
                      "fresh_process_first_inference": cold_process(run / "model.onnx", run / "benchmark_input.joblib", "onnx")}
        benchmarks["onnx_batch_256"]["rows_per_second"] = 256 * 1000 / benchmarks["onnx_batch_256"]["mean"]
        benchmarks["native_batch_256"]["rows_per_second"] = 256 * 1000 / benchmarks["native_batch_256"]["mean"]
        # Socket restrictions must remain visible; they do not invalidate model evaluation.
        try:
            benchmarks["http_batch_1"] = benchmark_http(runtime, x[b:])
        except PermissionError as error:
            benchmarks["http_batch_1"] = {"status": "blocked", "reason": str(error)}
        write_json(run / "metrics.json", reports)
        write_json(run / "benchmarks.json", benchmarks)
        onnx_p95 = benchmarks["onnx_batch_1"]["p95"]
        target = (f"The provisional model-only p95 target is 1 ms; measured ONNX batch-1 p95 was {onnx_p95:.4g} ms, "
                  f"{'within' if onnx_p95 < 1 else 'above'} the target on this host. ")
        (run / "report.md").write_text(
            f"# D06: Fast CPU classifier\n\nValidation selected **{winner}**. Native/ONNX maximum probability difference: {parity:.3g} (tolerance 1e-5).\n\n"
            f"Held-out metrics:\n```json\n{json.dumps(reports, indent=2)}\n```\n\n"
            f"Benchmark results:\n```json\n{json.dumps(benchmarks, indent=2)}\n```\n\n"
            "All rows are synthetic. Final 20% is a future drift slice. AP, recall, and FPR use validation-selected thresholds. "
            "ONNX runs on CPU with one intra/inter-op thread. Batch-256 throughput is not HTTP throughput. "
            + target + "Session loading is an in-process measurement, not cold process startup. "
            "The fresh-process check imports only onnxruntime, numpy and joblib, as a deployed scorer would.\n")
        complete_run(run, dataset_path=str(folder.relative_to(ROOT)), dataset_sha256=digest(folder / "manifest.json"),
                     synthetic=True, features=FEATURES, selected_model=winner, fit_seconds=timings,
                     onnx_max_abs_error=parity, onnx_tolerance=1e-5, execution_provider="CPUExecutionProvider",
                     intra_op_threads=1, inter_op_threads=1, p95_target_ms=1,
                     p95_target_met=onnx_p95 < 1,
                     http_evaluated="status" not in benchmarks["http_batch_1"])
    except BaseException as error:
        mark_failed(run, error)
        raise
    return run


def reevaluate(run, tolerance=0.):
    verify_run(run)
    info = read_json(run / "run.json")
    folder = ROOT / info["dataset_path"]
    if digest(folder / "manifest.json") != info["dataset_sha256"]:
        raise ValueError("Dataset manifest changed")
    data = np.load(folder / "requests.npz")
    if digest(folder / "requests.npz") != read_json(folder / "manifest.json")["sha256"]:
        raise ValueError("Dataset changed")
    x, y = data["x"], data["y"]
    b = int(.8 * len(y))
    original = read_json(run / "metrics.json")
    for name, expected in original.items():
        model = joblib.load(run / f"{name}.joblib")
        p = model.predict_proba(x[b:])[:, 1]
        require_reproduced(binary_metrics(y[b:], p, expected["threshold"]), expected,
                           f"D06 {name} test metrics", tolerance)
    runtime = session(run / "model.onnx")
    model = joblib.load(run / f"{info['selected_model']}.joblib")
    np.testing.assert_allclose(probabilities(runtime, x[b:]), model.predict_proba(x[b:])[:, 1], atol=1e-5, rtol=0)
    return {"status": "reproduced", "run": str(run), "onnx_parity": "passed"}
