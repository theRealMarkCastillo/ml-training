"""Re-evaluate the latest local CPU runs from the current Python environment."""

import argparse
from http.server import ThreadingHTTPServer
import json
import subprocess
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ml_training import explainability, fast_classifier, tabular
from ml_training.common import ROOT, digest, environment, read_json, write_json


def latest(demo, seed):
    candidates = [
        p
        for p in (ROOT / "artifacts" / demo).glob(f"local-s{seed}-*")
        if read_json(p / "run.json")["status"] == "evaluated"
    ]
    if not candidates:
        raise RuntimeError(f"Missing evaluated {demo} seed {seed}")
    return sorted(candidates)[-1]


def check_http(run):
    runtime = fast_classifier.session(run / "model.onnx")
    server = ThreadingHTTPServer(("127.0.0.1", 0), fast_classifier.handler(runtime))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    cases = [
        ({"features": [[3, 0.01, 5, 2, 800, 0, 0.5, 3]]}, 200),
        ({"features": []}, 400),
        ({"features": [[True] * 8]}, 400),
        ({"features": [[float("nan")] * 8]}, 400),
        ({"features": [[1] * 7]}, 400),
        ({"extra": 1}, 400),
    ]
    results = []
    try:
        for payload, expected in cases:
            req = Request(
                f"http://127.0.0.1:{server.server_port}/predict",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urlopen(req, timeout=5) as response:
                    code = response.status
            except HTTPError as response:
                code = response.code
                response.close()
            if code != expected:
                raise AssertionError(f"HTTP returned {code}; expected {expected}")
            results.append({"expected_status": expected, "actual_status": code})
        return results
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true")
    args = parser.parse_args()
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    evidence = {
        "python_executable": sys.executable,
        "environment": environment(),
        "tests": {"returncode": tests.returncode, "stdout": tests.stdout, "stderr": tests.stderr},
        "evaluations": [],
        "http": {"status": "not requested"},
    }
    if tests.returncode:
        print(tests.stdout)
        print(tests.stderr)
        raise SystemExit(tests.returncode)
    for demo, evaluate in [("tabular", tabular.reevaluate), ("fast_classifier", fast_classifier.reevaluate)]:
        for seed in [41, 42, 43]:
            run = latest(demo, seed)
            result = evaluate(run)
            result["source_manifest_sha256"] = digest(run / "run.json")
            evidence["evaluations"].append(result)
    run = latest("explainability", 41)
    result = explainability.reevaluate(run)
    result["source_manifest_sha256"] = digest(run / "run.json")
    evidence["evaluations"].append(result)
    if args.http:
        evidence["http"] = {"status": "passed", "cases": check_http(latest("fast_classifier", 41))}
    output = ROOT / "artifacts" / "validation" / ("cpu-with-http.json" if args.http else "cpu.json")
    write_json(output, evidence)
    print(tests.stdout.strip())
    print(f"Reproduced {len(evidence['evaluations'])} saved runs; evidence: {output}")


if __name__ == "__main__":
    main()
