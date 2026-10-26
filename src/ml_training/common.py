"""Shared provenance, persistence, and measurement contracts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text())


def environment() -> dict:
    packages = {}
    for name in ["numpy", "polars", "scikit-learn", "xgboost", "joblib",
                 "onnxruntime", "skl2onnx", "shap", "pytest", "torch", "transformers",
                 "sentence-transformers", "rank-bm25", "mlx", "mlx-lm", "trl", "peft", "accelerate"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "architecture": platform.machine(), "logical_cpus": os.cpu_count(),
        "free_disk_bytes": shutil.disk_usage(ROOT).free,
        "packages": packages,
        "thread_environment": {k: os.environ.get(k) for k in
                               ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
    }


def source_identity() -> dict:
    paths = sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("tests/**/*.py"), *ROOT.glob("scripts/*.py"),
                    *ROOT.glob("configs/*.json"), *ROOT.glob("data/fixtures/*.json"),
                    *ROOT.glob("environments/*/pyproject.toml"),
                    ROOT / "pyproject.toml"])
    result = {str(p.relative_to(ROOT)): digest(p) for p in paths if p.exists()}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True)
    return {"git_revision": git.stdout.strip() if git.returncode == 0 else None,
            "files": result,
            "environment_locks": {str(p.relative_to(ROOT)): digest(p) for p in ROOT.glob("environments/*/uv.lock")},
            "lock_sha256": digest(ROOT / "uv.lock") if (ROOT / "uv.lock").exists() else None}


def new_run(demo: str, profile: str, seed: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = ROOT / "artifacts" / demo / f"{profile}-s{seed}-{stamp}-{uuid4().hex[:6]}"
    run.mkdir(parents=True)
    write_json(run / "run.json", {
        "demo": demo, "profile": profile, "seed": seed, "status": "started",
        "started_at": stamp, "environment": environment(), "source": source_identity(),
    })
    return run


def complete_run(run: Path, **evidence) -> None:
    manifest = read_json(run / "run.json")
    manifest.update(evidence)
    manifest["status"] = "smoke" if manifest["profile"] == "smoke" else "evaluated"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    manifest["process_peak_rss_bytes"] = peak if platform.system() == "Darwin" else peak * 1024
    manifest["files"] = {str(p.relative_to(run)): {"sha256": digest(p), "bytes": p.stat().st_size}
                         for p in sorted(run.rglob("*")) if p.is_file() and p != run / "run.json"}
    write_json(run / "run.json", manifest)


def verify_run(run: Path) -> None:
    manifest = read_json(run / "run.json")
    if manifest["status"] not in ("smoke", "evaluated"):
        raise ValueError(f"Run is incomplete: {run}")
    for name, record in manifest["files"].items():
        if digest(run / name) != record["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {name}")


def benchmark(fn, warmups: int = 20, samples: int = 200) -> dict:
    for _ in range(warmups):
        fn()
    values = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        fn()
        values.append((time.perf_counter_ns() - start) / 1e6)
    return {"unit": "ms", "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "mean": float(np.mean(values)),
            "samples": samples, "warmups": warmups, "concurrency": 1}


def cold_process(model_path: Path, input_path: Path, backend: str) -> dict:
    """Fresh interpreter, imports, model load, and first inference; no cache claims."""
    start = time.perf_counter()
    child = subprocess.run([sys.executable, "-m", "ml_training.cold_infer", backend,
                            str(model_path), str(input_path)], cwd=ROOT,
                           capture_output=True, text=True, check=True, timeout=60)
    return {"process_start_to_exit_seconds": time.perf_counter() - start,
            "prediction": json.loads(child.stdout), "samples": 1,
            "scope": "Fresh Python process, imports, artifact/input load and first inference; OS file cache uncontrolled"}
