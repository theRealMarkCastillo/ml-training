"""Seeded synthetic events with causal features and immutable temporal splits."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import polars as pl

from ml_training.common import ROOT, digest, new_dataset_folder, read_json, write_json

NUMERIC = [
    "amount",
    "hour_sin",
    "hour_cos",
    "failed_auth",
    "is_foreign",
    "account_age_days",
    "prior_count_1h",
    "prior_failure_rate_1h",
    "prior_mean_amount_1h",
]
CATEGORICAL = ["merchant"]
FEATURES = NUMERIC + CATEGORICAL
DATA_VERSION = "events-v1"


def generate_events(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    timestamp = np.arange(n, dtype=np.int64) * 30
    entity = rng.integers(0, 350, n)
    # New entities in the final period provide an explicit unseen-entity slice.
    new_mask = (np.arange(n) >= int(n * 0.8)) & (rng.random(n) < 0.25)
    entity[new_mask] = rng.integers(350, 420, new_mask.sum())
    susceptibility = rng.normal(0, 0.7, 420)  # latent; never enters the feature matrix
    drift = np.arange(n) >= int(n * 0.8)
    merchant = rng.choice(["grocery", "travel", "digital", "retail"], n)
    amount = rng.lognormal(3.5 + 0.3 * drift, 1, n)
    foreign = rng.binomial(1, 0.12 + 0.07 * drift, n)
    failed = rng.poisson(0.25 + 0.25 * drift, n)
    age = rng.integers(1, 1800, n)
    hour = (timestamp // 3600) % 24
    # A stochastic outcome, with interactions and unobserved entity variation.
    logit = (
        -4.8
        + 0.65 * (np.log1p(amount) - 3.5)
        + 0.8 * foreign
        + 0.55 * failed
        + 1.25 * (foreign * (merchant == "digital"))
        + 0.8 * (age < 45)
        + 0.65 * ((hour < 5) & (failed > 0))
        + susceptibility[entity]
        + 0.7 * drift * (merchant == "travel")
    )
    label = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    amount_missing = rng.random(n) < 0.025
    observed_amount = [None if missing else float(a) for a, missing in zip(amount, amount_missing)]
    return pl.DataFrame(
        {
            "event_id": np.arange(n),
            "timestamp": timestamp,
            "entity_id": entity,
            "amount": observed_amount,
            "merchant": merchant,
            "failed_auth": failed,
            "is_foreign": foreign,
            "account_age_days": age,
            "label": label,
        }
    )


def build_features(events: pl.DataFrame) -> pl.DataFrame:
    """Only strictly earlier events in the previous hour may influence a row.

    Labels are never consulted. Events at the same timestamp do not influence
    one another. The deque stores observed metadata, not delayed outcomes.
    """
    if events["event_id"].n_unique() != events.height:
        raise ValueError("Duplicate event IDs")
    events = events.sort(["timestamp", "event_id"])
    histories = defaultdict(deque)
    rolling = []
    for row in events.iter_rows(named=True):
        history = histories[row["entity_id"]]
        now = row["timestamp"]
        while history and history[0][0] < now - 3600:
            history.popleft()
        prior = [item for item in history if item[0] < now]
        amounts = [a for _, a, _ in prior if a is not None]
        rolling.append(
            (
                len(prior),
                sum(f > 0 for _, _, f in prior) / len(prior) if prior else 0,
                float(np.mean(amounts)) if amounts else None,
            )
        )
        history.append((now, row["amount"], row["failed_auth"]))
    return events.with_columns(
        pl.Series("prior_count_1h", [r[0] for r in rolling]),
        pl.Series("prior_failure_rate_1h", [r[1] for r in rolling], dtype=pl.Float64),
        pl.Series("prior_mean_amount_1h", [r[2] for r in rolling], dtype=pl.Float64),
        (((pl.col("timestamp") // 3600) % 24) * (2 * np.pi / 24)).sin().alias("hour_sin"),
        (((pl.col("timestamp") // 3600) % 24) * (2 * np.pi / 24)).cos().alias("hour_cos"),
    )


def split_frame(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    # Validation is divided into calibration and threshold/model selection.
    n = frame.height
    a, b, c = int(n * 0.6), int(n * 0.7), int(n * 0.8)
    return {"train": frame[:a], "calibration": frame[a:b], "validation": frame[b:c], "test": frame[c:]}


def audit_splits(parts: dict[str, pl.DataFrame]) -> dict:
    seen = set()
    previous_time = -1
    report = {}
    training_entities = set(parts["train"]["entity_id"].to_list())
    for name in ["train", "calibration", "validation", "test"]:
        frame = parts[name]
        ids = set(frame["event_id"].to_list())
        if seen & ids or frame["timestamp"].min() <= previous_time:
            raise ValueError("Temporal split leakage")
        seen.update(ids)
        previous_time = frame["timestamp"].max()
        report[name] = {
            "rows": frame.height,
            "positive_count": int(frame["label"].sum()),
            "positive_rate": float(frame["label"].mean()),
            "time_min": frame["timestamp"].min(),
            "time_max": previous_time,
            "entities_seen_in_training": len(set(frame["entity_id"].to_list()) & training_entities),
        }
    report["entity_policy"] = (
        "Chronological split; recurring entities permitted, IDs excluded from predictors; unseen test entities reported separately."
    )
    return report


def prepare(profile: str, seed: int) -> Path:
    n = 2000 if profile == "smoke" else 50_000
    folder = ROOT / "data" / "generated" / f"{DATA_VERSION}-{profile}-s{seed}"
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        for name, sha in manifest["files"].items():
            if digest(folder / name) != sha:
                raise ValueError(f"Frozen dataset modified: {name}")
        return folder
    new_dataset_folder(folder)
    events = generate_events(n, seed)
    events.write_parquet(folder / "events.parquet")
    parts = split_frame(build_features(events))
    audit = audit_splits(parts)
    for name, frame in parts.items():
        frame.write_parquet(folder / f"{name}.parquet")
    write_json(
        manifest_path,
        {
            "version": DATA_VERSION,
            "synthetic": True,
            "seed": seed,
            "profile": profile,
            "features": FEATURES,
            "audit": audit,
            "files": {p.name: digest(p) for p in folder.glob("*.parquet")},
        },
    )
    return folder


def load_parts(folder: Path) -> dict[str, pl.DataFrame]:
    manifest = read_json(folder / "manifest.json")
    for name, sha in manifest["files"].items():
        if digest(folder / name) != sha:
            raise ValueError(f"Frozen dataset modified: {name}")
    return {
        name: pl.read_parquet(folder / f"{name}.parquet") for name in ["train", "calibration", "validation", "test"]
    }
