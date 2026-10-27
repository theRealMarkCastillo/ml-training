"""D05: actual classical training, calibration, reload, and held-out evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from ml_training.common import (ROOT, benchmark, cold_process, complete_run, digest, mark_failed, new_run,
                                read_json, require_reproduced, verify_run, write_json)
from ml_training.data import (CATEGORICAL, FEATURES, NUMERIC, load_parts, prepare)
from ml_training.metrics import binary_metrics, bootstrap_ap, threshold_at_fpr


def matrix(frame: pl.DataFrame) -> np.ndarray:
    numeric = frame.select(NUMERIC).to_numpy().astype(float)
    categorical = frame.select(CATEGORICAL).to_numpy()
    return np.column_stack([numeric, categorical])


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                              ("scale", StandardScaler())]), list(range(len(NUMERIC)))),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [len(NUMERIC)]),
    ])


def logits(probabilities):
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


@dataclass
class CalibratedModel:
    pipeline: Pipeline
    calibration: LogisticRegression | None

    def raw_probability(self, x):
        return self.pipeline.predict_proba(x)[:, 1]

    def predict_proba(self, x):
        raw = self.raw_probability(x)
        positive = self.calibration.predict_proba(logits(raw))[:, 1] if self.calibration is not None else raw
        return np.column_stack([1 - positive, positive])


def factories(seed: int, smoke: bool) -> dict:
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic": LogisticRegression(max_iter=500, random_state=seed),
        "xgboost": XGBClassifier(n_estimators=25 if smoke else 160, max_depth=4,
                                  learning_rate=.06, subsample=.9, colsample_bytree=.9,
                                  reg_lambda=5, min_child_weight=10, n_jobs=1,
                                  tree_method="hist", random_state=seed),
    }


def evaluate_model(model, frame, threshold, seed, training_entities) -> tuple[dict, np.ndarray]:
    x, y = matrix(frame), frame["label"].to_numpy()
    scores = model.predict_proba(x)[:, 1]
    report = binary_metrics(y, scores, threshold)
    report["ap_95_interval"] = bootstrap_ap(y, scores, seed)
    slices = {}
    masks = {f"merchant={name}": frame["merchant"].to_numpy() == name
             for name in sorted(frame["merchant"].unique().to_list())}
    masks["new_entity"] = np.array([v not in training_entities for v in frame["entity_id"].to_list()])
    masks["existing_entity"] = ~masks["new_entity"]
    masks["missing_amount"] = frame["amount"].is_null().to_numpy()
    for name, mask in masks.items():
        if mask.any():
            slices[name] = binary_metrics(y[mask], scores[mask], threshold)
    report["slices"] = slices
    return report, scores


def train(profile: str, seed: int) -> Path:
    folder = prepare(profile, seed)
    parts = load_parts(folder)
    run = new_run("tabular", profile, seed)
    try:
        shutil.copyfile(folder / "manifest.json", run / "dataset.json")
        x_train, y_train = matrix(parts["train"]), parts["train"]["label"].to_numpy()
        x_cal, y_cal = matrix(parts["calibration"]), parts["calibration"]["label"].to_numpy()
        x_val, y_val = matrix(parts["validation"]), parts["validation"]["label"].to_numpy()
        joblib.dump(x_val[:1], run / "benchmark_input.joblib")
        training_entities = set(parts["train"]["entity_id"].to_list())
        validation = {}
        models = {}
        training = {}
        for name, estimator in factories(seed, profile == "smoke").items():
            start = time.perf_counter()
            pipeline = Pipeline([("preprocess", preprocessor()), ("model", estimator)])
            pipeline.fit(x_train, y_train)
            calibration = None
            if name != "dummy":
                calibration = LogisticRegression(C=10, max_iter=500)
                calibration.fit(logits(pipeline.predict_proba(x_cal)[:, 1]), y_cal)
            model = CalibratedModel(pipeline, calibration)
            fit_seconds = time.perf_counter() - start
            scores = model.predict_proba(x_val)[:, 1]
            threshold = threshold_at_fpr(y_val, scores)
            validation[name] = binary_metrics(y_val, scores, threshold)
            validation[name]["uncalibrated"] = binary_metrics(y_val, model.raw_probability(x_val), .5)
            path = run / f"{name}.joblib"
            joblib.dump(model, path)
            start = time.perf_counter()
            loaded = joblib.load(path)
            reload_seconds = time.perf_counter() - start
            parity = float(np.max(np.abs(scores - loaded.predict_proba(x_val)[:, 1])))
            if parity > 1e-12:
                raise AssertionError(f"Reload changed {name} predictions: {parity}")
            models[name] = loaded
            training[name] = {"fit_seconds": fit_seconds, "reload_seconds": reload_seconds,
                              "reload_note": "In-process reload, OS file cache may be warm; not cold process startup.",
                              "reload_max_abs_error": parity,
                              "fresh_process_first_inference": cold_process(path, run / "benchmark_input.joblib", "joblib"),
                              "model_parameters": {k: "NaN (missing-value marker)" if isinstance(v, float) and np.isnan(v) else v
                                                   for k, v in estimator.get_params().items()},
                              "latency_batch_1_with_preprocessing": benchmark(lambda: loaded.predict_proba(x_val[:1]))}
        # Selection is persisted before the test split is evaluated.
        winner = max(validation, key=lambda n: validation[n]["pr_auc_average_precision"])
        write_json(run / "selection.json", {"criterion": "highest validation average precision",
                                            "fpr_budget": .05, "selected_model": winner, "validation": validation})
        reports = {}
        predictions = parts["test"].select(["event_id", "timestamp", "entity_id", "merchant", "label"])
        for name, model in models.items():
            report, scores = evaluate_model(model, parts["test"], validation[name]["threshold"], seed, training_entities)
            reports[name] = report
            predictions = predictions.with_columns(pl.Series(f"{name}_probability", scores),
                                                    pl.Series(f"{name}_prediction", scores >= validation[name]["threshold"]))
        predictions.write_parquet(run / "predictions.parquet")
        write_json(run / "metrics.json", reports)
        # Retain false positives and false negatives, including scores and actual labels.
        failures = predictions.filter(pl.col(f"{winner}_prediction").cast(pl.Int64) != pl.col("label"))
        failures.head(30).write_json(run / "failure_examples.json")
        challenge = read_json(ROOT / "data" / "fixtures" / "tabular_challenge.json")
        challenge_frame = pl.DataFrame(challenge["rows"])
        challenge_scores = {n: m.predict_proba(matrix(challenge_frame))[:, 1].tolist() for n, m in models.items()}
        write_json(run / "challenge.json", {"description": challenge["description"], "scores": challenge_scores,
                                           "ordering_checks": {n: s[1] > s[0] for n, s in challenge_scores.items()}})
        write_report(run, reports, validation, winner, training, challenge_scores)
        complete_run(run, dataset_path=str(folder.relative_to(ROOT)), dataset_sha256=digest(folder / "manifest.json"),
                     synthetic=True, training=training, selected_model=winner,
                     selection_rule="validation average precision; test never selects models or thresholds",
                     features=FEATURES, transformer_fit_partition="train", calibration_partition="calibration",
                     test_partition="future drift period; includes new entities", teacher_or_api_cost_usd=0)
    except BaseException as error:
        mark_failed(run, error)
        raise
    return run


def write_report(run, reports, validation, winner, training, challenge_scores):
    latency = next(iter(training.values()))["latency_batch_1_with_preprocessing"]
    lines = ["# D05: Synthetic transaction anomaly experiment", "",
             f"Selected on validation average precision: **{winner}**.", "",
             "| Model | Validation AP | Test AP | Test ROC-AUC | Test Brier | Recall | Actual test FPR | Fit seconds |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, m in reports.items():
        lines.append(f"| {name} | {validation[name]['pr_auc_average_precision']:.4f} | {m['pr_auc_average_precision']:.4f} | {m['roc_auc']:.4f} | {m['brier']:.4f} | {m['recall']:.4f} | {m['false_positive_rate']:.4f} | {training[name]['fit_seconds']:.2f} |")
    lines += ["", "AP is average precision (a precision-recall summary). Its no-skill reference is positive prevalence, not 0.5.",
              "Calibration uses an earlier validation slice; threshold selection uses the following slice at a 5% empirical FPR budget. Future-period FPR can exceed that budget under drift.",
              "", "The test period has changed amount, authentication, and category risk distributions. Compare validation and test metrics as a drift sensitivity check, not a controlled estimate of the causal effect of drift.",
              "", "Full slice metrics, row-bootstrap intervals (which ignore entity correlation), and raw predictions accompany this report. Entity IDs are excluded; recurring entities are permitted chronologically, with unseen test entities reported separately.",
              "", f"Independent hand-authored low/high-risk challenge probabilities: `{challenge_scores}`. These are plausibility checks, not real labeled observations.",
              "", "Synthetic data demonstrates engineering mechanics. It does not establish effectiveness on real transactions. No operational deployment is recommended without representative data and an explicit cost-of-errors analysis.",
              "", f"Latency measurements include preprocessing at batch size 1, concurrency {latency['concurrency']}, {latency['warmups']} warmups and {latency['samples']} samples. Model threads are bounded where supported. Reload timing is not a cold-process startup benchmark.",
              "", f"Artifacts: run.json, dataset.json, selection.json, metrics.json, predictions.parquet, failure_examples.json, challenge.json, and {len(training)} reloadable joblib models ({', '.join(training)})."]
    (run / "report.md").write_text("\n".join(lines) + "\n")


def reevaluate(run: Path, tolerance: float = 0.) -> dict:
    verify_run(run)
    manifest = read_json(run / "run.json")
    folder = ROOT / manifest["dataset_path"]
    if digest(folder / "manifest.json") != manifest["dataset_sha256"]:
        raise ValueError("Dataset manifest changed")
    parts = load_parts(folder)
    original = read_json(run / "metrics.json")
    selection = read_json(run / "selection.json")
    for name in original:
        model = joblib.load(run / f"{name}.joblib")
        actual, _ = evaluate_model(model, parts["test"], selection["validation"][name]["threshold"],
                                   manifest["seed"], set(parts["train"]["entity_id"].to_list()))
        require_reproduced(actual, original[name], f"D05 {name} test metrics", tolerance)
    return {"status": "reproduced", "run": str(run), "models": list(original)}
