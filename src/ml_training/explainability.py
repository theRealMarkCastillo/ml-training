"""D07: explanations of actual D05 predictions, with executable consistency checks."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import polars as pl
import shap
from xgboost import XGBClassifier

from ml_training.common import ROOT, complete_run, digest, new_run, read_json, verify_run, write_json
from ml_training.data import FEATURES, NUMERIC, load_parts
from ml_training.tabular import matrix


def setup(source: Path):
    verify_run(source)
    info = read_json(source / "run.json")
    folder = ROOT / info["dataset_path"]
    if digest(folder / "manifest.json") != info["dataset_sha256"]:
        raise ValueError("D05 dataset manifest changed")
    parts = load_parts(folder)
    tree = joblib.load(source / "xgboost.joblib")
    preprocessing = tree.pipeline.named_steps["preprocess"]
    # Deterministic, train-only background. No evaluation rows affect explanations.
    background_rows = parts["train"].sample(n=min(100, parts["train"].height), seed=info["seed"])
    background = preprocessing.transform(matrix(background_rows)).astype(float)
    explainer = shap.TreeExplainer(tree.pipeline.named_steps["model"], data=background,
                                   feature_perturbation="interventional", model_output="raw")
    names = preprocessing.get_feature_names_out(FEATURES).tolist()
    return info, parts, tree, explainer, names, background_rows["event_id"].to_list()


def explain_rows(tree, explainer, names, rows):
    x = matrix(rows)
    transformed = tree.pipeline.named_steps["preprocess"].transform(x).astype(float)
    values = np.asarray(explainer.shap_values(transformed))
    base = float(explainer.expected_value)
    margin = tree.pipeline.named_steps["model"].predict(transformed, output_margin=True)
    residual = np.abs(values.sum(axis=1) + base - margin)
    if float(residual.max()) > 1e-5:
        raise AssertionError(f"SHAP reconstruction failed: {residual.max()}")
    result = []
    calibrated = tree.predict_proba(x)[:, 1]
    raw = tree.raw_probability(x)
    for i in range(rows.height):
        order = np.argsort(-np.abs(values[i]))
        result.append({
            "event_id": int(rows["event_id"][i]) if "event_id" in rows.columns else None,
            "calibrated_probability": float(calibrated[i]), "uncalibrated_probability": float(raw[i]),
            "raw_log_odds": float(margin[i]), "base_log_odds": base,
            "reconstruction_abs_error": float(residual[i]),
            "contributions_log_odds": dict(zip(names, values[i].tolist())),
            "reasons": [f"{names[j]} {'increases' if values[i,j] >= 0 else 'decreases'} raw model log-odds by {abs(values[i,j]):.4f} relative to the background" for j in order[:3]],
        })
    return result, values


def modify_row(row: pl.DataFrame, updates: dict) -> pl.DataFrame:
    for key, value in updates.items():
        if key not in FEATURES:
            raise ValueError(f"Unknown or immutable feature: {key}")
        if key == "merchant":
            if value not in ["grocery", "travel", "digital", "retail"]:
                raise ValueError("Unknown merchant")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError("Feature edits require finite numbers")
            if key in ["hour_sin", "hour_cos"] and not -1 <= value <= 1:
                raise ValueError("Hour encoding must be in [-1, 1]")
            if key in ["is_foreign", "prior_failure_rate_1h"] and not 0 <= value <= 1:
                raise ValueError("Rate must be in [0, 1]")
            if key == "is_foreign" and value not in [0, 1]:
                raise ValueError("is_foreign must be 0 or 1")
            if key in ["failed_auth", "prior_count_1h", "account_age_days"] and int(value) != value:
                raise ValueError("Counts and age require whole numbers")
            if key not in ["hour_sin", "hour_cos"] and value < 0:
                raise ValueError("Feature must be nonnegative")
        row = row.with_columns(pl.lit(value).alias(key))
    return row


def correlated_feature_example(seed):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=600)
    x = np.column_stack([signal, signal])
    y = (signal > 0).astype(int)
    # Two indistinguishable predictors: changing column order can move credit.
    records = []
    for names in [["signal", "duplicate"], ["duplicate", "signal"]]:
        model = XGBClassifier(n_estimators=10, max_depth=2, n_jobs=1, random_state=seed)
        model.fit(x[:500], y[:500])
        explainer = shap.TreeExplainer(model, data=x[:100], feature_perturbation="interventional", model_output="raw")
        values = explainer.shap_values(x[500:])
        records.append({"column_order": names, "mean_abs_attribution": dict(zip(names, np.abs(values).mean(axis=0).tolist())),
                        "probabilities": model.predict_proba(x[500:])[:, 1].tolist()})
    return {"description": "Separate synthetic demonstration with exactly duplicated features (correlation=1). Prediction equivalence does not identify a unique causal source of risk.",
            "max_probability_difference": float(np.max(np.abs(np.array(records[0]["probabilities"]) - records[1]["probabilities"]))),
            "variants": records}


def run(source: Path, row_index: int = 0, updates: dict | None = None) -> Path:
    info, parts, tree, explainer, names, background_ids = setup(source)
    result = new_run("explainability", info["profile"], info["seed"])
    selected = parts["test"].head(100)
    explanations, values = explain_rows(tree, explainer, names, selected)
    write_json(result / "explanations.json", explanations)
    global_attribution = dict(zip(names, np.abs(values).mean(axis=0).tolist()))
    write_json(result / "global_attribution.json", global_attribution)
    if not 0 <= row_index < parts["test"].height:
        raise ValueError("Row index is outside held-out data")
    row = parts["test"][row_index:row_index + 1]
    updates = updates if updates is not None else {"amount": float(row["amount"][0] or 30) * 1.01}
    changed = modify_row(row, updates)
    before, before_values = explain_rows(tree, explainer, names, row)
    after, after_values = explain_rows(tree, explainer, names, changed)
    write_json(result / "feature_edit.json", {"row_index": row_index, "updates": updates, "before": before[0], "after": after[0],
                                             "attribution_l1_change": float(np.abs(before_values - after_values).sum()),
                                             "note": "Conditional feature edit; causal downstream history is not regenerated."})
    logistic = joblib.load(source / "logistic.joblib")
    coefficients = logistic.pipeline.named_steps["model"].coef_[0]
    write_json(result / "logistic_coefficients.json", {
        "space": "Uncalibrated log-odds: numeric coefficients per training-standardized unit; categorical indicators from one-hot encoding.",
        "intercept": float(logistic.pipeline.named_steps["model"].intercept_[0]),
        "coefficients": dict(zip(names, coefficients.tolist())),
        "numeric_training_means": dict(zip(NUMERIC, logistic.pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"].mean_.tolist())),
        "numeric_training_scales": dict(zip(NUMERIC, logistic.pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"].scale_.tolist())),
    })
    write_json(result / "correlated_features.json", correlated_feature_example(info["seed"]))
    max_error = max(e["reconstruction_abs_error"] for e in explanations)
    (result / "report.md").write_text(
        f"# D07: Explaining actual D05 predictions\n\nSource run: `{source.name}`. Explained {len(explanations)} held-out rows. "
        f"Maximum additivity residual: {max_error:.3g} (tolerance 1e-5).\n\n"
        "TreeSHAP uses interventional perturbation with 100 training background rows and raw log-odds output. "
        "Contributions reconstruct the uncalibrated model margin, not calibrated probability. Both probabilities are shown separately. "
        "Global importance is the mean absolute contribution over these 100 rows, not the whole population.\n\n"
        "`feature_edit.json` compares a selected held-out row before and after an allowed edit. "
        "This is a conditional feature edit, not a causal intervention. `logistic_coefficients.json` describes standardized numeric coefficients and category indicators. "
        "`correlated_features.json` shows two models with duplicate predictors: equivalent predictions can allocate feature credit differently. "
        "Attributions and reason codes do not establish causality or regulatory compliance.\n")
    complete_run(result, source_run=str(source.relative_to(ROOT)), source_manifest_sha256=digest(source / "run.json"),
                 background_event_ids=background_ids, output_space="raw log-odds", perturbation="interventional",
                 rows_explained=len(explanations), max_additivity_residual=max_error, tolerance=1e-5,
                 edited_row_index=row_index, updates=updates)
    return result


def reevaluate(run: Path) -> dict:
    verify_run(run)
    info = read_json(run / "run.json")
    source = ROOT / info["source_run"]
    if digest(source / "run.json") != info["source_manifest_sha256"]:
        raise ValueError("D05 source manifest changed")
    _, parts, tree, explainer, names, _ = setup(source)
    actual, _ = explain_rows(tree, explainer, names, parts["test"].head(info["rows_explained"]))
    expected = read_json(run / "explanations.json")
    if actual != expected:
        raise AssertionError("Explanations did not reproduce")
    return {"status": "reproduced", "run": str(run), "rows": len(actual)}


def infer_edit(run: Path, row_index: int, updates: dict) -> dict:
    """Recompute both predictions and contributions from saved D05 weights."""
    verify_run(run)
    info = read_json(run / "run.json")
    source = ROOT / info["source_run"]
    if digest(source / "run.json") != info["source_manifest_sha256"]:
        raise ValueError("D05 source manifest changed")
    _, parts, tree, explainer, names, _ = setup(source)
    if not 0 <= row_index < parts["test"].height:
        raise ValueError("Row index is outside held-out data")
    row = parts["test"][row_index:row_index+1]
    changed = modify_row(row,updates)
    before, bv = explain_rows(tree,explainer,names,row)
    after, av = explain_rows(tree,explainer,names,changed)
    return {"execution":"fresh_saved_model","source_run":str(source),"row_index":row_index,
            "features_before":row.select(FEATURES).to_dicts()[0],"features_after":changed.select(FEATURES).to_dicts()[0],
            "updates":updates,"before":before[0],"after":after[0],"attribution_l1_change":float(np.abs(bv-av).sum()),
            "note":"Conditional feature edit; causal downstream history is not regenerated. Contributions are raw log-odds, not probability or causal effects."}
