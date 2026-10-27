import numpy as np
import polars as pl
import pytest

from ml_training.data import FEATURES, audit_splits, build_features, generate_events, split_frame
from ml_training.metrics import threshold_at_fpr


def test_future_events_and_labels_cannot_change_past_features():
    events = generate_events(1000, 17)
    whole = build_features(events)
    prefix = build_features(events[:600])
    assert whole[:600].select(FEATURES).equals(prefix.select(FEATURES))
    relabeled = build_features(events.with_columns((1 - pl.col("label")).alias("label")))
    assert whole.select(FEATURES).equals(relabeled.select(FEATURES))


def test_current_event_does_not_enter_its_own_history():
    events = generate_events(3, 17).with_columns(pl.lit(1).alias("entity_id"), pl.Series("timestamp", [0, 0, 3601]))
    frame = build_features(events)
    assert frame["prior_count_1h"].to_list() == [0, 0, 0]


def test_history_uses_prior_event_only():
    events = generate_events(3, 17).with_columns(
        pl.lit(1).alias("entity_id"), pl.Series("timestamp", [0, 30, 60]), pl.Series("amount", [10.0, 30.0, 900.0])
    )
    frame = build_features(events)
    assert frame["prior_count_1h"].to_list() == [0, 1, 2]
    assert frame["prior_mean_amount_1h"].to_list() == [None, 10.0, 20.0]


def test_split_audit_rejects_overlap_and_excludes_identifiers():
    parts = split_frame(build_features(generate_events(1000, 17)))
    audit_splits(parts)
    assert not {"label", "entity_id", "event_id", "timestamp"}.intersection(FEATURES)
    parts["test"] = parts["train"][:10]
    with pytest.raises(ValueError, match="leakage"):
        audit_splits(parts)


def test_threshold_fpr_budget_handles_ties():
    y = np.array([0] * 100 + [1] * 10)
    scores = np.array([0.1] * 95 + [0.9] * 5 + [0.9] * 10)
    threshold = threshold_at_fpr(y, scores)
    assert (scores[y == 0] >= threshold).mean() <= 0.05
    assert (scores[y == 1] >= threshold).mean() == 1


def test_preprocessing_is_fit_on_training_only():
    from ml_training.tabular import matrix, preprocessor

    frames = split_frame(build_features(generate_events(1000, 41)))
    transform = preprocessor().fit(matrix(frames["train"]))
    before = transform.named_transformers_["numeric"].named_steps["impute"].statistics_.copy()
    shifted = frames["test"].with_columns(pl.lit(1e12).alias("amount"))
    transform.transform(matrix(shifted))
    np.testing.assert_array_equal(before, transform.named_transformers_["numeric"].named_steps["impute"].statistics_)
