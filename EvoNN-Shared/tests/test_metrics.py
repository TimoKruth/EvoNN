from __future__ import annotations

import numpy as np

from evonn_shared.metrics import (
    compute_task_metric,
    metric_direction_for_task,
    metric_name_for_task,
    quality_from_metric,
)


def test_task_metric_names_and_directions_are_canonical() -> None:
    assert metric_name_for_task("classification") == "accuracy"
    assert metric_name_for_task("regression") == "mse"
    assert metric_name_for_task("language_modeling") == "perplexity"
    assert metric_direction_for_task("classification") == "max"
    assert metric_direction_for_task("regression") == "min"
    assert metric_direction_for_task("language_modeling") == "min"


def test_compute_classification_metric_accepts_label_predictions() -> None:
    metric = compute_task_metric(
        "classification",
        np.array([0, 2, 1]),
        np.array([0, 2, 0]),
        classification_predictions_are_labels=True,
    )

    assert metric.metric_name == "accuracy"
    assert metric.metric_direction == "max"
    assert metric.metric_value == 2 / 3
    assert metric.quality == 2 / 3


def test_compute_regression_metric_uses_negative_mse_quality() -> None:
    metric = compute_task_metric("regression", np.array([1.0, 2.0]), np.array([2.0, 2.0]))

    assert metric.metric_name == "mse"
    assert metric.metric_direction == "min"
    assert metric.metric_value == 0.5
    assert metric.quality == -0.5
    assert quality_from_metric("regression", 0.5) == -0.5


def test_compute_language_modeling_metric_accepts_probabilities() -> None:
    probs = np.array(
        [
            [[0.8, 0.2], [0.1, 0.9]],
            [[0.7, 0.3], [0.6, 0.4]],
        ]
    )
    targets = np.array([[0, 1], [0, 1]])

    metric = compute_task_metric(
        "language_modeling",
        targets,
        probs,
        language_prediction_kind="probabilities",
    )

    assert metric.metric_name == "perplexity"
    assert metric.metric_direction == "min"
    assert metric.metric_value > 1.0
    assert metric.quality == -metric.metric_value
