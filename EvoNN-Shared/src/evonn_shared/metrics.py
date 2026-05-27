"""Shared task metric semantics for EvoNN engines.

This module is intentionally narrow. It defines only the compare-facing metric
contract all engines already share; it does not own model execution, training,
search pressure, or candidate selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


TaskName = Literal["classification", "regression", "language_modeling"]
MetricDirection = Literal["max", "min"]
LanguagePredictionKind = Literal["logits", "probabilities"]


@dataclass(frozen=True)
class TaskMetric:
    """Normalized metric fields emitted by engine-local evaluators."""

    metric_name: str
    metric_direction: MetricDirection
    metric_value: float
    quality: float


def metric_name_for_task(task: str) -> str:
    """Return the canonical metric name for a benchmark task."""

    if task == "language_modeling":
        return "perplexity"
    if task == "regression":
        return "mse"
    return "accuracy"


def metric_direction_for_task(task: str) -> MetricDirection:
    """Return the canonical optimization direction for a benchmark task."""

    return "min" if task in {"regression", "language_modeling"} else "max"


def quality_from_metric(task: str, metric_value: float) -> float:
    """Map metric value to the shared higher-is-better quality convention."""

    if task in {"regression", "language_modeling"}:
        return -float(metric_value)
    return float(metric_value)


def compute_task_metric(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    classification_predictions_are_labels: bool = False,
    language_prediction_kind: LanguagePredictionKind = "logits",
) -> TaskMetric:
    """Compute the shared compare-facing metric for engine-local predictions."""

    if task == "language_modeling":
        return _language_modeling_metric(y_true, y_pred, prediction_kind=language_prediction_kind)
    if task == "regression":
        return _regression_metric(y_true, y_pred)
    return _classification_metric(
        y_true,
        y_pred,
        predictions_are_labels=classification_predictions_are_labels,
    )


def _classification_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    predictions_are_labels: bool,
) -> TaskMetric:
    if predictions_are_labels:
        preds = np.asarray(y_pred).reshape(-1).astype(int)
    elif y_pred.ndim == 2 and y_pred.shape[1] > 1:
        preds = np.argmax(y_pred, axis=1).reshape(-1).astype(int)
    else:
        preds = (np.asarray(y_pred).reshape(-1) > 0.5).astype(int)
    y_true_flat = np.asarray(y_true).reshape(-1).astype(int)
    accuracy = float(np.mean(preds == y_true_flat)) if y_true_flat.size else 0.0
    return TaskMetric("accuracy", "max", accuracy, accuracy)


def _language_modeling_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prediction_kind: LanguagePredictionKind,
) -> TaskMetric:
    predictions = np.asarray(y_pred)
    if predictions.ndim == 2:
        predictions = predictions[:, None, :]
    if predictions.ndim != 3:
        raise ValueError(f"language modeling predictions must be 2D or 3D, got {predictions.shape}")

    if prediction_kind == "probabilities":
        probs = np.clip(predictions.astype(np.float64), 1e-8, 1.0)
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
    else:
        shifted = predictions.astype(np.float64) - np.max(predictions, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        probs = np.clip(probs, 1e-8, 1.0)

    probs_flat = probs.reshape(-1, probs.shape[-1])
    targets_flat = np.asarray(y_true).reshape(-1).astype(int)
    if targets_flat.size != probs_flat.shape[0]:
        raise ValueError(
            "language modeling target/prediction length mismatch: "
            f"{targets_flat.size} targets vs {probs_flat.shape[0]} predictions"
        )
    targets_flat = np.clip(targets_flat, 0, probs_flat.shape[-1] - 1)
    cross_entropy = float(-np.mean(np.log(probs_flat[np.arange(targets_flat.shape[0]), targets_flat])))
    perplexity = float(np.exp(np.clip(cross_entropy, -20.0, 20.0)))
    return TaskMetric("perplexity", "min", perplexity, -perplexity)


def _regression_metric(y_true: np.ndarray, y_pred: np.ndarray) -> TaskMetric:
    y_pred_flat = np.asarray(y_pred).reshape(-1).astype(np.float64)
    y_true_flat = np.asarray(y_true).reshape(-1).astype(np.float64)
    mse = float(np.mean((y_pred_flat - y_true_flat) ** 2)) if y_true_flat.size else float("inf")
    return TaskMetric("mse", "min", mse, -mse)
