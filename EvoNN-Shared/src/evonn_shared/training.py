"""Shared training preprocessing helpers.

These helpers are intentionally narrow: they normalize regression targets and
calibrate final regression predictions without owning any engine search logic.
"""

from __future__ import annotations

import math

import numpy as np


def regression_target_stats(y_train: np.ndarray) -> tuple[float, float]:
    """Return stable mean/std stats for regression target normalization."""

    values = y_train.reshape(-1).astype(np.float32)
    mean = float(np.mean(values)) if values.size else 0.0
    std = float(np.std(values)) if values.size else 1.0
    if not math.isfinite(std) or std < 1e-8:
        std = 1.0
    return mean, std


def standardize_regression_targets(y_train: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Map regression targets to a stable training scale."""

    return ((y_train.reshape(-1).astype(np.float32) - mean) / std).astype(np.float32)


def restore_regression_predictions(y_pred: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Map standardized regression predictions back to the original target scale."""

    return (y_pred.reshape(-1).astype(np.float32) * std + mean).astype(np.float32)


def calibrate_regression_predictions(
    *,
    train_pred: np.ndarray,
    y_train: np.ndarray,
    val_pred: np.ndarray,
) -> np.ndarray:
    """Apply a final affine calibration fitted on training predictions.

    If the calibration fit is degenerate, the original validation predictions
    are returned unchanged on the same float32 scale.
    """

    train_x = train_pred.reshape(-1).astype(np.float64)
    train_y = y_train.reshape(-1).astype(np.float64)
    val_x = val_pred.reshape(-1).astype(np.float64)
    if train_x.size < 2 or train_y.size != train_x.size or float(np.std(train_x)) < 1e-8:
        return val_pred.reshape(-1).astype(np.float32)

    design = np.column_stack([train_x, np.ones_like(train_x)])
    slope, intercept = np.linalg.lstsq(design, train_y, rcond=None)[0]
    if not (math.isfinite(float(slope)) and math.isfinite(float(intercept))):
        return val_pred.reshape(-1).astype(np.float32)

    return (val_x * float(slope) + float(intercept)).astype(np.float32)
