from __future__ import annotations

import numpy as np

from evonn_shared.training import (
    calibrate_regression_predictions,
    regression_target_stats,
    restore_regression_predictions,
    standardize_regression_targets,
)


def test_regression_target_round_trip_preserves_original_scale() -> None:
    y_train = np.array([10.0, 12.0, 14.0, 16.0], dtype=np.float32)

    mean, std = regression_target_stats(y_train)
    standardized = standardize_regression_targets(y_train, mean, std)
    restored = restore_regression_predictions(standardized, mean, std)

    assert np.allclose(restored, y_train)


def test_regression_target_stats_guard_degenerate_std() -> None:
    mean, std = regression_target_stats(np.array([7.0, 7.0, 7.0], dtype=np.float32))

    assert mean == 7.0
    assert std == 1.0


def test_calibrate_regression_predictions_fits_affine_mapping() -> None:
    train_pred = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    y_train = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)
    val_pred = np.array([4.0, 5.0], dtype=np.float32)

    calibrated = calibrate_regression_predictions(
        train_pred=train_pred,
        y_train=y_train,
        val_pred=val_pred,
    )

    assert np.allclose(calibrated, np.array([9.0, 11.0], dtype=np.float32))


def test_calibrate_regression_predictions_leaves_degenerate_fit_unchanged() -> None:
    val_pred = np.array([4.0, 5.0], dtype=np.float32)

    calibrated = calibrate_regression_predictions(
        train_pred=np.array([2.0, 2.0, 2.0], dtype=np.float32),
        y_train=np.array([1.0, 3.0, 5.0], dtype=np.float32),
        val_pred=val_pred,
    )

    assert np.array_equal(calibrated, val_pred)
