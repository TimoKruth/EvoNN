"""Training loop for compiled MLX models."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from evonn_shared.metrics import compute_task_metric, metric_name_for_task
from evonn_shared.training import (
    calibrate_regression_predictions,
    regression_target_stats,
    restore_regression_predictions,
    standardize_regression_targets,
)


@dataclass
class EvaluationResult:
    metric_name: str
    metric_value: float
    quality: float  # normalized [0, 1] for classification; raw for regression
    parameter_count: int
    train_seconds: float
    failure_reason: str | None = None
    inherited_from: str | None = None


def cosine_lr(base_lr: float, step: int, total_steps: int, min_lr: float = 1e-6) -> float:
    """Cosine annealing learning rate schedule."""
    if total_steps <= 0:
        return base_lr
    progress = min(1.0, step / max(1, total_steps))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _batch_indices(total: int, batch_size: int):
    """Yield slices for mini-batch iteration."""
    for start in range(0, total, batch_size):
        yield slice(start, min(total, start + batch_size))


def _clip_grad_norm(grads, max_norm: float):
    """Clip gradient norms to prevent exploding gradients."""
    import mlx.core as mx
    import mlx.utils

    leaves = mlx.utils.tree_flatten(grads)
    total_sq = 0.0
    for _, g in leaves:
        total_sq += mx.sum(mx.square(g)).item()
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads
    scale = max_norm / (total_norm + 1e-6)
    return mlx.utils.tree_map(lambda g: g * scale, grads)


def _loss_fn(model, task: str, x, y):
    """Compute loss for classification, regression, or language modeling."""
    import mlx.nn as nn

    logits = model(x)
    if task == "classification":
        return nn.losses.cross_entropy(logits, y, reduction="mean")
    if task == "language_modeling":
        if logits.ndim == 2:
            return nn.losses.cross_entropy(logits, y, reduction="mean")
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = y.reshape(-1)
        return nn.losses.cross_entropy(logits_flat, targets_flat, reduction="mean")
    return nn.losses.mse_loss(logits.reshape(-1), y.reshape(-1), reduction="mean")


def _compute_metric(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, float]:
    """Compute (metric_name, metric_value, quality) for the given task.

    For classification: accuracy in [0, 1] (higher = better quality).
    For language modeling: perplexity (lower = better); quality = -perplexity.
    For regression: MSE (lower = better); quality = -MSE so higher = better.
    """
    metric = compute_task_metric(task, y_true, y_pred)
    return metric.metric_name, metric.metric_value, metric.quality


def train_and_evaluate(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    task: str,
    epochs: int,
    lr: float,
    batch_size: int,
    lr_schedule: str = "cosine",
    grad_clip_norm: float = 1.0,
    weight_decay: float = 0.0,
    early_stopping_patience: int = 3,
    parameter_count: int = 0,
) -> EvaluationResult:
    """Train an MLX model and return evaluation result.

    Args:
        model: Compiled MLX model (nn.Module).
        X_train / y_train: Training data as numpy arrays.
        X_val / y_val: Validation data as numpy arrays.
        task: "classification", "regression", or "language_modeling".
        epochs: Maximum training epochs.
        lr: Base learning rate.
        batch_size: Mini-batch size.
        lr_schedule: "cosine" or "constant".
        grad_clip_norm: Max gradient norm (0 to disable).
        weight_decay: L2 regularization coefficient.
        early_stopping_patience: Stop after N epochs without improvement.
        parameter_count: Pre-computed parameter count for reporting.

    Returns:
        EvaluationResult with metric value and quality score.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    start_time = time.perf_counter()

    try:
        # Setup optimizer
        optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

        # Determine dtypes. Regression trains on standardized targets but
        # reports metrics on the original target scale.
        regression_target_mean = 0.0
        regression_target_std = 1.0
        y_train_for_loss = y_train
        if task == "regression":
            regression_target_mean, regression_target_std = regression_target_stats(y_train)
            y_train_for_loss = standardize_regression_targets(
                y_train,
                regression_target_mean,
                regression_target_std,
            )

        y_dtype = np.int32 if task in {"classification", "language_modeling"} else np.float32
        x_dtype = np.int32 if task == "language_modeling" else np.float32
        x_train = mx.array(X_train.astype(x_dtype))
        y_train_t = mx.array(y_train_for_loss.astype(y_dtype))
        x_val = mx.array(X_val.astype(x_dtype))

        # Setup loss+grad function
        loss_and_grad = nn.value_and_grad(
            model,
            lambda m, x, y: _loss_fn(m, task, x, y),
        )

        # Compute total steps for LR scheduling
        n_train = X_train.shape[0]
        steps_per_epoch = max(1, (n_train + batch_size - 1) // batch_size)
        total_steps = steps_per_epoch * epochs
        use_cosine = lr_schedule == "cosine"
        use_clip = grad_clip_norm > 0

        # Training loop
        model.train()
        best_val_quality = float("-inf")
        epochs_without_improvement = 0
        global_step = 0

        for epoch in range(epochs):
            for batch in _batch_indices(n_train, batch_size):
                # Update learning rate
                if use_cosine:
                    current_lr = cosine_lr(lr, global_step, total_steps)
                    optimizer.learning_rate = current_lr

                x_batch = x_train[batch]
                y_batch = y_train_t[batch]

                loss, grads = loss_and_grad(model, x_batch, y_batch)

                # Check for NaN loss
                if math.isnan(float(loss.item())):
                    return EvaluationResult(
                        metric_name=_metric_name(task),
                        metric_value=float("nan"),
                        quality=float("-inf"),
                        parameter_count=parameter_count,
                        train_seconds=time.perf_counter() - start_time,
                        failure_reason="nan_loss",
                    )

                # Gradient clipping
                if use_clip:
                    grads = _clip_grad_norm(grads, grad_clip_norm)

                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                global_step += 1

            # Validation
            model.eval()
            val_preds = model(x_val)
            mx.eval(val_preds)
            val_preds_np = np.array(val_preds)
            if task == "regression":
                val_preds_np = restore_regression_predictions(
                    val_preds_np,
                    regression_target_mean,
                    regression_target_std,
                )
            _, _, val_quality = _compute_metric(task, y_val, val_preds_np)
            model.train()

            # Early stopping
            if val_quality > best_val_quality + 1e-9:
                best_val_quality = val_quality
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                early_stopping_patience >= 0
                and epochs_without_improvement >= early_stopping_patience
                and epoch >= 1  # train at least 2 epochs
            ):
                break

        # Final evaluation on validation set
        model.eval()
        train_preds_np = None
        if task == "regression":
            train_preds = model(x_train)
            mx.eval(train_preds)
            train_preds_np = restore_regression_predictions(
                np.array(train_preds),
                regression_target_mean,
                regression_target_std,
            )
        val_preds = model(x_val)
        mx.eval(val_preds)
        val_preds_np = np.array(val_preds)
        if task == "regression":
            val_preds_np = restore_regression_predictions(
                val_preds_np,
                regression_target_mean,
                regression_target_std,
            )
            if train_preds_np is not None:
                val_preds_np = calibrate_regression_predictions(
                    train_pred=train_preds_np,
                    y_train=y_train,
                    val_pred=val_preds_np,
                )
        metric_name, metric_value, quality = _compute_metric(task, y_val, val_preds_np)

        return EvaluationResult(
            metric_name=metric_name,
            metric_value=metric_value,
            quality=quality,
            parameter_count=parameter_count,
            train_seconds=time.perf_counter() - start_time,
        )

    except Exception as exc:
        return EvaluationResult(
            metric_name=_metric_name(task),
            metric_value=float("nan"),
            quality=float("-inf"),
            parameter_count=parameter_count,
            train_seconds=time.perf_counter() - start_time,
            failure_reason=_format_failure(exc),
        )


def _metric_name(task: str) -> str:
    return metric_name_for_task(task)


def _format_failure(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    if len(message) > 96:
        message = f"{message[:93]}..."
    return f"runtime_error:{type(exc).__name__}:{message}"
