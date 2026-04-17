"""Training loop for compiled MLX models."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationResult:
    metric_name: str
    metric_value: float
    quality: float  # normalized [0, 1] for classification; raw for regression
    parameter_count: int
    train_seconds: float
    failure_reason: str | None = None


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
    return nn.losses.mse_loss(logits, y, reduction="mean")


def _compute_metric(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, float]:
    """Compute (metric_name, metric_value, quality) for the given task.

    For classification: accuracy in [0, 1] (higher = better quality).
    For language modeling: perplexity (lower = better); quality = -perplexity.
    For regression: MSE (lower = better); quality = -MSE so higher = better.
    """
    if task == "classification":
        if y_pred.ndim == 2 and y_pred.shape[1] > 1:
            preds = np.argmax(y_pred, axis=1)
        else:
            preds = (y_pred.ravel() > 0.5).astype(int)
        y_true_flat = y_true.ravel().astype(int)
        accuracy = float(np.mean(preds == y_true_flat))
        return "accuracy", accuracy, accuracy
    if task == "language_modeling":
        logits = y_pred
        if logits.ndim == 2:
            logits = logits[:, None, :]
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        probs = np.clip(probs, 1e-8, 1.0)

        probs_flat = probs.reshape(-1, probs.shape[-1])
        targets_flat = y_true.reshape(-1).astype(int)
        cross_entropy = float(-np.mean(np.log(probs_flat[np.arange(targets_flat.shape[0]), targets_flat])))
        perplexity = float(np.exp(np.clip(cross_entropy, -20.0, 20.0)))
        return "perplexity", perplexity, -perplexity
    else:
        y_pred_flat = y_pred.ravel()
        y_true_flat = y_true.ravel()
        mse = float(np.mean((y_pred_flat - y_true_flat) ** 2))
        return "mse", mse, -mse


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

        # Determine dtypes
        y_dtype = np.int32 if task in {"classification", "language_modeling"} else np.float32
        x_dtype = np.int32 if task == "language_modeling" else np.float32
        x_train = mx.array(X_train.astype(x_dtype))
        y_train_t = mx.array(y_train.astype(y_dtype))
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
        val_preds = model(x_val)
        mx.eval(val_preds)
        val_preds_np = np.array(val_preds)
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
    if task == "classification":
        return "accuracy"
    if task == "language_modeling":
        return "perplexity"
    return "mse"


def _format_failure(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    if len(message) > 96:
        message = f"{message[:93]}..."
    return f"runtime_error:{type(exc).__name__}:{message}"
