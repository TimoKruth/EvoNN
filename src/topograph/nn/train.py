"""Training loop, fitness evaluation, and weight snapshot utilities."""

import math
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np

from topograph.genome.genome import INPUT_INNOVATION, OUTPUT_INNOVATION, Genome


@dataclass
class EvaluationResult:
    metric_name: str
    metric_direction: str
    metric_value: float
    quality: float
    native_fitness: float
    train_seconds: float
    failure_reason: str | None = None


# ---------------------------------------------------------------------------
# LR schedules
# ---------------------------------------------------------------------------

def cosine_lr(base_lr: float, step: int, total_steps: int) -> float:
    """Cosine annealing with linear warmup (first 10% of steps)."""
    warmup_steps = int(total_steps * 0.1)
    min_lr = base_lr / 100.0
    if step < warmup_steps:
        start_lr = base_lr / 10.0
        return start_lr + (base_lr - start_lr) * (step / max(warmup_steps, 1))
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 0:
        return base_lr
    progress = min((step - warmup_steps) / decay_steps, 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def _fixed_lr(base_lr: float, step: int, total_steps: int) -> float:
    return base_lr


_SCHEDULES = {"cosine": cosine_lr, "fixed": _fixed_lr}


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _classification_loss(model: nn.Module, X: mx.array, y: mx.array) -> mx.array:
    probs = model(X)
    loss = -mx.mean(mx.log(probs[mx.arange(y.shape[0]), y] + 1e-8))
    if hasattr(model, "moe") and model.moe is not None:
        loss = loss + model._load_balance_weight * model.moe.load_balance_loss()
    return loss


def _language_modeling_loss(model: nn.Module, X: mx.array, y: mx.array) -> mx.array:
    probs = model(X)
    probs = mx.clip(probs, 1e-8, 1.0)

    if probs.ndim == 3 and y.ndim == 2:
        batch_size, seq_len, vocab_size = probs.shape
        probs_flat = probs.reshape(batch_size * seq_len, vocab_size)
        targets_flat = y.reshape(batch_size * seq_len)
        loss = -mx.mean(mx.log(probs_flat[mx.arange(targets_flat.shape[0]), targets_flat]))
    else:
        if probs.ndim == 3:
            probs = probs[:, -1, :]
        loss = -mx.mean(mx.log(probs[mx.arange(y.shape[0]), y] + 1e-8))

    if hasattr(model, "moe") and model.moe is not None:
        loss = loss + model._load_balance_weight * model.moe.load_balance_loss()
    return loss


def _regression_loss(model: nn.Module, X: mx.array, y: mx.array) -> mx.array:
    preds = model(X)
    loss = mx.mean((preds.squeeze() - y) ** 2)
    if hasattr(model, "moe") and model.moe is not None:
        loss = loss + model._load_balance_weight * model.moe.load_balance_loss()
    return loss


def _compute_metric(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[str, str, float, float]:
    """Return canonical cross-system metric fields plus optimizer quality."""
    if task == "classification":
        if y_pred.ndim == 2 and y_pred.shape[1] > 1:
            preds = np.argmax(y_pred, axis=1)
        else:
            preds = (y_pred.ravel() > 0.5).astype(int)
        y_true_flat = y_true.ravel().astype(int)
        accuracy = float(np.mean(preds == y_true_flat))
        return "accuracy", "max", accuracy, accuracy
    if task == "language_modeling":
        probs = np.clip(y_pred, 1e-8, 1.0)
        if probs.ndim == 3 and y_true.ndim == 2:
            batch_size, seq_len, vocab_size = probs.shape
            probs_flat = probs.reshape(batch_size * seq_len, vocab_size)
            targets_flat = y_true.reshape(batch_size * seq_len).astype(int)
            cross_entropy = float(
                -np.mean(np.log(probs_flat[np.arange(targets_flat.shape[0]), targets_flat]))
            )
        else:
            if probs.ndim == 3:
                probs = probs[:, -1, :]
            targets_flat = y_true.ravel().astype(int)
            cross_entropy = float(
                -np.mean(np.log(probs[np.arange(targets_flat.shape[0]), targets_flat]))
            )

        perplexity = float(np.exp(np.clip(cross_entropy, -20.0, 20.0)))
        return "perplexity", "min", perplexity, -perplexity

    y_pred_flat = y_pred.ravel()
    y_true_flat = y_true.ravel()
    mse = float(np.mean((y_pred_flat - y_true_flat) ** 2))
    return "mse", "min", mse, -mse


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

def _clip_grad_norm(grads: dict, max_norm: float) -> dict:
    """Clip gradient dict by global L2 norm."""
    leaves = mlx.utils.tree_flatten(grads)
    total_norm_sq = sum(mx.sum(g * g).item() for _, g in leaves)
    total_norm = total_norm_sq ** 0.5
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        grads = mlx.utils.tree_map(lambda g: g * scale, grads)
    return grads


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    X_train: np.ndarray | mx.array,
    y_train: np.ndarray | mx.array,
    X_val: np.ndarray | mx.array,
    y_val: np.ndarray | mx.array,
    epochs: int,
    lr: float,
    batch_size: int,
    task: str = "classification",
    lr_schedule: str = "cosine",
    weight_decay: float = 0.01,
    grad_clip_norm: float = 1.0,
    divergence_threshold: float = 10.0,
    patience: int = 2,
) -> float:
    """Train model and return final validation loss.

    Supports cosine/fixed LR schedule, AdamW, gradient clipping, and
    early stopping on divergence or plateau.
    """
    return train_and_evaluate(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        task=task,
        lr_schedule=lr_schedule,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        divergence_threshold=divergence_threshold,
        patience=patience,
    ).native_fitness


def train_and_evaluate(
    model: nn.Module,
    X_train: np.ndarray | mx.array,
    y_train: np.ndarray | mx.array,
    X_val: np.ndarray | mx.array,
    y_val: np.ndarray | mx.array,
    epochs: int,
    lr: float,
    batch_size: int,
    task: str = "classification",
    lr_schedule: str = "cosine",
    weight_decay: float = 0.01,
    grad_clip_norm: float = 1.0,
    divergence_threshold: float = 10.0,
    patience: int = 2,
) -> EvaluationResult:
    """Train model and return canonical metric fields plus native optimizer fitness."""
    start_time = time.perf_counter()

    X = X_train if isinstance(X_train, mx.array) else mx.array(X_train)
    y = y_train if isinstance(y_train, mx.array) else mx.array(y_train)
    X_v = X_val if isinstance(X_val, mx.array) else mx.array(X_val)
    y_v = y_val if isinstance(y_val, mx.array) else mx.array(y_val)
    n = X.shape[0]

    steps_per_epoch = max(1, (n + batch_size - 1) // batch_size)
    total_steps = steps_per_epoch * epochs
    schedule_fn = _SCHEDULES.get(lr_schedule, cosine_lr)

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
    if task == "regression":
        loss_fn = _regression_loss
    elif task == "language_modeling":
        loss_fn = _language_modeling_loss
    else:
        loss_fn = _classification_loss
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    last_loss = float("inf")
    epoch_losses: list[float] = []
    global_step = 0

    try:
        for _epoch in range(epochs):
            indices = mx.array(np.random.permutation(n))
            for start in range(0, n, batch_size):
                current_lr = schedule_fn(lr, global_step, total_steps)
                optimizer.learning_rate = mx.array(current_lr)

                batch_idx = indices[start : start + batch_size]
                loss_val, grads = loss_and_grad(model, X[batch_idx], y[batch_idx])

                if grad_clip_norm > 0:
                    grads = _clip_grad_norm(grads, grad_clip_norm)

                optimizer.update(model, grads)
                last_loss = loss_val.item()
                global_step += 1

            mx.eval(model.parameters(), optimizer.state)
            epoch_losses.append(last_loss)

            if last_loss > divergence_threshold:
                return EvaluationResult(
                    metric_name=_metric_name_for_task(task),
                    metric_direction=_metric_direction_for_task(task),
                    metric_value=float("nan"),
                    quality=float("-inf"),
                    native_fitness=float("inf"),
                    train_seconds=time.perf_counter() - start_time,
                    failure_reason="diverged",
                )

            if len(epoch_losses) >= patience + 1:
                if all(
                    epoch_losses[-(i + 1)] >= epoch_losses[-(i + 2)]
                    for i in range(patience)
                ):
                    break

            if task == "regression" and len(epoch_losses) >= 3:
                prev, curr = epoch_losses[-3], epoch_losses[-1]
                if prev > 0 and (prev - curr) / (abs(prev) + 1e-10) < 0.01:
                    break

        preds = model(X_v)
        mx.eval(preds)
        preds_np = np.array(preds)

        if task == "regression":
            native_fitness = mx.mean((preds.squeeze() - y_v) ** 2).item()
        elif task == "language_modeling":
            probs = mx.clip(preds, 1e-8, 1.0)
            if probs.ndim == 3 and y_v.ndim == 2:
                batch_size, seq_len, vocab_size = probs.shape
                probs_flat = probs.reshape(batch_size * seq_len, vocab_size)
                targets_flat = y_v.reshape(batch_size * seq_len)
                native_fitness = -mx.mean(
                    mx.log(probs_flat[mx.arange(targets_flat.shape[0]), targets_flat])
                ).item()
            else:
                if probs.ndim == 3:
                    probs = probs[:, -1, :]
                native_fitness = -mx.mean(
                    mx.log(probs[mx.arange(y_v.shape[0]), y_v] + 1e-8)
                ).item()
        else:
            probs = preds
            native_fitness = -mx.mean(
                mx.log(probs[mx.arange(y_v.shape[0]), y_v] + 1e-8)
            ).item()

        metric_name, metric_direction, metric_value, quality = _compute_metric(
            task,
            np.array(y_val),
            preds_np,
        )
        return EvaluationResult(
            metric_name=metric_name,
            metric_direction=metric_direction,
            metric_value=metric_value,
            quality=quality,
            native_fitness=float(native_fitness),
            train_seconds=time.perf_counter() - start_time,
        )
    except Exception as exc:
        return EvaluationResult(
            metric_name=_metric_name_for_task(task),
            metric_direction=_metric_direction_for_task(task),
            metric_value=float("nan"),
            quality=float("-inf"),
            native_fitness=float("inf"),
            train_seconds=time.perf_counter() - start_time,
            failure_reason=f"runtime_error:{type(exc).__name__}",
        )


def _metric_name_for_task(task: str) -> str:
    if task == "language_modeling":
        return "perplexity"
    if task == "regression":
        return "mse"
    return "accuracy"


def _metric_direction_for_task(task: str) -> str:
    return "min" if task in {"regression", "language_modeling"} else "max"


# ---------------------------------------------------------------------------
# Percentile fitness
# ---------------------------------------------------------------------------

def compute_percentile_fitness(
    all_results: dict[str, list[float]],
) -> list[float]:
    """Per-benchmark fractional ranks, averaged across benchmarks.

    Returns a list of mean percentile values (lower = better), one per genome.
    Ties receive the same rank.
    """
    if not all_results:
        return []
    pop_size = len(next(iter(all_results.values())))

    all_percentiles: list[list[float]] = []
    for losses in all_results.values():
        sorted_idx = sorted(range(pop_size), key=lambda i: losses[i])
        ranks = [0.0] * pop_size
        i = 0
        while i < pop_size:
            j = i + 1
            while j < pop_size and losses[sorted_idx[j]] == losses[sorted_idx[i]]:
                j += 1
            mean_rank = sum(range(i, j)) / (j - i)
            for k in range(i, j):
                ranks[sorted_idx[k]] = mean_rank / max(pop_size - 1, 1)
            i = j
        all_percentiles.append(ranks)

    return [
        sum(p[i] for p in all_percentiles) / len(all_percentiles)
        for i in range(pop_size)
    ]


# ---------------------------------------------------------------------------
# Model size estimation
# ---------------------------------------------------------------------------

def effective_model_bytes(genome: Genome, input_dim: int = 0, num_classes: int = 0) -> int:
    """Precision-aware byte count: per-connection accounting with layer widths, bits, sparsity."""
    layer_map = {g.innovation: g for g in genome.enabled_layers}
    total = 0
    for conn in genome.enabled_connections:
        tgt_inn = conn.target
        if tgt_inn == OUTPUT_INNOVATION:
            tgt_width = num_classes
            bits = 16.0
            sparsity = 0.0
        else:
            tgt_lg = layer_map.get(tgt_inn)
            if tgt_lg is None:
                continue
            tgt_width = tgt_lg.width
            bits = 1.58 if tgt_lg.weight_bits.value == 2 else float(tgt_lg.weight_bits.value)
            sparsity = tgt_lg.sparsity

        if conn.source == INPUT_INNOVATION:
            src_width = input_dim
        elif conn.source == OUTPUT_INNOVATION:
            continue
        else:
            src_lg = layer_map.get(conn.source)
            src_width = src_lg.width if src_lg else 0

        density = 1.0 - sparsity
        total += int(src_width * tgt_width * bits * density / 8)
    return total


# ---------------------------------------------------------------------------
# Weight snapshots
# ---------------------------------------------------------------------------

def extract_weights(model: nn.Module) -> dict[str, np.ndarray]:
    """Snapshot model parameters to NumPy arrays."""
    return {name: np.array(val) for name, val in mlx.utils.tree_flatten(model.parameters())}


def load_weight_snapshot(model: nn.Module, snapshot: dict[str, np.ndarray | mx.array]) -> int:
    """Load subset of weights matching by name and shape. Returns count loaded."""
    current = dict(mlx.utils.tree_flatten(model.parameters()))
    compatible: dict[str, mx.array] = {}
    for name, value in snapshot.items():
        target = current.get(name)
        if target is None:
            continue
        source = mx.array(value)
        if tuple(source.shape) != tuple(target.shape):
            continue
        compatible[name] = source

    if compatible:
        update_tree = _build_update_tree(model.parameters(), compatible)
        model.update(update_tree, strict=False)
    return len(compatible)


def _build_update_tree(reference: dict, flat_weights: dict[str, mx.array]) -> dict:
    tree: dict = {}
    for path, value in flat_weights.items():
        _assign_path(tree, reference, path.split("."), value)
    return tree


def _assign_path(
    target: dict | list, reference: dict | list, parts: list[str], value: mx.array,
) -> None:
    key = parts[0]
    if len(parts) == 1:
        if isinstance(reference, list):
            idx = int(key)
            while len(target) <= idx:
                target.append({})
            target[idx] = value
        else:
            target[key] = value
        return
    if isinstance(reference, list):
        idx = int(key)
        while len(target) <= idx:
            target.append({})
        next_ref = reference[idx]
        if not isinstance(target[idx], (dict, list)):
            target[idx] = [] if isinstance(next_ref, list) else {}
        _assign_path(target[idx], next_ref, parts[1:], value)
    else:
        next_ref = reference[key]
        if key not in target or not isinstance(target[key], (dict, list)):
            target[key] = [] if isinstance(next_ref, list) else {}
        _assign_path(target[key], next_ref, parts[1:], value)
