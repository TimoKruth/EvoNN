"""Benchmark evaluator with trainable neural readout heads for Stratograph."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from stratograph.benchmarks.spec import BenchmarkSpec
from stratograph.genome import HierarchicalGenome
from stratograph.runtime import compile_genome


@dataclass(frozen=True)
class TrainingArtifact:
    task: str
    model_name: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class EvaluationRecord:
    metric_value: float
    quality: float
    parameter_count: int
    train_seconds: float
    architecture_summary: str
    genome_id: str
    status: str
    failure_reason: str | None = None


@dataclass(frozen=True)
class EvaluationOutcome:
    record: EvaluationRecord
    training_artifact: TrainingArtifact | None = None


def evaluate_candidate(
    genome: HierarchicalGenome,
    spec: BenchmarkSpec,
    *,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    inherited_state: TrainingArtifact | None = None,
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> EvaluationRecord:
    """Backward-compatible wrapper."""
    return evaluate_candidate_with_state(
        genome,
        spec,
        data=data,
        inherited_state=inherited_state,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    ).record


def evaluate_candidate_with_state(
    genome: HierarchicalGenome,
    spec: BenchmarkSpec,
    *,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    inherited_state: TrainingArtifact | None = None,
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> EvaluationOutcome:
    """Evaluate one candidate and return optional training artifact."""
    started = time.perf_counter()
    compiled = compile_genome(genome)
    x_train, y_train, x_val, y_val = _cap_data(spec, data)
    training_artifact: TrainingArtifact | None = None
    parameter_count = compiled.parameter_count()

    if spec.task == "language_modeling":
        metric_value, training_artifact, head_params = _evaluate_language_modeling(
            compiled,
            genome,
            x_train,
            y_train,
            x_val,
            y_val,
            inherited_state=inherited_state,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        quality = -metric_value
        parameter_count += head_params
    else:
        train_features = compiled.encode(x_train).reshape(x_train.shape[0], -1)
        val_features = compiled.encode(x_val).reshape(x_val.shape[0], -1)
        predictions, training_artifact, head_params = _predict_classification(
            spec=spec,
            genome=genome,
            train_features=train_features,
            y_train=y_train,
            val_features=val_features,
            y_val=y_val,
            inherited_state=inherited_state,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        metric_value = float(accuracy_score(y_val, predictions))
        quality = metric_value
        parameter_count += head_params

    elapsed = time.perf_counter() - started
    return EvaluationOutcome(
        record=EvaluationRecord(
            metric_value=metric_value,
            quality=quality,
            parameter_count=parameter_count,
            train_seconds=elapsed,
            architecture_summary=compiled.architecture_summary(),
            genome_id=genome.genome_id,
            status="ok",
        ),
        training_artifact=training_artifact,
    )


def _predict_classification(
    *,
    spec: BenchmarkSpec,
    genome: HierarchicalGenome,
    train_features: np.ndarray,
    y_train: np.ndarray,
    val_features: np.ndarray,
    y_val: np.ndarray,
    inherited_state: TrainingArtifact | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[np.ndarray, TrainingArtifact | None, int]:
    if len(np.unique(y_train)) <= 1:
        fill = y_train[0] if len(y_train) else 0
        return np.full_like(y_val, fill_value=fill), None, 0

    fit_x, holdout_x, fit_y, holdout_y = train_test_split(
        train_features,
        y_train,
        test_size=0.2 if len(y_train) >= 40 else 0.25,
        random_state=42,
        stratify=y_train,
    )
    scaler = StandardScaler(with_mean=True, with_std=True)
    fit_x_s = scaler.fit_transform(fit_x).astype(np.float32)
    holdout_x_s = scaler.transform(holdout_x).astype(np.float32)
    train_x_s = scaler.transform(train_features).astype(np.float32)
    val_x_s = scaler.transform(val_features).astype(np.float32)

    classes = np.unique(y_train)
    class_to_index = {int(label): index for index, label in enumerate(classes.tolist())}
    fit_y_i = np.asarray([class_to_index[int(label)] for label in fit_y], dtype=np.int64)
    holdout_y_i = np.asarray([class_to_index[int(label)] for label in holdout_y], dtype=np.int64)
    hidden_dim = _classifier_hidden_dim(spec, genome, fit_x_s.shape[1], len(classes))

    params = _init_classifier_params(
        feature_dim=int(fit_x_s.shape[1]),
        hidden_dim=hidden_dim,
        num_classes=len(classes),
        inherited_state=inherited_state,
    )
    best_params = _copy_params(params)
    best_score = float("-inf")
    train_steps = max(10, epochs * (12 if spec.source == "image" else 10))
    step_lr = max(0.0025, learning_rate * (4.0 if spec.source == "image" else 6.0))

    for epoch_index in range(train_steps):
        for batch_x, batch_y in _iter_batches(fit_x_s, fit_y_i, batch_size=max(16, batch_size), seed=epoch_index):
            grads = _classifier_grads(batch_x, batch_y, params, weight_decay=1e-4)
            _apply_grads(params, grads, step_lr)
        holdout_pred = _classifier_predict(holdout_x_s, params, classes)
        holdout_score = float(accuracy_score(holdout_y, holdout_pred))
        if holdout_score >= best_score:
            best_score = holdout_score
            best_params = _copy_params(params)

    train_pred = _classifier_predict(val_x_s, best_params, classes)
    artifact = TrainingArtifact(
        task="classification",
        model_name="neural_classifier",
        payload={
            "feature_dim": int(train_x_s.shape[1]),
            "hidden_dim": hidden_dim,
            "num_classes": len(classes),
            "classes": classes.astype(np.int64),
            "w1": best_params["w1"].astype(np.float32),
            "b1": best_params["b1"].astype(np.float32),
            "w2": best_params["w2"].astype(np.float32),
            "b2": best_params["b2"].astype(np.float32),
        },
    )
    head_params = int(best_params["w1"].size + best_params["b1"].size + best_params["w2"].size + best_params["b2"].size)
    return train_pred, artifact, head_params


def _evaluate_language_modeling(
    compiled,
    genome: HierarchicalGenome,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    inherited_state: TrainingArtifact | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[float, TrainingArtifact | None, int]:
    train_encoded = compiled.encode(x_train)
    val_encoded = compiled.encode(x_val)
    train_repr = train_encoded.reshape(-1, train_encoded.shape[-1]).astype(np.float32)
    val_repr = val_encoded.reshape(-1, val_encoded.shape[-1]).astype(np.float32)
    train_targets = y_train.reshape(-1).astype(np.int64)
    val_targets = y_val.reshape(-1).astype(np.int64)
    if len(train_repr) > 24_576:
        train_repr = train_repr[:24_576]
        train_targets = train_targets[:24_576]
    if len(val_repr) > 8_192:
        val_repr = val_repr[:8_192]
        val_targets = val_targets[:8_192]

    feature_dim = int(train_repr.shape[1])
    vocab_size = int(max(train_targets.max(initial=0), val_targets.max(initial=0), genome.output_dim - 1) + 1)
    params = _init_lm_params(
        feature_dim=feature_dim,
        vocab_size=vocab_size,
        inherited_state=inherited_state,
        train_targets=train_targets,
    )
    best_params = _copy_params(params)
    best_loss = float("inf")
    train_steps = max(8, epochs * 10)
    step_lr = max(0.002, learning_rate * 10.0)

    for epoch_index in range(train_steps):
        for batch_x, batch_y in _iter_batches(train_repr, train_targets, batch_size=max(64, batch_size * 2), seed=epoch_index):
            grads = _lm_grads(batch_x, batch_y, params, weight_decay=5e-5)
            _apply_grads(params, grads, step_lr)
        loss = _lm_loss(val_repr, val_targets, params)
        if loss <= best_loss:
            best_loss = loss
            best_params = _copy_params(params)

    metric_value = float(np.exp(min(20.0, _lm_loss(val_repr, val_targets, best_params))))
    artifact = TrainingArtifact(
        task="language_modeling",
        model_name="neural_lm_head",
        payload={
            "feature_dim": feature_dim,
            "vocab_size": vocab_size,
            "w": best_params["w"].astype(np.float32),
            "b": best_params["b"].astype(np.float32),
        },
    )
    head_params = int(best_params["w"].size + best_params["b"].size)
    return metric_value, artifact, head_params


def _classifier_hidden_dim(spec: BenchmarkSpec, genome: HierarchicalGenome, feature_dim: int, num_classes: int) -> int:
    base = 24 if spec.source != "image" else 48
    depth_bonus = int((genome.macro_depth - 1) * 8 + genome.reuse_ratio * 16)
    return max(16, min(128, base + depth_bonus, max(num_classes * 8, min(feature_dim, 128))))


def _init_classifier_params(
    *,
    feature_dim: int,
    hidden_dim: int,
    num_classes: int,
    inherited_state: TrainingArtifact | None,
) -> dict[str, np.ndarray]:
    if (
        inherited_state is not None
        and inherited_state.model_name == "neural_classifier"
        and int(inherited_state.payload.get("feature_dim", -1)) == feature_dim
        and int(inherited_state.payload.get("hidden_dim", -1)) == hidden_dim
        and int(inherited_state.payload.get("num_classes", -1)) == num_classes
    ):
        return {
            "w1": np.asarray(inherited_state.payload["w1"], dtype=np.float32).copy(),
            "b1": np.asarray(inherited_state.payload["b1"], dtype=np.float32).copy(),
            "w2": np.asarray(inherited_state.payload["w2"], dtype=np.float32).copy(),
            "b2": np.asarray(inherited_state.payload["b2"], dtype=np.float32).copy(),
        }
    rng = np.random.default_rng(42 + feature_dim + hidden_dim + num_classes)
    return {
        "w1": (rng.normal(scale=1.0 / max(1, feature_dim) ** 0.5, size=(feature_dim, hidden_dim))).astype(np.float32),
        "b1": np.zeros(hidden_dim, dtype=np.float32),
        "w2": (rng.normal(scale=1.0 / max(1, hidden_dim) ** 0.5, size=(hidden_dim, num_classes))).astype(np.float32),
        "b2": np.zeros(num_classes, dtype=np.float32),
    }


def _init_lm_params(
    *,
    feature_dim: int,
    vocab_size: int,
    inherited_state: TrainingArtifact | None,
    train_targets: np.ndarray,
) -> dict[str, np.ndarray]:
    if (
        inherited_state is not None
        and inherited_state.model_name == "neural_lm_head"
        and int(inherited_state.payload.get("feature_dim", -1)) == feature_dim
        and int(inherited_state.payload.get("vocab_size", -1)) == vocab_size
    ):
        return {
            "w": np.asarray(inherited_state.payload["w"], dtype=np.float32).copy(),
            "b": np.asarray(inherited_state.payload["b"], dtype=np.float32).copy(),
        }
    rng = np.random.default_rng(123 + feature_dim + vocab_size)
    bias = np.zeros(vocab_size, dtype=np.float32)
    counts = np.bincount(train_targets, minlength=vocab_size).astype(np.float32)
    bias[:] = np.log((counts + 1.0) / max(1.0, counts.sum() + vocab_size))
    return {
        "w": (rng.normal(scale=1.0 / max(1, feature_dim) ** 0.5, size=(feature_dim, vocab_size))).astype(np.float32),
        "b": bias,
    }


def _classifier_predict(x: np.ndarray, params: dict[str, np.ndarray], classes: np.ndarray) -> np.ndarray:
    hidden = _gelu(x @ params["w1"] + params["b1"])
    logits = hidden @ params["w2"] + params["b2"]
    return classes[np.argmax(logits, axis=1)]


def _classifier_grads(
    x: np.ndarray,
    y: np.ndarray,
    params: dict[str, np.ndarray],
    *,
    weight_decay: float,
) -> dict[str, np.ndarray]:
    hidden_pre = x @ params["w1"] + params["b1"]
    hidden = _gelu(hidden_pre)
    logits = hidden @ params["w2"] + params["b2"]
    probs = _softmax(logits)
    probs[np.arange(len(y)), y] -= 1.0
    probs /= max(1, len(y))

    grads_w2 = hidden.T @ probs + weight_decay * params["w2"]
    grads_b2 = probs.sum(axis=0)
    hidden_grad = (probs @ params["w2"].T) * _gelu_grad(hidden_pre)
    grads_w1 = x.T @ hidden_grad + weight_decay * params["w1"]
    grads_b1 = hidden_grad.sum(axis=0)
    return {"w1": grads_w1, "b1": grads_b1, "w2": grads_w2, "b2": grads_b2}


def _lm_grads(
    x: np.ndarray,
    y: np.ndarray,
    params: dict[str, np.ndarray],
    *,
    weight_decay: float,
) -> dict[str, np.ndarray]:
    logits = x @ params["w"] + params["b"]
    probs = _softmax(logits)
    probs[np.arange(len(y)), y] -= 1.0
    probs /= max(1, len(y))
    grads_w = x.T @ probs + weight_decay * params["w"]
    grads_b = probs.sum(axis=0)
    return {"w": grads_w, "b": grads_b}


def _lm_loss(x: np.ndarray, y: np.ndarray, params: dict[str, np.ndarray]) -> float:
    logits = x @ params["w"] + params["b"]
    logits = logits - logits.max(axis=1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True) + 1e-12)
    return float(-np.mean(log_probs[np.arange(len(y)), y]))


def _apply_grads(params: dict[str, np.ndarray], grads: dict[str, np.ndarray], learning_rate: float) -> None:
    for key, grad in grads.items():
        params[key] -= learning_rate * grad.astype(np.float32)


def _copy_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype=np.float32).copy() for key, value in params.items()}


def _iter_batches(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    seed: int,
):
    rng = np.random.default_rng(10_000 + seed)
    order = rng.permutation(len(x))
    for start in range(0, len(order), batch_size):
        index = order[start : start + batch_size]
        yield x[index], y[index]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted).astype(np.float32)
    return exp / (exp.sum(axis=1, keepdims=True) + 1e-12)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x**3))))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    tanh_term = np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x**3)))
    sech2 = 1.0 - tanh_term**2
    inner_grad = np.sqrt(2.0 / np.pi) * (1.0 + 0.134145 * (x**2))
    return 0.5 * (1.0 + tanh_term) + 0.5 * x * sech2 * inner_grad


def _cap_data(
    spec: BenchmarkSpec,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, y_train, x_val, y_val = data
    if spec.task == "language_modeling":
        train_cap = min(len(x_train), spec.max_train_samples or 2048)
        val_cap = min(len(x_val), spec.max_val_samples or 512)
        return (
            x_train[:train_cap],
            y_train[:train_cap],
            x_val[:val_cap],
            y_val[:val_cap],
        )

    train_cap = 2048 if spec.source in {"openml", "image"} else 1024
    val_cap = 1024 if spec.source in {"openml", "image"} else 512
    return (
        x_train[: min(len(x_train), train_cap)],
        y_train[: min(len(y_train), train_cap)],
        x_val[: min(len(x_val), val_cap)],
        y_val[: min(len(y_val), val_cap)],
    )
