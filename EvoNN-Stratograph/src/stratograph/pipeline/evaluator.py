"""Benchmark evaluator and lightweight training runtime for Stratograph."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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
) -> EvaluationRecord:
    """Backward-compatible wrapper."""
    return evaluate_candidate_with_state(
        genome,
        spec,
        data=data,
        inherited_state=inherited_state,
    ).record


def evaluate_candidate_with_state(
    genome: HierarchicalGenome,
    spec: BenchmarkSpec,
    *,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    inherited_state: TrainingArtifact | None = None,
) -> EvaluationOutcome:
    """Evaluate one candidate and return optional training artifact."""
    started = time.perf_counter()
    compiled = compile_genome(genome)
    x_train, y_train, x_val, y_val = _cap_data(spec, data)
    training_artifact: TrainingArtifact | None = None
    if spec.task == "language_modeling":
        metric_value = _evaluate_language_modeling(compiled, genome, x_train, y_train, x_val, y_val)
        quality = -metric_value
    else:
        train_features = compiled.encode(x_train).reshape(x_train.shape[0], -1)
        val_features = compiled.encode(x_val).reshape(x_val.shape[0], -1)
        predictions, training_artifact = _predict_classification(
            spec=spec,
            genome=genome,
            train_features=train_features,
            y_train=y_train,
            val_features=val_features,
            y_val=y_val,
            inherited_state=inherited_state,
        )
        metric_value = float(accuracy_score(y_val, predictions))
        quality = metric_value

    elapsed = time.perf_counter() - started
    return EvaluationOutcome(
        record=EvaluationRecord(
            metric_value=metric_value,
            quality=quality,
            parameter_count=compiled.parameter_count(),
            train_seconds=elapsed,
            architecture_summary=compiled.architecture_summary(),
            genome_id=genome.genome_id,
            status="ok",
        ),
        training_artifact=training_artifact,
    )


def _evaluate_language_modeling(
    compiled,
    genome: HierarchicalGenome,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    train_prev = x_train.reshape(-1)
    train_next = y_train.reshape(-1)
    val_prev = x_val.reshape(-1)
    val_next = y_val.reshape(-1)
    vocab_size = int(
        max(
            train_prev.max(initial=0),
            train_next.max(initial=0),
            val_prev.max(initial=0),
            val_next.max(initial=0),
        )
        + 1
    )
    alpha = 0.35 + (genome.average_cell_depth * 0.08) + (genome.reuse_ratio * 0.15)

    unigram = np.bincount(train_next, minlength=vocab_size).astype(np.float64)
    unigram_probs = (unigram + alpha) / (unigram.sum() + alpha * vocab_size)

    bigram_counts: dict[int, Counter[int]] = defaultdict(Counter)
    bigram_totals: Counter[int] = Counter()
    for prev_token, next_token in zip(train_prev.tolist(), train_next.tolist()):
        bigram_counts[prev_token][next_token] += 1
        bigram_totals[prev_token] += 1

    train_ctx2 = x_train[:, -2:].reshape(-1, 2) if x_train.shape[1] >= 2 else np.stack([train_prev, train_prev], axis=1)
    val_ctx2 = x_val[:, -2:].reshape(-1, 2) if x_val.shape[1] >= 2 else np.stack([val_prev, val_prev], axis=1)
    trigram_counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    trigram_totals: Counter[tuple[int, int]] = Counter()
    for context, next_token in zip(train_ctx2.tolist(), train_next.tolist()):
        key = (int(context[0]), int(context[1]))
        trigram_counts[key][next_token] += 1
        trigram_totals[key] += 1

    train_repr = compiled.encode(x_train)[:, -1, :]
    val_repr = compiled.encode(x_val)[:, -1, :]
    train_buckets = _feature_buckets(train_repr)
    val_buckets = _feature_buckets(val_repr)
    bucket_counts: dict[int, Counter[int]] = defaultdict(Counter)
    bucket_totals: Counter[int] = Counter()
    for bucket, next_token in zip(train_buckets.tolist(), train_next.tolist()):
        bucket_counts[int(bucket)][int(next_token)] += 1
        bucket_totals[int(bucket)] += 1

    trigram_mix = min(0.6, 0.35 + genome.reuse_ratio * 0.15 + min(genome.average_cell_depth, 4.0) * 0.03)
    bigram_mix = 0.25
    bucket_mix = min(0.25, 0.1 + genome.reuse_ratio * 0.1)
    unigram_mix = max(0.05, 1.0 - trigram_mix - bigram_mix - bucket_mix)

    log_probs: list[float] = []
    for trigram_context, prev_token, bucket, next_token in zip(
        val_ctx2.tolist(),
        val_prev.tolist(),
        val_buckets.tolist(),
        val_next.tolist(),
    ):
        tri_key = (int(trigram_context[0]), int(trigram_context[1]))
        tri_total = trigram_totals.get(tri_key, 0)
        tri_count = trigram_counts.get(tri_key, Counter()).get(int(next_token), 0)
        tri_prob = (tri_count + alpha) / (tri_total + alpha * vocab_size)

        bi_total = bigram_totals.get(int(prev_token), 0)
        bi_count = bigram_counts.get(int(prev_token), Counter()).get(int(next_token), 0)
        bi_prob = (bi_count + alpha) / (bi_total + alpha * vocab_size)

        bucket_total = bucket_totals.get(int(bucket), 0)
        bucket_count = bucket_counts.get(int(bucket), Counter()).get(int(next_token), 0)
        bucket_prob = (bucket_count + alpha) / (bucket_total + alpha * vocab_size)

        prob = (
            trigram_mix * tri_prob
            + bigram_mix * bi_prob
            + bucket_mix * bucket_prob
            + unigram_mix * unigram_probs[int(next_token)]
        )
        log_probs.append(math.log(max(prob, 1e-12)))
    return float(math.exp(-sum(log_probs) / max(1, len(log_probs))))


def _predict_classification(
    *,
    spec: BenchmarkSpec,
    genome: HierarchicalGenome,
    train_features: np.ndarray,
    y_train: np.ndarray,
    val_features: np.ndarray,
    y_val: np.ndarray,
    inherited_state: TrainingArtifact | None,
) -> tuple[np.ndarray, TrainingArtifact | None]:
    if len(np.unique(y_train)) <= 1:
        return np.full_like(y_val, fill_value=y_train[0] if len(y_train) else 0), None

    test_size = 0.2 if len(y_train) >= 40 else 0.25
    fit_x, holdout_x, fit_y, holdout_y = train_test_split(
        train_features,
        y_train,
        test_size=test_size,
        random_state=42,
        stratify=y_train,
    )
    scaler = StandardScaler(with_mean=True, with_std=True)
    fit_x_s = scaler.fit_transform(fit_x)
    holdout_x_s = scaler.transform(holdout_x)
    train_x_s = scaler.transform(train_features)
    val_x_s = scaler.transform(val_features)

    models: list[tuple[str, Any, np.ndarray, np.ndarray]] = [
        ("ridge", RidgeClassifier(alpha=max(0.1, 1.0 - genome.reuse_ratio)), fit_x_s, holdout_x_s),
        ("lda", LinearDiscriminantAnalysis(), fit_x_s, holdout_x_s),
        (
            "logreg",
            LogisticRegression(
                max_iter=500,
                C=1.0 + genome.reuse_ratio,
                solver="lbfgs",
            ),
            fit_x_s,
            holdout_x_s,
        ),
        (
            "sgd",
            _init_sgd_classifier(
                genome=genome,
                inherited_state=inherited_state,
                feature_dim=int(fit_x_s.shape[1]),
                classes=np.unique(y_train),
            ),
            fit_x_s,
            holdout_x_s,
        ),
    ]
    if spec.source == "image":
        models.extend(
            [
                ("svc", LinearSVC(C=1.0 + genome.reuse_ratio, dual="auto"), fit_x_s, holdout_x_s),
                ("knn", KNeighborsClassifier(n_neighbors=5), fit_x_s, holdout_x_s),
            ]
        )

    best_name = "ridge"
    best_model: Any | None = None
    best_score = float("-inf")
    for name, model, fit_source, holdout_source in models:
        try:
            if name == "sgd":
                _fit_sgd(model, fit_source, fit_y)
            else:
                model.fit(fit_source, fit_y)
            score = accuracy_score(holdout_y, model.predict(holdout_source))
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    if best_model is None:
        fallback = RidgeClassifier(alpha=1.0)
        fallback.fit(train_x_s, y_train)
        return fallback.predict(val_x_s), None

    if best_name == "sgd":
        _fit_sgd(best_model, train_x_s, y_train)
    else:
        best_model.fit(train_x_s, y_train)

    artifact = None
    if best_name == "sgd":
        artifact = TrainingArtifact(
            task="classification",
            model_name="sgd",
            payload={
                "coef": np.asarray(best_model.coef_, dtype=np.float32),
                "intercept": np.asarray(best_model.intercept_, dtype=np.float32),
                "classes": np.asarray(best_model.classes_),
                "feature_dim": int(train_x_s.shape[1]),
            },
        )
    return best_model.predict(val_x_s), artifact


def _init_sgd_classifier(
    *,
    genome: HierarchicalGenome,
    inherited_state: TrainingArtifact | None,
    feature_dim: int,
    classes: np.ndarray,
) -> SGDClassifier:
    model = SGDClassifier(
        loss="log_loss",
        alpha=max(1e-4, 5e-4 - genome.reuse_ratio * 2e-4),
        max_iter=200,
        tol=1e-3,
        random_state=42,
        learning_rate="optimal",
        warm_start=True,
    )
    if inherited_state is None or inherited_state.model_name != "sgd":
        return model
    payload = inherited_state.payload
    if int(payload.get("feature_dim", -1)) != feature_dim:
        return model
    if tuple(np.asarray(payload.get("classes", []))) != tuple(classes):
        return model
    return _attach_sgd_state(model, payload)


def _fit_sgd(model: SGDClassifier, x: np.ndarray, y: np.ndarray) -> None:
    classes = np.unique(y)
    if not hasattr(model, "coef_"):
        model.partial_fit(x, y, classes=classes)
        return
    model.partial_fit(x, y)


def _attach_sgd_state(model: SGDClassifier, payload: dict[str, Any]) -> SGDClassifier:
    coef = np.asarray(payload["coef"], dtype=np.float64)
    intercept = np.asarray(payload["intercept"], dtype=np.float64)
    classes = np.asarray(payload["classes"])
    model.classes_ = classes
    model.coef_ = coef.copy()
    model.intercept_ = intercept.copy()
    model.n_features_in_ = coef.shape[1]
    model.t_ = 1.0
    return model


def _feature_buckets(encoded: np.ndarray, width: int = 64) -> np.ndarray:
    if encoded.ndim != 2:
        encoded = encoded.reshape(encoded.shape[0], -1)
    clipped = np.clip(encoded[:, : min(width, encoded.shape[1])], -2.0, 2.0)
    signs = (clipped > 0).astype(np.int32)
    powers = (1 << np.arange(signs.shape[1], dtype=np.int64)) % 997
    return ((signs * powers).sum(axis=1) % 997).astype(np.int32)


def _cap_data(
    spec: BenchmarkSpec,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, y_train, x_val, y_val = data
    if spec.task == "language_modeling":
        train_cap = min(len(x_train), spec.max_train_samples or 1024)
        val_cap = min(len(x_val), spec.max_val_samples or 256)
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
