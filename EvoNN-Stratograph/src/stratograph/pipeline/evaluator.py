"""Fast benchmark evaluator for Stratograph candidates."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
import math
import time

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stratograph.benchmarks.spec import BenchmarkSpec
from stratograph.genome import HierarchicalGenome
from stratograph.runtime import compile_genome


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


def evaluate_candidate(
    genome: HierarchicalGenome,
    spec: BenchmarkSpec,
    *,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> EvaluationRecord:
    """Evaluate one candidate on one benchmark."""
    started = time.perf_counter()
    compiled = compile_genome(genome)
    x_train, y_train, x_val, y_val = _cap_data(spec, data)
    if spec.task == "language_modeling":
        metric_value = _evaluate_language_modeling(genome, x_train, y_train, x_val, y_val)
        quality = -metric_value
    else:
        train_features = compiled.encode(x_train).reshape(x_train.shape[0], -1)
        val_features = compiled.encode(x_val).reshape(x_val.shape[0], -1)
        predictions = _predict_classification(
            genome=genome,
            train_features=train_features,
            y_train=y_train,
            val_features=val_features,
            y_val=y_val,
        )
        metric_value = float(accuracy_score(y_val, predictions))
        quality = metric_value

    elapsed = time.perf_counter() - started
    return EvaluationRecord(
        metric_value=metric_value,
        quality=quality,
        parameter_count=compiled.parameter_count(),
        train_seconds=elapsed,
        architecture_summary=compiled.architecture_summary(),
        genome_id=genome.genome_id,
        status="ok",
    )


def _evaluate_language_modeling(
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
    vocab_size = int(max(train_prev.max(initial=0), train_next.max(initial=0), val_prev.max(initial=0), val_next.max(initial=0)) + 1)
    unigram = np.bincount(train_next, minlength=vocab_size).astype(np.float64)
    alpha = 0.25 + (genome.average_cell_depth * 0.1) + (genome.reuse_ratio * 0.2)
    unigram_probs = (unigram + alpha) / (unigram.sum() + alpha * vocab_size)

    bigram_counts: dict[int, Counter[int]] = defaultdict(Counter)
    totals: Counter[int] = Counter()
    for prev_token, next_token in zip(train_prev.tolist(), train_next.tolist()):
        bigram_counts[prev_token][next_token] += 1
        totals[prev_token] += 1

    mix = min(0.95, 0.6 + genome.reuse_ratio * 0.15 + min(genome.average_cell_depth, 4.0) * 0.05)
    log_probs: list[float] = []
    for prev_token, next_token in zip(val_prev.tolist(), val_next.tolist()):
        cond_total = totals.get(prev_token, 0)
        cond_count = bigram_counts.get(prev_token, Counter()).get(next_token, 0)
        cond_prob = (cond_count + alpha) / (cond_total + alpha * vocab_size)
        prob = mix * cond_prob + (1.0 - mix) * unigram_probs[next_token]
        log_probs.append(math.log(max(prob, 1e-12)))
    return float(math.exp(-sum(log_probs) / max(1, len(log_probs))))


def _predict_classification(
    *,
    genome: HierarchicalGenome,
    train_features: np.ndarray,
    y_train: np.ndarray,
    val_features: np.ndarray,
    y_val: np.ndarray,
) -> np.ndarray:
    if len(np.unique(y_train)) <= 1:
        return np.full_like(y_val, fill_value=y_train[0] if len(y_train) else 0)

    test_size = 0.2 if len(y_train) >= 40 else 0.25
    fit_x, holdout_x, fit_y, holdout_y = train_test_split(
        train_features,
        y_train,
        test_size=test_size,
        random_state=42,
        stratify=y_train,
    )

    head_choices = [
        (
            "ridge",
            make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                RidgeClassifier(alpha=max(0.1, 1.0 - genome.reuse_ratio)),
            ),
        ),
        (
            "lda",
            make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                LinearDiscriminantAnalysis(),
            ),
        ),
        (
            "logreg",
            make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                LogisticRegression(
                    max_iter=400,
                    C=1.0 + genome.reuse_ratio,
                    solver="lbfgs",
                ),
            ),
        ),
    ]

    best_name = "ridge"
    best_score = float("-inf")
    for name, model in head_choices:
        try:
            model.fit(fit_x, fit_y)
            score = accuracy_score(holdout_y, model.predict(holdout_x))
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_name = name

    model = dict(head_choices)[best_name]
    model.fit(train_features, y_train)
    return model.predict(val_features)


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
