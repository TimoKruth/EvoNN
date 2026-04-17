"""Contender registry and evaluation helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol
import math
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LanguageModel(Protocol):
    """Minimal LM contender protocol."""

    parameter_count: int

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None: ...

    def perplexity(self, x_val: np.ndarray, y_val: np.ndarray) -> float: ...


@dataclass(frozen=True)
class ContenderSpec:
    """One contender definition."""

    name: str
    family: str
    group: str


_TABULAR = {
    "hist_gb": ContenderSpec("hist_gb", "gradient_boosting", "tabular"),
    "hist_gb_small": ContenderSpec("hist_gb_small", "gradient_boosting", "tabular"),
    "hist_gb_leaf63": ContenderSpec("hist_gb_leaf63", "gradient_boosting", "tabular"),
    "hist_gb_deep": ContenderSpec("hist_gb_deep", "gradient_boosting", "tabular"),
    "extra_trees": ContenderSpec("extra_trees", "tree_ensemble", "tabular"),
    "extra_trees_small": ContenderSpec("extra_trees_small", "tree_ensemble", "tabular"),
    "extra_trees_deep": ContenderSpec("extra_trees_deep", "tree_ensemble", "tabular"),
    "extra_trees_entropy": ContenderSpec("extra_trees_entropy", "tree_ensemble", "tabular"),
    "random_forest": ContenderSpec("random_forest", "tree_ensemble", "tabular"),
    "random_forest_small": ContenderSpec("random_forest_small", "tree_ensemble", "tabular"),
    "random_forest_deep": ContenderSpec("random_forest_deep", "tree_ensemble", "tabular"),
    "random_forest_entropy": ContenderSpec("random_forest_entropy", "tree_ensemble", "tabular"),
    "mlp": ContenderSpec("mlp", "mlp", "tabular"),
    "mlp_small": ContenderSpec("mlp_small", "mlp", "tabular"),
    "mlp_wide": ContenderSpec("mlp_wide", "mlp", "tabular"),
    "mlp_deep": ContenderSpec("mlp_deep", "mlp", "tabular"),
    "logistic": ContenderSpec("logistic", "linear", "tabular"),
    "logistic_c1": ContenderSpec("logistic_c1", "linear", "tabular"),
    "logistic_c10": ContenderSpec("logistic_c10", "linear", "tabular"),
    "logistic_balanced": ContenderSpec("logistic_balanced", "linear", "tabular"),
}

_SYNTHETIC = {
    "hist_gb": ContenderSpec("hist_gb", "gradient_boosting", "synthetic"),
    "hist_gb_deep": ContenderSpec("hist_gb_deep", "gradient_boosting", "synthetic"),
    "extra_trees": ContenderSpec("extra_trees", "tree_ensemble", "synthetic"),
    "extra_trees_deep": ContenderSpec("extra_trees_deep", "tree_ensemble", "synthetic"),
    "random_forest": ContenderSpec("random_forest", "tree_ensemble", "synthetic"),
    "random_forest_deep": ContenderSpec("random_forest_deep", "tree_ensemble", "synthetic"),
    "mlp": ContenderSpec("mlp", "mlp", "synthetic"),
    "mlp_wide": ContenderSpec("mlp_wide", "mlp", "synthetic"),
}

_IMAGE = {
    "mlp": ContenderSpec("mlp", "mlp", "image"),
    "mlp_small": ContenderSpec("mlp_small", "mlp", "image"),
    "mlp_wide": ContenderSpec("mlp_wide", "mlp", "image"),
    "mlp_deep": ContenderSpec("mlp_deep", "mlp", "image"),
    "logistic": ContenderSpec("logistic", "linear", "image"),
    "logistic_c1": ContenderSpec("logistic_c1", "linear", "image"),
    "logistic_c10": ContenderSpec("logistic_c10", "linear", "image"),
    "random_forest": ContenderSpec("random_forest", "tree_ensemble", "image"),
    "extra_trees": ContenderSpec("extra_trees", "tree_ensemble", "image"),
    "hist_gb": ContenderSpec("hist_gb", "gradient_boosting", "image"),
}

_LM = {
    "bigram_lm": ContenderSpec("bigram_lm", "ngram", "language_modeling"),
    "bigram_lm_a01": ContenderSpec("bigram_lm_a01", "ngram", "language_modeling"),
    "bigram_lm_a10": ContenderSpec("bigram_lm_a10", "ngram", "language_modeling"),
    "bigram_lm_a20": ContenderSpec("bigram_lm_a20", "ngram", "language_modeling"),
    "unigram_lm": ContenderSpec("unigram_lm", "ngram", "language_modeling"),
    "unigram_lm_a01": ContenderSpec("unigram_lm_a01", "ngram", "language_modeling"),
    "unigram_lm_a05": ContenderSpec("unigram_lm_a05", "ngram", "language_modeling"),
    "unigram_lm_a20": ContenderSpec("unigram_lm_a20", "ngram", "language_modeling"),
}

_GROUPS = {
    "tabular": _TABULAR,
    "synthetic": _SYNTHETIC,
    "image": _IMAGE,
    "language_modeling": _LM,
}


class UnigramLanguageModel:
    """Smoothed unigram next-token model."""

    def __init__(self, vocab_size: int, alpha: float = 1.0) -> None:
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.counts = np.zeros(vocab_size, dtype=np.int64)
        self.total = 0
        self.parameter_count = 0

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        del x_train
        flat = y_train.reshape(-1)
        self.counts = np.bincount(flat, minlength=self.vocab_size).astype(np.int64, copy=False)
        self.total = int(self.counts.sum())
        self.parameter_count = int(np.count_nonzero(self.counts))

    def perplexity(self, x_val: np.ndarray, y_val: np.ndarray) -> float:
        del x_val
        denom = self.total + self.alpha * self.vocab_size
        log_prob_sum = 0.0
        token_count = 0
        for token in y_val.reshape(-1):
            prob = (self.counts[int(token)] + self.alpha) / denom
            log_prob_sum += -math.log(prob)
            token_count += 1
        return float(math.exp(log_prob_sum / max(1, token_count)))


class BigramLanguageModel:
    """Smoothed bigram next-token model keyed by previous token."""

    def __init__(self, vocab_size: int, alpha: float = 0.5) -> None:
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.row_totals: dict[int, int] = defaultdict(int)
        self.transition_counts: dict[int, Counter[int]] = defaultdict(Counter)
        self.fallback = np.zeros(vocab_size, dtype=np.int64)
        self.parameter_count = 0

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        prev_tokens = x_train.reshape(-1)
        next_tokens = y_train.reshape(-1)
        for prev, nxt in zip(prev_tokens, next_tokens, strict=False):
            prev_id = int(prev)
            next_id = int(nxt)
            self.transition_counts[prev_id][next_id] += 1
            self.row_totals[prev_id] += 1
            self.fallback[next_id] += 1
        self.parameter_count = int(sum(len(counter) for counter in self.transition_counts.values()))

    def perplexity(self, x_val: np.ndarray, y_val: np.ndarray) -> float:
        flat_prev = x_val.reshape(-1)
        flat_next = y_val.reshape(-1)
        fallback_total = int(self.fallback.sum())
        log_prob_sum = 0.0
        token_count = 0
        for prev, nxt in zip(flat_prev, flat_next, strict=False):
            prev_id = int(prev)
            next_id = int(nxt)
            row = self.transition_counts.get(prev_id)
            if row is None:
                prob = (self.fallback[next_id] + self.alpha) / (fallback_total + self.alpha * self.vocab_size)
            else:
                prob = (row[next_id] + self.alpha) / (self.row_totals[prev_id] + self.alpha * self.vocab_size)
            log_prob_sum += -math.log(prob)
            token_count += 1
        return float(math.exp(log_prob_sum / max(1, token_count)))


def benchmark_group(spec: Any) -> str:
    """Map benchmark spec to contender group."""
    if spec.task == "language_modeling":
        return "language_modeling"
    if spec.name in {"blobs_f2_c2", "circles", "moons", "circles_n02_f3"}:
        return "synthetic"
    if spec.name in {"digits", "fashion_mnist", "mnist"} or spec.source == "image":
        return "image"
    return "tabular"


def resolve_contenders(group: str, names: list[str]) -> list[ContenderSpec]:
    """Resolve names to contender specs."""
    try:
        registry = _GROUPS[group]
    except KeyError as exc:
        raise KeyError(f"Unknown contender group: {group}") from exc
    resolved: list[ContenderSpec] = []
    for name in names:
        try:
            contender = registry[name]
        except KeyError as exc:
            raise KeyError(f"Unknown contender {name} for group {group}") from exc
        resolved.append(contender)
    return resolved


def contender_names_for_config(config: Any, group: str) -> list[str]:
    """Get contender names for one group from config."""
    if group == "synthetic":
        return config.contender_pool.synthetic
    if group == "image":
        return config.contender_pool.image
    if group == "language_modeling":
        return config.contender_pool.language_modeling
    return config.contender_pool.tabular


def evaluate_contender(
    spec: Any,
    contender: ContenderSpec,
    *,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, Any]:
    """Fit one contender and return metric payload."""
    contender_id = f"{spec.name}:{contender.name}:seed{seed}"
    started = perf_counter()
    try:
        if spec.task == "language_modeling":
            metric_value, parameter_count = _run_language_model(
                contender.name,
                spec.model_output_dim,
                x_train,
                y_train,
                x_val,
                y_val,
            )
            status = "ok"
            failure_reason = None
        else:
            metric_value, parameter_count = _run_classifier(
                contender.name,
                seed,
                x_train,
                y_train,
                x_val,
                y_val,
            )
            status = "ok"
            failure_reason = None
    except Exception as exc:
        metric_value = None
        parameter_count = 0
        status = "failed"
        failure_reason = str(exc)
    train_seconds = perf_counter() - started
    quality = _quality_from_metric(spec.metric_direction, metric_value)
    return {
        "contender_name": contender.name,
        "family": contender.family,
        "metric_name": spec.metric_name,
        "metric_direction": spec.metric_direction,
        "metric_value": metric_value,
        "quality": quality,
        "parameter_count": parameter_count,
        "train_seconds": train_seconds,
        "architecture_summary": contender.name,
        "contender_id": contender_id,
        "status": status,
        "failure_reason": failure_reason,
    }


def choose_best(metric_direction: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick best successful contender or first failure."""
    ok_records = [record for record in records if record["status"] == "ok" and record["metric_value"] is not None]
    if not ok_records:
        failed = records[0].copy()
        failed["quality"] = _quality_from_metric(metric_direction, None)
        return failed
    reverse = metric_direction == "max"
    return sorted(ok_records, key=lambda record: record["metric_value"], reverse=reverse)[0]


def _run_classifier(
    contender_name: str,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, int]:
    estimator = _build_classifier(contender_name, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_val)
    metric = float(accuracy_score(y_val, y_pred))
    return metric, _estimate_parameter_count(estimator)


def _build_classifier(contender_name: str, seed: int) -> Any:
    if contender_name == "hist_gb":
        return HistGradientBoostingClassifier(random_state=seed)
    if contender_name == "hist_gb_small":
        return HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=15, random_state=seed)
    if contender_name == "hist_gb_leaf63":
        return HistGradientBoostingClassifier(max_leaf_nodes=63, learning_rate=0.05, random_state=seed)
    if contender_name == "hist_gb_deep":
        return HistGradientBoostingClassifier(max_depth=12, max_leaf_nodes=63, learning_rate=0.05, random_state=seed)
    if contender_name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=256, random_state=seed, n_jobs=-1)
    if contender_name == "extra_trees_small":
        return ExtraTreesClassifier(n_estimators=128, max_depth=12, random_state=seed, n_jobs=-1)
    if contender_name == "extra_trees_deep":
        return ExtraTreesClassifier(n_estimators=384, max_depth=None, min_samples_leaf=1, random_state=seed, n_jobs=-1)
    if contender_name == "extra_trees_entropy":
        return ExtraTreesClassifier(n_estimators=256, criterion="entropy", random_state=seed, n_jobs=-1)
    if contender_name == "random_forest":
        return RandomForestClassifier(n_estimators=256, random_state=seed, n_jobs=-1)
    if contender_name == "random_forest_small":
        return RandomForestClassifier(n_estimators=128, max_depth=12, random_state=seed, n_jobs=-1)
    if contender_name == "random_forest_deep":
        return RandomForestClassifier(n_estimators=384, max_depth=None, min_samples_leaf=1, random_state=seed, n_jobs=-1)
    if contender_name == "random_forest_entropy":
        return RandomForestClassifier(n_estimators=256, criterion="entropy", random_state=seed, n_jobs=-1)
    if contender_name == "mlp":
        return _build_mlp(seed, hidden_layer_sizes=(128, 64), max_iter=150)
    if contender_name == "mlp_small":
        return _build_mlp(seed, hidden_layer_sizes=(64, 32), max_iter=120)
    if contender_name == "mlp_wide":
        return _build_mlp(seed, hidden_layer_sizes=(256, 128), max_iter=180)
    if contender_name == "mlp_deep":
        return _build_mlp(seed, hidden_layer_sizes=(256, 128, 64), max_iter=180)
    if contender_name == "logistic":
        return _build_logistic(seed, c=1.0, class_weight=None)
    if contender_name == "logistic_c1":
        return _build_logistic(seed, c=0.1, class_weight=None)
    if contender_name == "logistic_c10":
        return _build_logistic(seed, c=10.0, class_weight=None)
    if contender_name == "logistic_balanced":
        return _build_logistic(seed, c=1.0, class_weight="balanced")
    raise KeyError(f"Unknown classifier contender: {contender_name}")


def _run_language_model(
    contender_name: str,
    vocab_size: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, int]:
    model: LanguageModel
    if contender_name == "unigram_lm":
        model = UnigramLanguageModel(vocab_size)
    elif contender_name == "unigram_lm_a01":
        model = UnigramLanguageModel(vocab_size, alpha=0.1)
    elif contender_name == "unigram_lm_a05":
        model = UnigramLanguageModel(vocab_size, alpha=0.5)
    elif contender_name == "unigram_lm_a20":
        model = UnigramLanguageModel(vocab_size, alpha=2.0)
    elif contender_name == "bigram_lm":
        model = BigramLanguageModel(vocab_size)
    elif contender_name == "bigram_lm_a01":
        model = BigramLanguageModel(vocab_size, alpha=0.1)
    elif contender_name == "bigram_lm_a10":
        model = BigramLanguageModel(vocab_size, alpha=1.0)
    elif contender_name == "bigram_lm_a20":
        model = BigramLanguageModel(vocab_size, alpha=2.0)
    else:
        raise KeyError(f"Unknown LM contender: {contender_name}")
    model.fit(x_train, y_train)
    return model.perplexity(x_val, y_val), model.parameter_count


def _build_mlp(seed: int, *, hidden_layer_sizes: tuple[int, ...], max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    max_iter=max_iter,
                    early_stopping=True,
                    random_state=seed,
                ),
            ),
        ]
    )


def _build_logistic(seed: int, *, c: float, class_weight: str | None) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    max_iter=500,
                    solver="lbfgs",
                    class_weight=class_weight,
                    random_state=seed,
                ),
            ),
        ]
    )


def _quality_from_metric(metric_direction: str, metric_value: float | None) -> float:
    if metric_value is None:
        return float("-inf") if metric_direction == "max" else float("inf")
    return float(metric_value if metric_direction == "max" else -metric_value)


def _estimate_parameter_count(estimator: Any) -> int:
    if isinstance(estimator, Pipeline):
        estimator = estimator[-1]
    if hasattr(estimator, "coefs_"):
        return int(sum(np.asarray(weight).size for weight in estimator.coefs_) + sum(np.asarray(bias).size for bias in estimator.intercepts_))
    if hasattr(estimator, "coef_"):
        total = np.asarray(estimator.coef_).size
        if hasattr(estimator, "intercept_"):
            total += np.asarray(estimator.intercept_).size
        return int(total)
    if hasattr(estimator, "estimators_"):
        total = 0
        for sub_estimator in estimator.estimators_:
            tree = getattr(sub_estimator, "tree_", None)
            if tree is not None:
                total += int(tree.node_count)
        return int(total)
    tree = getattr(estimator, "tree_", None)
    if tree is not None:
        return int(tree.node_count)
    predictors = getattr(estimator, "_predictors", None)
    if predictors is not None:
        total = 0
        for stage in predictors:
            for node in stage:
                tree_obj = getattr(node, "nodes", None)
                if tree_obj is not None:
                    total += int(len(tree_obj))
        return int(total)
    return 0
