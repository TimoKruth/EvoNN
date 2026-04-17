"""Contender runtime dispatch."""

from __future__ import annotations

from collections import Counter, defaultdict
from time import perf_counter
from typing import Any, Protocol
import math
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from evonn_contenders.contenders.registry import ContenderSpec


class LanguageModel(Protocol):
    """Minimal LM contender protocol."""

    parameter_count: int

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None: ...

    def perplexity(self, x_val: np.ndarray, y_val: np.ndarray) -> float: ...


class OptionalDependencyError(RuntimeError):
    """Raised when optional contender backend is not installed."""


class GuardrailError(RuntimeError):
    """Raised when contender should not run on current benchmark size."""


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


def evaluate_contender(
    spec: Any,
    contender: ContenderSpec,
    *,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: Any | None = None,
) -> dict[str, Any]:
    """Fit one contender and return metric payload."""
    contender_id = f"{spec.name}:{contender.name}:seed{seed}"
    started = perf_counter()
    try:
        if contender.backend in {"sklearn_classifier", "boosted_classifier"}:
            metric_value, parameter_count = _run_classifier(
                spec,
                contender.name,
                seed,
                x_train,
                y_train,
                x_val,
                y_val,
                config,
            )
        elif contender.backend == "ngram_lm":
            metric_value, parameter_count = _run_ngram_language_model(
                contender.name,
                spec.model_output_dim,
                x_train,
                y_train,
                x_val,
                y_val,
            )
        elif contender.backend == "torch_cnn":
            metric_value, parameter_count = _run_cnn_classifier(
                spec,
                contender.name,
                seed,
                x_train,
                y_train,
                x_val,
                y_val,
                config,
            )
        elif contender.backend == "torch_transformer_lm":
            metric_value, parameter_count = _run_transformer_language_model(
                spec,
                contender.name,
                seed,
                x_train,
                y_train,
                x_val,
                y_val,
                config,
            )
        else:
            raise KeyError(f"Unknown contender backend: {contender.backend}")
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
    spec: Any,
    contender_name: str,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: Any | None,
) -> tuple[float, int]:
    estimator = _build_classifier(spec, contender_name, seed, x_train, config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_val)
    metric = float(accuracy_score(y_val, y_pred))
    return metric, _estimate_parameter_count(estimator)


def _build_classifier(
    spec: Any,
    contender_name: str,
    seed: int,
    x_train: np.ndarray,
    config: Any | None,
) -> Any:
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
    if contender_name == "linear_svc":
        return _build_linear_svc(seed, class_weight=None)
    if contender_name == "linear_svc_balanced":
        return _build_linear_svc(seed, class_weight="balanced")
    if contender_name == "rbf_svc_small":
        _check_kernel_svm_guardrails(spec, x_train, config)
        return _build_rbf_svc(seed)
    if contender_name == "svm_nystroem_rbf":
        return _build_nystroem_svc(seed, x_train.shape[1], len(x_train))
    if contender_name == "xgb_default":
        return _build_xgboost(seed, num_classes=spec.model_output_dim, small=False)
    if contender_name == "xgb_small":
        return _build_xgboost(seed, num_classes=spec.model_output_dim, small=True)
    if contender_name == "lgbm_default":
        return _build_lightgbm(seed, num_classes=spec.model_output_dim, small=False)
    if contender_name == "lgbm_small":
        return _build_lightgbm(seed, num_classes=spec.model_output_dim, small=True)
    if contender_name == "catboost_default":
        return _build_catboost(seed, num_classes=spec.model_output_dim, small=False)
    if contender_name == "catboost_small":
        return _build_catboost(seed, num_classes=spec.model_output_dim, small=True)
    raise KeyError(f"Unknown classifier contender: {contender_name}")


def _run_ngram_language_model(
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


def _run_cnn_classifier(
    spec: Any,
    contender_name: str,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: Any | None,
) -> tuple[float, int]:
    torch = _import_torch()
    from torch.utils.data import DataLoader, TensorDataset

    from evonn_contenders.contenders.torch_models import build_cnn

    torch.manual_seed(seed)
    device = _torch_device(config)
    x_train_t = _prepare_image_tensor(spec, x_train, device)
    x_val_t = _prepare_image_tensor(spec, x_val, device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=device)

    model = build_cnn(
        contender_name,
        channels=x_train_t.shape[1],
        num_classes=spec.model_output_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=getattr(getattr(config, "torch", None), "learning_rate", 1e-3))
    loss_fn = torch.nn.CrossEntropyLoss()

    dataset = TensorDataset(x_train_t, y_train_t)
    loader = DataLoader(
        dataset,
        batch_size=getattr(getattr(config, "torch", None), "batch_size", 64),
        shuffle=True,
    )
    epochs = getattr(getattr(config, "torch", None), "classifier_epochs", 3)
    max_batches = getattr(getattr(config, "torch", None), "max_batches_per_epoch", 32)

    model.train()
    for _ in range(epochs):
        for batch_index, (batch_x, batch_y) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            if batch_index + 1 >= max_batches:
                break

    model.eval()
    with torch.no_grad():
        logits = model(x_val_t)
        predictions = logits.argmax(dim=1)
        metric = float((predictions == y_val_t).float().mean().item())
    return metric, _torch_parameter_count(model)


def _run_transformer_language_model(
    spec: Any,
    contender_name: str,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: Any | None,
) -> tuple[float, int]:
    torch = _import_torch()
    from torch.utils.data import DataLoader, TensorDataset

    from evonn_contenders.contenders.torch_models import build_transformer_lm

    torch.manual_seed(seed)
    device = _torch_device(config)
    context_limit = getattr(getattr(config, "torch", None), "context_length_override", 64)
    context_length = min(x_train.shape[1], context_limit) if context_limit is not None else x_train.shape[1]
    train_limit = getattr(getattr(config, "torch", None), "max_train_samples", 1024)
    val_limit = getattr(getattr(config, "torch", None), "max_val_samples", 256)

    x_train_t = torch.as_tensor(x_train[:train_limit, :context_length], dtype=torch.long, device=device)
    y_train_t = torch.as_tensor(y_train[:train_limit, :context_length], dtype=torch.long, device=device)
    x_val_t = torch.as_tensor(x_val[:val_limit, :context_length], dtype=torch.long, device=device)
    y_val_t = torch.as_tensor(y_val[:val_limit, :context_length], dtype=torch.long, device=device)

    model = build_transformer_lm(
        contender_name,
        vocab_size=spec.model_output_dim,
        context_length=context_length,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=getattr(getattr(config, "torch", None), "learning_rate", 1e-3))
    loss_fn = torch.nn.CrossEntropyLoss()
    dataset = TensorDataset(x_train_t, y_train_t)
    loader = DataLoader(
        dataset,
        batch_size=getattr(getattr(config, "torch", None), "batch_size", 32),
        shuffle=True,
    )
    max_steps = getattr(getattr(config, "torch", None), "lm_steps", 40)

    model.train()
    steps = 0
    while steps < max_steps:
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), batch_y.reshape(-1))
            loss.backward()
            optimizer.step()
            steps += 1
            if steps >= max_steps:
                break

    model.eval()
    with torch.no_grad():
        logits = model(x_val_t)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y_val_t.reshape(-1))
        metric = float(torch.exp(loss).item())
    return metric, _torch_parameter_count(model)


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


def _build_linear_svc(seed: int, *, class_weight: str | None) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LinearSVC(
                    C=1.0,
                    class_weight=class_weight,
                    random_state=seed,
                    max_iter=4000,
                    dual="auto",
                ),
            ),
        ]
    )


def _build_rbf_svc(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", SVC(C=2.0, kernel="rbf", gamma="scale", random_state=seed)),
        ]
    )


def _build_nystroem_svc(seed: int, input_dim: int, train_rows: int) -> Pipeline:
    components = min(256, max(32, input_dim * 4), train_rows)
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("kernel", Nystroem(kernel="rbf", gamma=0.5, n_components=components, random_state=seed)),
            ("clf", LinearSVC(C=1.0, random_state=seed, max_iter=4000, dual="auto")),
        ]
    )


def _build_xgboost(seed: int, *, num_classes: int, small: bool):
    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError("Optional dependency missing for xgb contender: xgboost") from exc
    kwargs: dict[str, Any] = {
        "n_estimators": 96 if small else 160,
        "max_depth": 4 if small else 6,
        "learning_rate": 0.08 if small else 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "mlogloss",
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": 0,
    }
    if num_classes <= 2:
        kwargs["objective"] = "binary:logistic"
        kwargs["eval_metric"] = "logloss"
    else:
        kwargs["objective"] = "multi:softprob"
        kwargs["num_class"] = num_classes
    return XGBClassifier(**kwargs)


def _build_lightgbm(seed: int, *, num_classes: int, small: bool):
    try:
        from lightgbm import LGBMClassifier
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError("Optional dependency missing for lgbm contender: lightgbm") from exc
    kwargs: dict[str, Any] = {
        "n_estimators": 96 if small else 160,
        "num_leaves": 31 if small else 63,
        "learning_rate": 0.08 if small else 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": seed,
        "verbose": -1,
        "n_jobs": 1,
    }
    if num_classes > 2:
        kwargs["objective"] = "multiclass"
        kwargs["num_class"] = num_classes
    return LGBMClassifier(**kwargs)


def _build_catboost(seed: int, *, num_classes: int, small: bool):
    try:
        from catboost import CatBoostClassifier
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError("Optional dependency missing for catboost contender: catboost") from exc
    return CatBoostClassifier(
        iterations=96 if small else 160,
        depth=5 if small else 7,
        learning_rate=0.08 if small else 0.05,
        loss_function="Logloss" if num_classes <= 2 else "MultiClass",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def _check_kernel_svm_guardrails(spec: Any, x_train: np.ndarray, config: Any | None) -> None:
    svm_cfg = getattr(config, "svm", None)
    max_rows = getattr(svm_cfg, "kernel_svm_max_train_samples", 4000)
    max_dim = getattr(svm_cfg, "kernel_svm_max_input_dim", 256)
    if x_train.shape[0] > max_rows:
        raise GuardrailError(
            f"Kernel SVM blocked for {spec.name}: train rows {x_train.shape[0]} > limit {max_rows}"
        )
    if x_train.shape[1] > max_dim:
        raise GuardrailError(
            f"Kernel SVM blocked for {spec.name}: input dim {x_train.shape[1]} > limit {max_dim}"
        )


def _import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError("Optional dependency missing for torch contender: torch") from exc
    return torch


def _torch_device(config: Any | None):
    torch = _import_torch()
    requested = getattr(getattr(config, "torch", None), "device", "cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare_image_tensor(spec: Any, x: np.ndarray, device: Any):
    torch = _import_torch()
    height, width, channels = spec.resolved_image_shape
    if x.ndim == 2:
        expected = height * width * channels
        if x.shape[1] != expected:
            raise ValueError(f"Cannot reshape {spec.name}: expected {expected} flat features, got {x.shape[1]}")
        image = x.reshape(-1, height, width, channels)
    elif x.ndim == 4:
        image = x
    else:
        raise ValueError(f"Unsupported image tensor rank for {spec.name}: {x.ndim}")
    image = image.astype(np.float32, copy=False)
    max_value = float(np.max(image)) if image.size else 1.0
    if max_value > 1.0:
        image = image / max_value
    image = np.transpose(image, (0, 3, 1, 2))
    return torch.as_tensor(image, dtype=torch.float32, device=device)


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
            for predictor in stage:
                nodes = getattr(getattr(predictor, "nodes", None), "shape", None)
                if nodes is not None:
                    total += int(predictor.nodes.shape[0])
        return int(total)
    return 0


def _torch_parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
