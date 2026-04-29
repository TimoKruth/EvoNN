"""Contender registry definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContenderSpec:
    """One contender definition."""

    name: str
    family: str
    group: str
    backend: str
    task_kind: str
    supports_groups: tuple[str, ...]
    optional_dependency: str | None = None
    budget_mode: str = "fit_once"


def _spec(
    name: str,
    family: str,
    group: str,
    *,
    backend: str,
    task_kind: str,
    optional_dependency: str | None = None,
    budget_mode: str = "fit_once",
) -> ContenderSpec:
    return ContenderSpec(
        name=name,
        family=family,
        group=group,
        backend=backend,
        task_kind=task_kind,
        supports_groups=(group,),
        optional_dependency=optional_dependency,
        budget_mode=budget_mode,
    )


_TABULAR = {
    "hist_gb": _spec("hist_gb", "gradient_boosting", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "hist_gb_small": _spec("hist_gb_small", "gradient_boosting", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "hist_gb_leaf63": _spec("hist_gb_leaf63", "gradient_boosting", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "hist_gb_deep": _spec("hist_gb_deep", "gradient_boosting", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "extra_trees": _spec("extra_trees", "tree_ensemble", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "extra_trees_small": _spec("extra_trees_small", "tree_ensemble", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "extra_trees_deep": _spec("extra_trees_deep", "tree_ensemble", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "extra_trees_entropy": _spec("extra_trees_entropy", "tree_ensemble", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "random_forest": _spec("random_forest", "tree_ensemble", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "random_forest_small": _spec("random_forest_small", "tree_ensemble", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "random_forest_deep": _spec("random_forest_deep", "tree_ensemble", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "random_forest_entropy": _spec("random_forest_entropy", "tree_ensemble", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "mlp": _spec("mlp", "mlp", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "mlp_small": _spec("mlp_small", "mlp", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "mlp_wide": _spec("mlp_wide", "mlp", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "mlp_deep": _spec("mlp_deep", "mlp", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "logistic": _spec("logistic", "linear", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "logistic_c1": _spec("logistic_c1", "linear", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "logistic_c10": _spec("logistic_c10", "linear", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "logistic_balanced": _spec("logistic_balanced", "linear", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "linear_svc": _spec("linear_svc", "svm", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "linear_svc_balanced": _spec("linear_svc_balanced", "svm", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "rbf_svc_small": _spec("rbf_svc_small", "svm", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "svm_nystroem_rbf": _spec("svm_nystroem_rbf", "svm", "tabular", backend="sklearn_classifier", task_kind="classification"),
    "xgb_default": _spec("xgb_default", "boosted_tree", "tabular", backend="boosted_classifier", task_kind="classification", optional_dependency="xgboost"),
    "xgb_small": _spec("xgb_small", "boosted_tree", "tabular", backend="boosted_classifier", task_kind="classification", optional_dependency="xgboost"),
    "lgbm_default": _spec("lgbm_default", "boosted_tree", "tabular", backend="boosted_classifier", task_kind="classification", optional_dependency="lightgbm"),
    "lgbm_small": _spec("lgbm_small", "boosted_tree", "tabular", backend="boosted_classifier", task_kind="classification", optional_dependency="lightgbm"),
    "catboost_default": _spec("catboost_default", "boosted_tree", "tabular", backend="boosted_classifier", task_kind="classification", optional_dependency="catboost"),
    "catboost_small": _spec("catboost_small", "boosted_tree", "tabular", backend="boosted_classifier", task_kind="classification", optional_dependency="catboost"),
}

_SYNTHETIC = {
    "hist_gb": _spec("hist_gb", "gradient_boosting", "synthetic", backend="sklearn_classifier", task_kind="classification"),
    "hist_gb_deep": _spec("hist_gb_deep", "gradient_boosting", "synthetic", backend="sklearn_classifier", task_kind="classification"),
    "extra_trees": _spec("extra_trees", "tree_ensemble", "synthetic", backend="sklearn_classifier", task_kind="classification"),
    "extra_trees_deep": _spec("extra_trees_deep", "tree_ensemble", "synthetic", backend="sklearn_classifier", task_kind="classification"),
    "random_forest": _spec("random_forest", "tree_ensemble", "synthetic", backend="sklearn_classifier", task_kind="classification"),
    "random_forest_deep": _spec("random_forest_deep", "tree_ensemble", "synthetic", backend="sklearn_classifier", task_kind="classification"),
    "mlp": _spec("mlp", "mlp", "synthetic", backend="sklearn_classifier", task_kind="classification"),
    "mlp_wide": _spec("mlp_wide", "mlp", "synthetic", backend="sklearn_classifier", task_kind="classification"),
    "linear_svc": _spec("linear_svc", "svm", "synthetic", backend="sklearn_classifier", task_kind="classification"),
    "rbf_svc_small": _spec("rbf_svc_small", "svm", "synthetic", backend="sklearn_classifier", task_kind="classification"),
    "svm_nystroem_rbf": _spec("svm_nystroem_rbf", "svm", "synthetic", backend="sklearn_classifier", task_kind="classification"),
    "xgb_small": _spec("xgb_small", "boosted_tree", "synthetic", backend="boosted_classifier", task_kind="classification", optional_dependency="xgboost"),
    "lgbm_small": _spec("lgbm_small", "boosted_tree", "synthetic", backend="boosted_classifier", task_kind="classification", optional_dependency="lightgbm"),
    "catboost_small": _spec("catboost_small", "boosted_tree", "synthetic", backend="boosted_classifier", task_kind="classification", optional_dependency="catboost"),
}

_IMAGE = {
    "mlp": _spec("mlp", "mlp", "image", backend="sklearn_classifier", task_kind="classification"),
    "mlp_small": _spec("mlp_small", "mlp", "image", backend="sklearn_classifier", task_kind="classification"),
    "mlp_wide": _spec("mlp_wide", "mlp", "image", backend="sklearn_classifier", task_kind="classification"),
    "mlp_deep": _spec("mlp_deep", "mlp", "image", backend="sklearn_classifier", task_kind="classification"),
    "logistic": _spec("logistic", "linear", "image", backend="sklearn_classifier", task_kind="classification"),
    "logistic_c1": _spec("logistic_c1", "linear", "image", backend="sklearn_classifier", task_kind="classification"),
    "logistic_c10": _spec("logistic_c10", "linear", "image", backend="sklearn_classifier", task_kind="classification"),
    "random_forest": _spec("random_forest", "tree_ensemble", "image", backend="sklearn_classifier", task_kind="classification"),
    "extra_trees": _spec("extra_trees", "tree_ensemble", "image", backend="sklearn_classifier", task_kind="classification"),
    "hist_gb": _spec("hist_gb", "gradient_boosting", "image", backend="sklearn_classifier", task_kind="classification"),
    "linear_svc": _spec("linear_svc", "svm", "image", backend="sklearn_classifier", task_kind="classification"),
    "cnn_small": _spec("cnn_small", "cnn", "image", backend="torch_cnn", task_kind="classification", optional_dependency="torch", budget_mode="torch_epochs"),
    "cnn_medium": _spec("cnn_medium", "cnn", "image", backend="torch_cnn", task_kind="classification", optional_dependency="torch", budget_mode="torch_epochs"),
    "cnn_regularized": _spec("cnn_regularized", "cnn", "image", backend="torch_cnn", task_kind="classification", optional_dependency="torch", budget_mode="torch_epochs"),
}

_LM = {
    "bigram_lm": _spec("bigram_lm", "ngram", "language_modeling", backend="ngram_lm", task_kind="language_modeling"),
    "bigram_lm_a01": _spec("bigram_lm_a01", "ngram", "language_modeling", backend="ngram_lm", task_kind="language_modeling"),
    "bigram_lm_a10": _spec("bigram_lm_a10", "ngram", "language_modeling", backend="ngram_lm", task_kind="language_modeling"),
    "bigram_lm_a20": _spec("bigram_lm_a20", "ngram", "language_modeling", backend="ngram_lm", task_kind="language_modeling"),
    "unigram_lm": _spec("unigram_lm", "ngram", "language_modeling", backend="ngram_lm", task_kind="language_modeling"),
    "unigram_lm_a01": _spec("unigram_lm_a01", "ngram", "language_modeling", backend="ngram_lm", task_kind="language_modeling"),
    "unigram_lm_a05": _spec("unigram_lm_a05", "ngram", "language_modeling", backend="ngram_lm", task_kind="language_modeling"),
    "unigram_lm_a20": _spec("unigram_lm_a20", "ngram", "language_modeling", backend="ngram_lm", task_kind="language_modeling"),
    "transformer_lm_tiny": _spec("transformer_lm_tiny", "transformer", "language_modeling", backend="torch_transformer_lm", task_kind="language_modeling", optional_dependency="torch", budget_mode="torch_steps"),
    "transformer_lm_small": _spec("transformer_lm_small", "transformer", "language_modeling", backend="torch_transformer_lm", task_kind="language_modeling", optional_dependency="torch", budget_mode="torch_steps"),
}

_GROUPS = {
    "tabular": _TABULAR,
    "synthetic": _SYNTHETIC,
    "image": _IMAGE,
    "language_modeling": _LM,
}


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


def resolve_configured_contenders(config: Any, group: str) -> list[ContenderSpec]:
    """Resolve the effective configured contender set for one group."""

    contender_names = contender_names_for_config(config, group)
    max_contenders = getattr(getattr(config, "selection", None), "max_contenders_per_benchmark", None)
    if max_contenders is not None:
        contender_names = contender_names[:max_contenders]
    return resolve_contenders(group, contender_names)
