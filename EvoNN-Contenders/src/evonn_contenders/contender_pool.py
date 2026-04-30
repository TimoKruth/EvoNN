"""Backward-compatible contender registry facade."""

from evonn_contenders.contenders import (
    ContenderSpec,
    benchmark_group,
    choose_best,
    contender_names_for_config,
    evaluate_contender,
    resolve_configured_contenders,
    resolve_contenders,
)

__all__ = [
    "ContenderSpec",
    "benchmark_group",
    "choose_best",
    "contender_names_for_config",
    "evaluate_contender",
    "resolve_configured_contenders",
    "resolve_contenders",
]
