"""Lightweight statistics for campaign summaries."""
from __future__ import annotations

from math import comb

from evonn_compare.comparison.wilcoxon import (  # noqa: F401
    BootstrapCI,
    WilcoxonResult,
    bootstrap_confidence_interval,
    wilcoxon_signed_rank,
)


def two_sided_sign_test(successes: int, failures: int) -> float:
    trials = successes + failures
    if trials == 0:
        return 1.0
    tail = min(successes, failures)
    probability = sum(comb(trials, k) for k in range(0, tail + 1)) / (2 ** trials)
    return min(1.0, round(2.0 * probability, 6))
