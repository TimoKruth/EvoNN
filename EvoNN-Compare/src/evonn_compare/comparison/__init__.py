"""Comparison and campaign summary primitives."""

from evonn_compare.comparison.engine import (
    ComparisonEngine,
    ComparisonMatchup,
    ComparisonResult,
    ComparisonSummary,
)
from evonn_compare.comparison.fair_matrix import (
    FairMatrixSummary,
    MatrixBudgetRow,
    PairParityRow,
    build_matrix_summary,
    summarize_matrix_case,
)
from evonn_compare.comparison.leaderboard import (
    CampaignLeaderboard,
    CampaignRecord,
    build_campaign_leaderboard,
)

__all__ = [
    "CampaignLeaderboard",
    "CampaignRecord",
    "ComparisonEngine",
    "ComparisonMatchup",
    "ComparisonResult",
    "ComparisonSummary",
    "FairMatrixSummary",
    "MatrixBudgetRow",
    "PairParityRow",
    "build_matrix_summary",
    "build_campaign_leaderboard",
    "summarize_matrix_case",
]
