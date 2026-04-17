"""Comparison and campaign summary primitives."""

from evonn_compare.comparison.engine import (
    ComparisonEngine,
    ComparisonMatchup,
    ComparisonResult,
    ComparisonSummary,
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
    "build_campaign_leaderboard",
]
