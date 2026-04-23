"""Campaign generation and execution helpers."""

from evonn_compare.orchestration.config_gen import (
    CampaignCase,
    CampaignPaths,
    generate_budget_pack,
    generate_prism_config,
    generate_topograph_config,
)
from evonn_compare.orchestration.contenders import (
    ContenderArtifacts,
    ensure_contender_export,
    ensure_contender_run,
)
from evonn_compare.orchestration.fair_matrix import (
    MatrixCase,
    MatrixPaths,
    generate_contender_config,
    generate_primordia_config,
    generate_stratograph_config,
    prepare_fair_matrix_cases,
    run_fair_matrix_case,
)
from evonn_compare.orchestration.primordia import (
    PrimordiaArtifacts,
    ensure_primordia_export,
    ensure_primordia_run,
)

__all__ = [
    "CampaignCase",
    "CampaignPaths",
    "ContenderArtifacts",
    "PrimordiaArtifacts",
    "MatrixCase",
    "MatrixPaths",
    "ensure_contender_export",
    "ensure_contender_run",
    "ensure_primordia_export",
    "ensure_primordia_run",
    "generate_contender_config",
    "generate_primordia_config",
    "generate_budget_pack",
    "generate_prism_config",
    "generate_stratograph_config",
    "generate_topograph_config",
    "prepare_fair_matrix_cases",
    "run_fair_matrix_case",
]
