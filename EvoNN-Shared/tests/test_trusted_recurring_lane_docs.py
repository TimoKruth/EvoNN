from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "trust-layer-linux-ci.yml"
README_PATH = ROOT / "README.md"
CONTRIBUTING_PATH = ROOT / "CONTRIBUTING.md"
MONOREPO_PATH = ROOT / "MONOREPO.md"

EXPECTED_LANE_ENTRIES = [
    ("Shared", "evonn-shared", "scripts/ci/shared-checks.sh", "trusted recurring lane core"),
    ("Compare", "evonn-compare", "scripts/ci/compare-checks.sh", "trusted recurring lane core"),
    (
        "Contenders",
        "evonn-contenders",
        "scripts/ci/contenders-checks.sh",
        "trusted recurring lane challenger floor",
    ),
    (
        "Primordia",
        "evonn-primordia",
        "scripts/ci/primordia-checks.sh",
        "trusted recurring lane secondary challenger",
    ),
    (
        "Stratograph",
        "stratograph",
        "scripts/ci/stratograph-checks.sh",
        "trusted recurring lane secondary challenger",
    ),
]

EXPECTED_SCRIPTS = [entry[2] for entry in EXPECTED_LANE_ENTRIES]


def _workflow_entries(workflow_text: str) -> list[tuple[str, str, str, str]]:
    pattern = re.compile(
        r"- label: (?P<label>.+)\n"
        r"\s+package: (?P<package>.+)\n"
        r"\s+script: (?P<script>.+)\n"
        r"\s+scope: (?P<scope>.+)"
    )
    return [
        (match["label"], match["package"], match["script"], match["scope"])
        for match in pattern.finditer(workflow_text)
    ]


def test_trusted_recurring_lane_workflow_and_docs_stay_aligned() -> None:
    workflow_text = WORKFLOW_PATH.read_text()
    readme_text = README_PATH.read_text()
    contributing_text = CONTRIBUTING_PATH.read_text()
    monorepo_text = MONOREPO_PATH.read_text()

    assert _workflow_entries(workflow_text) == EXPECTED_LANE_ENTRIES
    assert '"README.md"' in workflow_text
    assert '"CONTRIBUTING.md"' in workflow_text

    assert "trusted recurring lane" in readme_text
    assert "trusted recurring lane" in contributing_text
    assert "trusted recurring lane" in monorepo_text

    for script in EXPECTED_SCRIPTS:
        assert script in readme_text
        assert script in contributing_text
        assert script in monorepo_text

    assert "trusted recurring lane core" in monorepo_text
    assert "trusted recurring lane challenger floor" in monorepo_text
    assert "trusted recurring lane secondary challengers" in monorepo_text
