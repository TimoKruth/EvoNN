"""Campaign runner wiring Prism and Topograph into compare contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from evonn_compare.comparison.engine import ComparisonEngine
from evonn_compare.contracts.parity import load_parity_pack
from evonn_compare.ingest.loader import SystemIngestor
from evonn_compare.orchestration.config_gen import CampaignCase
from evonn_compare.reporting.compare_md import render_comparison_markdown


@dataclass(frozen=True)
class CommandSpec:
    name: str
    cwd: Path
    argv: list[str]


class CampaignRunner:
    """Build runnable command specs and compare exported outputs."""

    def __init__(self, *, prism_root: Path, topograph_root: Path) -> None:
        self.prism_root = prism_root.resolve()
        self.topograph_root = topograph_root.resolve()

    def prism_run_dir(self, case: CampaignCase) -> Path:
        return self.prism_root / "runs" / case.prism_config_path.stem

    def prism_command(self, case: CampaignCase) -> CommandSpec:
        run_dir = self.prism_run_dir(case)
        script = (
            "from pathlib import Path; "
            "from prism.config import load_config; "
            "from prism.benchmarks.datasets import get_benchmark; "
            "from prism.pipeline.coordinator import run_evolution; "
            f"cfg = load_config(Path(r'{case.prism_config_path}')); "
            "benchmarks = [get_benchmark(name) for name in (cfg.benchmark_pack.benchmark_ids or [])]; "
            f"run_evolution(cfg, benchmarks, run_dir=r'{run_dir}', resume=False)"
        )
        return CommandSpec(
            name="prism_run",
            cwd=self.prism_root,
            argv=["uv", "run", "python", "-c", script],
        )

    def topograph_command(self, case: CampaignCase) -> CommandSpec:
        return CommandSpec(
            name="topograph_run",
            cwd=self.topograph_root,
            argv=[
                "uv",
                "run",
                "topograph",
                "evolve",
                "--config",
                str(case.topograph_config_path),
                "--run-dir",
                str(case.topograph_run_dir),
            ],
        )

    def prism_export_command(self, run_dir: Path, pack: str | Path) -> CommandSpec:
        script = (
            "from pathlib import Path; "
            "from prism.export.symbiosis import export_symbiosis_contract; "
            f"export_symbiosis_contract(run_dir=Path(r'{run_dir}'), pack_path=Path(r'{pack}'))"
        )
        return CommandSpec(
            name="prism_export",
            cwd=self.prism_root,
            argv=["uv", "run", "python", "-c", script],
        )

    def topograph_export_command(self, run_dir: Path, pack: str | Path) -> CommandSpec:
        script = (
            "from pathlib import Path; "
            "from topograph.export.symbiosis import export_symbiosis_contract; "
            f"export_symbiosis_contract(run_dir=Path(r'{run_dir}'), pack_path=Path(r'{pack}'))"
        )
        return CommandSpec(
            name="topograph_export",
            cwd=self.topograph_root,
            argv=["uv", "run", "python", "-c", script],
        )

    def planned_commands(self, case: CampaignCase) -> list[CommandSpec]:
        return [
            spec
            for stage in self.execution_stages(case)
            for spec in stage
        ]

    def execution_stages(self, case: CampaignCase) -> list[list[CommandSpec]]:
        prism_run_dir = self.prism_run_dir(case)
        return [
            [
                self.prism_command(case),
                self.topograph_command(case),
            ],
            [
                self.prism_export_command(prism_run_dir, case.pack_path),
                self.topograph_export_command(case.topograph_run_dir, case.pack_path),
            ],
        ]

    def execution_commands(self, case: CampaignCase) -> list[CommandSpec]:
        return self.planned_commands(case)

    def compare_exports(self, *, left_dir: Path, right_dir: Path, pack_path: Path, output_path: Path) -> None:
        pack = load_parity_pack(pack_path)
        left = SystemIngestor(left_dir)
        right = SystemIngestor(right_dir)
        left_report = left.validate(pack)
        right_report = right.validate(pack)
        if not left_report.ok or not right_report.ok:
            issues: list[str] = []
            for side, report in (("left", left_report), ("right", right_report)):
                for issue in report.issues:
                    issues.append(f"{side}:{issue.code}:{issue.message}")
            raise RuntimeError(
                "campaign exports failed validation against parity pack:\n" + "\n".join(issues)
            )
        result = ComparisonEngine().compare(
            left_manifest=left.load_manifest(),
            left_results=left.load_results(),
            right_manifest=right.load_manifest(),
            right_results=right.load_results(),
            pack=pack,
        )
        if result.parity_status != "fair":
            reasons = ", ".join(result.reasons) if result.reasons else "unknown parity mismatch"
            raise RuntimeError(f"comparison is not fair for pack {pack.name}: {reasons}")
        output_path.write_text(render_comparison_markdown(result), encoding="utf-8")
        output_path.with_suffix(".json").write_text(
            json.dumps(result.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
