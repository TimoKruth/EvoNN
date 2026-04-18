"""Run ingest implementation with normalization for Prism/Topograph exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter

from evonn_compare.contracts.models import ResultRecord, RunManifest
from evonn_compare.contracts.parity import ParityPack, load_parity_pack
from evonn_compare.contracts.validation import ValidationReport, validate_contract


class SystemIngestor:
    """Load and validate one exported run directory."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    def load_manifest(self) -> RunManifest:
        """Load `manifest.json` from the run directory."""

        manifest_path = self.run_dir / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return RunManifest.model_validate(_normalize_manifest(payload, self.run_dir))

    def load_results(self) -> list[ResultRecord]:
        """Load `results.json` from the run directory."""

        results_path = self.run_dir / "results.json"
        adapter = TypeAdapter(list[ResultRecord])
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        return adapter.validate_python(_normalize_results(payload))

    def validate(self, pack: ParityPack) -> ValidationReport:
        """Validate the run export against a parity pack."""

        return validate_contract(
            manifest=self.load_manifest(),
            results=self.load_results(),
            pack=pack,
            run_dir=self.run_dir,
        )

    def ingest(self, pack: ParityPack | None = None) -> dict:
        """Load and summarize one run export."""

        manifest = self.load_manifest()
        results = self.load_results()
        report = self.validate(pack) if pack is not None else ValidationReport(issues=[])
        return {
            "manifest": manifest.model_dump(mode="json"),
            "results": [record.model_dump(mode="json") for record in results],
            "validation": {
                "ok": report.ok,
                "issues": [issue.model_dump(mode="json") for issue in report.issues],
            },
        }


def ingest_run_dir(run_dir: Path, pack_name: str | None = None) -> dict:
    """Convenience wrapper for ingesting one run directory."""

    ingestor = SystemIngestor(run_dir)
    pack = load_parity_pack(pack_name) if pack_name else None
    return ingestor.ingest(pack)


def _normalize_manifest(payload: dict, run_dir: Path) -> dict:
    artifacts = payload.get("artifacts", {})
    if "config_snapshot" not in artifacts:
        artifacts["config_snapshot"] = _default_artifact(run_dir, "config.yaml", "config_snapshot.json")
    if "report_markdown" not in artifacts:
        artifacts["report_markdown"] = _default_artifact(run_dir, "report.md")
    payload["artifacts"] = artifacts

    budget = dict(payload.get("budget", {}))
    budget.setdefault("effective_training_epochs", None)
    budget.setdefault("wall_clock_seconds", None)
    budget.setdefault("generations", None)
    budget.setdefault("population_size", None)
    budget.setdefault("budget_policy_name", None)
    payload["budget"] = budget

    device = dict(payload.get("device", {}))
    device.setdefault("device_name", "unknown")
    device.setdefault("precision_mode", "unknown")
    device.setdefault("framework", None)
    device.setdefault("framework_version", None)
    payload["device"] = device

    payload.setdefault("schema_version", "1.0")
    payload.setdefault("run_id", run_dir.name)
    payload.setdefault("run_name", payload["run_id"])
    payload.setdefault("created_at", "2026-04-01T00:00:00Z")
    payload.setdefault("search_telemetry", None)

    fairness = dict(payload.get("fairness", {}))
    fairness.setdefault("benchmark_pack_id", payload.get("pack_name"))
    fairness.setdefault("seed", payload.get("seed"))
    fairness.setdefault("evaluation_count", budget.get("evaluation_count"))
    fairness.setdefault("budget_policy_name", budget.get("budget_policy_name"))
    fairness.setdefault("data_signature", _default_data_signature(payload))
    fairness.setdefault("code_version", None)
    payload["fairness"] = fairness

    normalized_benchmarks = []
    for entry in payload.get("benchmarks", []):
        row = dict(entry)
        row.setdefault("status", "ok")
        normalized_benchmarks.append(row)
    payload["benchmarks"] = normalized_benchmarks
    return payload


def _normalize_results(payload: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for record in payload:
        row = dict(record)
        row.setdefault("status", "ok")
        row.setdefault("failure_reason", None)
        normalized.append(row)
    return normalized


def _default_artifact(run_dir: Path, *candidates: str) -> str:
    for candidate in candidates:
        if (run_dir / candidate).exists():
            return candidate
    return candidates[0]


def _default_data_signature(payload: dict) -> str:
    artifacts = payload.get("artifacts", {}) or {}
    dataset_hash = artifacts.get("dataset_manifest_hash")
    if dataset_hash:
        return str(dataset_hash)
    benchmark_rows = [
        {
            "benchmark_id": entry.get("benchmark_id"),
            "metric_name": entry.get("metric_name"),
            "metric_direction": entry.get("metric_direction"),
            "task_kind": entry.get("task_kind"),
        }
        for entry in payload.get("benchmarks", [])
    ]
    raw = json.dumps(
        {
            "pack_name": payload.get("pack_name"),
            "benchmarks": benchmark_rows,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
