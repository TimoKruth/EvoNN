"""DuckDB-backed run storage for Stratograph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb


class RunStore:
    """Persist prototype run metadata and results."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.path))
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id VARCHAR PRIMARY KEY,
                run_name VARCHAR,
                created_at TIMESTAMP,
                seed INTEGER,
                config_json VARCHAR
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS genomes (
                run_id VARCHAR,
                generation INTEGER,
                genome_id VARCHAR,
                benchmark_name VARCHAR,
                payload_json VARCHAR,
                architecture_summary VARCHAR,
                parameter_count BIGINT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmark_results (
                run_id VARCHAR,
                benchmark_name VARCHAR,
                metric_name VARCHAR,
                metric_direction VARCHAR,
                metric_value DOUBLE,
                quality DOUBLE,
                parameter_count BIGINT,
                train_seconds DOUBLE,
                architecture_summary VARCHAR,
                genome_id VARCHAR,
                status VARCHAR,
                failure_reason VARCHAR
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS budget_meta (
                run_id VARCHAR,
                key VARCHAR,
                value_json VARCHAR
            )
            """
        )

    def record_run(self, *, run_id: str, run_name: str, created_at: str, seed: int, config: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?)",
            [run_id, run_name, created_at, seed, json.dumps(config)],
        )

    def record_genome(
        self,
        *,
        run_id: str,
        generation: int,
        genome_id: str,
        benchmark_name: str,
        payload: dict[str, Any],
        architecture_summary: str,
        parameter_count: int,
    ) -> None:
        self.conn.execute(
            "INSERT INTO genomes VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                generation,
                genome_id,
                benchmark_name,
                json.dumps(payload),
                architecture_summary,
                parameter_count,
            ],
        )

    def record_result(self, *, run_id: str, benchmark_name: str, record: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO benchmark_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                benchmark_name,
                record["metric_name"],
                record["metric_direction"],
                record.get("metric_value"),
                record.get("quality"),
                record.get("parameter_count"),
                record.get("train_seconds"),
                record.get("architecture_summary"),
                record.get("genome_id"),
                record["status"],
                record.get("failure_reason"),
            ],
        )

    def save_budget_metadata(self, *, run_id: str, payload: dict[str, Any]) -> None:
        self.conn.execute("DELETE FROM budget_meta WHERE run_id = ?", [run_id])
        for key, value in payload.items():
            self.conn.execute(
                "INSERT INTO budget_meta VALUES (?, ?, ?)",
                [run_id, key, json.dumps(value)],
            )

    def load_runs(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT run_id, run_name, created_at, seed, config_json FROM runs ORDER BY created_at DESC"
        ).fetchall()
        return [
            {
                "run_id": run_id,
                "run_name": run_name,
                "created_at": str(created_at),
                "seed": seed,
                "config": json.loads(config_json),
            }
            for run_id, run_name, created_at, seed, config_json in rows
        ]

    def load_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT run_id, run_name, created_at, seed, config_json FROM runs WHERE run_id = ?",
            [run_id],
        ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row[0],
            "run_name": row[1],
            "created_at": str(row[2]),
            "seed": row[3],
            "config": json.loads(row[4]),
        }

    def load_genomes(self, run_id: str, generation: int | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT generation, genome_id, benchmark_name, payload_json, architecture_summary, parameter_count
            FROM genomes
            WHERE run_id = ?
        """
        params: list[Any] = [run_id]
        if generation is not None:
            query += " AND generation = ?"
            params.append(generation)
        query += " ORDER BY generation, benchmark_name, genome_id"
        rows = self.conn.execute(query, params).fetchall()
        return [
            {
                "generation": generation_value,
                "genome_id": genome_id,
                "benchmark_name": benchmark_name,
                "payload": json.loads(payload_json),
                "architecture_summary": architecture_summary,
                "parameter_count": parameter_count,
            }
            for generation_value, genome_id, benchmark_name, payload_json, architecture_summary, parameter_count in rows
        ]

    def load_results(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT benchmark_name, metric_name, metric_direction, metric_value, quality,
                   parameter_count, train_seconds, architecture_summary, genome_id, status, failure_reason
            FROM benchmark_results
            WHERE run_id = ?
            ORDER BY benchmark_name
            """,
            [run_id],
        ).fetchall()
        return [
            {
                "benchmark_name": benchmark_name,
                "metric_name": metric_name,
                "metric_direction": metric_direction,
                "metric_value": metric_value,
                "quality": quality,
                "parameter_count": parameter_count,
                "train_seconds": train_seconds,
                "architecture_summary": architecture_summary,
                "genome_id": genome_id,
                "status": status,
                "failure_reason": failure_reason,
            }
            for (
                benchmark_name,
                metric_name,
                metric_direction,
                metric_value,
                quality,
                parameter_count,
                train_seconds,
                architecture_summary,
                genome_id,
                status,
                failure_reason,
            ) in rows
        ]

    def load_best_benchmark_results(self, run_id: str) -> list[dict[str, Any]]:
        return self.load_results(run_id)

    def load_budget_metadata(self, run_id: str) -> dict[str, Any]:
        rows = self.conn.execute(
            "SELECT key, value_json FROM budget_meta WHERE run_id = ?",
            [run_id],
        ).fetchall()
        return {key: json.loads(value_json) for key, value_json in rows}

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "RunStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
