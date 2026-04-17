"""DuckDB-backed storage for contender runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb


class RunStore:
    """Persist contender run metadata and results."""

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
            CREATE TABLE IF NOT EXISTS contenders (
                run_id VARCHAR,
                benchmark_name VARCHAR,
                contender_name VARCHAR,
                family VARCHAR,
                metric_name VARCHAR,
                metric_direction VARCHAR,
                metric_value DOUBLE,
                quality DOUBLE,
                parameter_count BIGINT,
                train_seconds DOUBLE,
                architecture_summary VARCHAR,
                contender_id VARCHAR,
                status VARCHAR,
                failure_reason VARCHAR
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmark_results (
                run_id VARCHAR,
                benchmark_name VARCHAR,
                contender_name VARCHAR,
                metric_name VARCHAR,
                metric_direction VARCHAR,
                metric_value DOUBLE,
                quality DOUBLE,
                parameter_count BIGINT,
                train_seconds DOUBLE,
                architecture_summary VARCHAR,
                contender_id VARCHAR,
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

    def clear_run_records(self, run_id: str) -> None:
        self.conn.execute("DELETE FROM contenders WHERE run_id = ?", [run_id])
        self.conn.execute("DELETE FROM benchmark_results WHERE run_id = ?", [run_id])
        self.conn.execute("DELETE FROM budget_meta WHERE run_id = ?", [run_id])

    def record_contender(self, *, run_id: str, benchmark_name: str, record: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO contenders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                benchmark_name,
                record["contender_name"],
                record["family"],
                record["metric_name"],
                record["metric_direction"],
                record.get("metric_value"),
                record.get("quality"),
                record.get("parameter_count"),
                record.get("train_seconds"),
                record.get("architecture_summary"),
                record.get("contender_id"),
                record["status"],
                record.get("failure_reason"),
            ],
        )

    def replace_contenders(self, *, run_id: str, benchmark_name: str, records: list[dict[str, Any]]) -> None:
        self.conn.execute(
            "DELETE FROM contenders WHERE run_id = ? AND benchmark_name = ?",
            [run_id, benchmark_name],
        )
        for record in records:
            self.record_contender(run_id=run_id, benchmark_name=benchmark_name, record=record)

    def record_result(self, *, run_id: str, benchmark_name: str, record: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO benchmark_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                benchmark_name,
                record["contender_name"],
                record["metric_name"],
                record["metric_direction"],
                record.get("metric_value"),
                record.get("quality"),
                record.get("parameter_count"),
                record.get("train_seconds"),
                record.get("architecture_summary"),
                record.get("contender_id"),
                record["status"],
                record.get("failure_reason"),
            ],
        )

    def replace_result(self, *, run_id: str, benchmark_name: str, record: dict[str, Any]) -> None:
        self.conn.execute(
            "DELETE FROM benchmark_results WHERE run_id = ? AND benchmark_name = ?",
            [run_id, benchmark_name],
        )
        self.record_result(run_id=run_id, benchmark_name=benchmark_name, record=record)

    def save_budget_metadata(self, *, run_id: str, payload: dict[str, Any]) -> None:
        self.conn.execute("DELETE FROM budget_meta WHERE run_id = ?", [run_id])
        for key, value in payload.items():
            self.conn.execute("INSERT INTO budget_meta VALUES (?, ?, ?)", [run_id, key, json.dumps(value)])

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

    def load_contenders(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT benchmark_name, contender_name, family, metric_name, metric_direction,
                   metric_value, quality, parameter_count, train_seconds,
                   architecture_summary, contender_id, status, failure_reason
            FROM contenders
            WHERE run_id = ?
            ORDER BY benchmark_name, contender_name
            """,
            [run_id],
        ).fetchall()
        return [
            {
                "benchmark_name": benchmark_name,
                "contender_name": contender_name,
                "family": family,
                "metric_name": metric_name,
                "metric_direction": metric_direction,
                "metric_value": metric_value,
                "quality": quality,
                "parameter_count": parameter_count,
                "train_seconds": train_seconds,
                "architecture_summary": architecture_summary,
                "contender_id": contender_id,
                "status": status,
                "failure_reason": failure_reason,
            }
            for (
                benchmark_name,
                contender_name,
                family,
                metric_name,
                metric_direction,
                metric_value,
                quality,
                parameter_count,
                train_seconds,
                architecture_summary,
                contender_id,
                status,
                failure_reason,
            ) in rows
        ]

    def load_contenders_for_benchmark(self, run_id: str, benchmark_name: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT benchmark_name, contender_name, family, metric_name, metric_direction,
                   metric_value, quality, parameter_count, train_seconds,
                   architecture_summary, contender_id, status, failure_reason
            FROM contenders
            WHERE run_id = ? AND benchmark_name = ?
            ORDER BY contender_name
            """,
            [run_id, benchmark_name],
        ).fetchall()
        return [
            {
                "benchmark_name": row[0],
                "contender_name": row[1],
                "family": row[2],
                "metric_name": row[3],
                "metric_direction": row[4],
                "metric_value": row[5],
                "quality": row[6],
                "parameter_count": row[7],
                "train_seconds": row[8],
                "architecture_summary": row[9],
                "contender_id": row[10],
                "status": row[11],
                "failure_reason": row[12],
            }
            for row in rows
        ]

    def load_results(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT benchmark_name, contender_name, metric_name, metric_direction,
                   metric_value, quality, parameter_count, train_seconds,
                   architecture_summary, contender_id, status, failure_reason
            FROM benchmark_results
            WHERE run_id = ?
            ORDER BY benchmark_name
            """,
            [run_id],
        ).fetchall()
        return [
            {
                "benchmark_name": benchmark_name,
                "contender_name": contender_name,
                "metric_name": metric_name,
                "metric_direction": metric_direction,
                "metric_value": metric_value,
                "quality": quality,
                "parameter_count": parameter_count,
                "train_seconds": train_seconds,
                "architecture_summary": architecture_summary,
                "contender_id": contender_id,
                "status": status,
                "failure_reason": failure_reason,
            }
            for (
                benchmark_name,
                contender_name,
                metric_name,
                metric_direction,
                metric_value,
                quality,
                parameter_count,
                train_seconds,
                architecture_summary,
                contender_id,
                status,
                failure_reason,
            ) in rows
        ]

    def load_result_for_benchmark(self, run_id: str, benchmark_name: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT benchmark_name, contender_name, metric_name, metric_direction,
                   metric_value, quality, parameter_count, train_seconds,
                   architecture_summary, contender_id, status, failure_reason
            FROM benchmark_results
            WHERE run_id = ? AND benchmark_name = ?
            LIMIT 1
            """,
            [run_id, benchmark_name],
        ).fetchone()
        if row is None:
            return None
        return {
            "benchmark_name": row[0],
            "contender_name": row[1],
            "metric_name": row[2],
            "metric_direction": row[3],
            "metric_value": row[4],
            "quality": row[5],
            "parameter_count": row[6],
            "train_seconds": row[7],
            "architecture_summary": row[8],
            "contender_id": row[9],
            "status": row[10],
            "failure_reason": row[11],
        }

    def load_budget_metadata(self, run_id: str) -> dict[str, Any]:
        rows = self.conn.execute(
            "SELECT key, value_json FROM budget_meta WHERE run_id = ?",
            [run_id],
        ).fetchall()
        return {key: json.loads(value_json) for key, value_json in rows}

    def close(self) -> None:
        self.conn.close()
