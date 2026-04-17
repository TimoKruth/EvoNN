"""DuckDB storage for per-run persistence of genomes, counters, and results."""

from __future__ import annotations

import json

from topograph.genome.codec import genome_to_dict
import uuid
from pathlib import Path

import duckdb


class RunStore:
    """Per-run DuckDB database for evolution state."""

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.path))
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id VARCHAR PRIMARY KEY,
                config_json VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS genomes (
                run_id VARCHAR,
                generation INTEGER,
                genome_idx INTEGER,
                genome_json VARCHAR,
                fitness DOUBLE,
                param_count INTEGER,
                model_bytes INTEGER,
                PRIMARY KEY (run_id, generation, genome_idx)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS innovation_counters (
                run_id VARCHAR PRIMARY KEY,
                counter_value INTEGER
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS budget_metadata (
                run_id VARCHAR PRIMARY KEY,
                metadata_json VARCHAR
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS run_states (
                run_id VARCHAR PRIMARY KEY,
                state_json VARCHAR
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_results (
                run_id VARCHAR,
                generation INTEGER,
                benchmark_name VARCHAR,
                metric_name VARCHAR,
                metric_direction VARCHAR,
                metric_value DOUBLE,
                quality DOUBLE,
                parameter_count INTEGER,
                train_seconds DOUBLE,
                architecture_summary VARCHAR,
                genome_id VARCHAR,
                genome_idx INTEGER,
                status VARCHAR,
                failure_reason VARCHAR,
                PRIMARY KEY (run_id, generation, benchmark_name)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_timings (
                run_id VARCHAR,
                generation INTEGER,
                benchmark_order INTEGER,
                benchmark_name VARCHAR,
                task VARCHAR,
                data_load_seconds DOUBLE,
                evaluation_seconds DOUBLE,
                total_seconds DOUBLE,
                trained_count INTEGER,
                reused_count INTEGER,
                failed_count INTEGER,
                requested_worker_count INTEGER,
                resolved_worker_count INTEGER,
                PRIMARY KEY (run_id, generation, benchmark_order)
            )
        """)

    # -- Run management --------------------------------------------------------

    def save_run(self, run_id: str | None, config_dict: dict) -> str:
        """Save a run config. Generates a run_id if not provided."""
        if run_id is None:
            run_id = uuid.uuid4().hex[:8]
        self.conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, config_json) VALUES (?, ?)",
            [run_id, json.dumps(config_dict)],
        )
        return run_id

    def load_run(self, run_id: str) -> dict:
        row = self.conn.execute(
            "SELECT config_json FROM runs WHERE run_id = ?", [run_id],
        ).fetchone()
        if not row:
            raise ValueError(f"Run not found: {run_id}")
        return json.loads(row[0])

    # -- Genome persistence ----------------------------------------------------

    def save_genomes(self, run_id: str, generation: int, genomes) -> None:
        """Save genomes for a generation. Accepts Genome objects or dicts."""
        for idx, genome in enumerate(genomes):
            if not isinstance(genome, dict):
                d = genome_to_dict(genome)
            else:
                d = genome
            fitness = d.pop("fitness", None)
            param_count = d.pop("param_count", None)
            model_bytes = d.pop("model_bytes", None)
            genome_json = json.dumps(d)
            self.conn.execute(
                """INSERT OR REPLACE INTO genomes
                   (run_id, generation, genome_idx, genome_json, fitness,
                    param_count, model_bytes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [run_id, generation, idx, genome_json, fitness, param_count, model_bytes],
            )

    def load_genomes(self, run_id: str, generation: int) -> list[dict]:
        rows = self.conn.execute(
            """SELECT genome_json, fitness, param_count, model_bytes
               FROM genomes WHERE run_id = ? AND generation = ?
               ORDER BY genome_idx""",
            [run_id, generation],
        ).fetchall()
        genomes: list[dict] = []
        for genome_json, fitness, param_count, model_bytes in rows:
            genome = json.loads(genome_json)
            genome["fitness"] = fitness
            genome["param_count"] = param_count
            genome["model_bytes"] = model_bytes
            genomes.append(genome)
        return genomes

    def load_latest_generation(self, run_id: str) -> int | None:
        row = self.conn.execute(
            "SELECT MAX(generation) FROM genomes WHERE run_id = ?", [run_id],
        ).fetchone()
        if not row or row[0] is None:
            return None
        return row[0]

    # -- Innovation counter ----------------------------------------------------

    def save_innovation_counter(self, run_id: str, value: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO innovation_counters (run_id, counter_value) VALUES (?, ?)",
            [run_id, value],
        )

    def load_innovation_counter(self, run_id: str) -> int | None:
        row = self.conn.execute(
            "SELECT counter_value FROM innovation_counters WHERE run_id = ?", [run_id],
        ).fetchone()
        if not row:
            return None
        return row[0]

    # -- Budget metadata -------------------------------------------------------

    def save_budget_metadata(self, run_id: str, metadata: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO budget_metadata (run_id, metadata_json) VALUES (?, ?)",
            [run_id, json.dumps(metadata)],
        )

    def load_budget_metadata(self, run_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT metadata_json FROM budget_metadata WHERE run_id = ?", [run_id],
        ).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def save_run_state(self, run_id: str, state: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO run_states (run_id, state_json) VALUES (?, ?)",
            [run_id, json.dumps(state)],
        )

    def load_run_state(self, run_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT state_json FROM run_states WHERE run_id = ?",
            [run_id],
        ).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    # -- Benchmark results -----------------------------------------------------

    def save_benchmark_results(
        self, run_id: str, generation: int, results: list[dict],
    ) -> None:
        for result in results:
            self.conn.execute(
                """INSERT OR REPLACE INTO benchmark_results (
                    run_id, generation, benchmark_name, metric_name,
                    metric_direction, metric_value, quality, parameter_count,
                    train_seconds, architecture_summary, genome_id, genome_idx,
                    status, failure_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    run_id, generation,
                    result["benchmark_name"], result["metric_name"],
                    result["metric_direction"], result["metric_value"],
                    result["quality"], result["parameter_count"],
                    result["train_seconds"], result["architecture_summary"],
                    result["genome_id"], result["genome_idx"],
                    result["status"], result["failure_reason"],
                ],
            )

    def load_best_benchmark_results(self, run_id: str) -> list[dict]:
        """Load the best result per benchmark for a run."""
        rows = self.conn.execute(
            """SELECT generation, benchmark_name, metric_name, metric_direction,
                      metric_value, quality, parameter_count, train_seconds,
                      architecture_summary, genome_id, genome_idx, status,
                      failure_reason
               FROM benchmark_results
               WHERE run_id = ?
               ORDER BY benchmark_name, generation DESC""",
            [run_id],
        ).fetchall()

        best: dict[str, dict] = {}
        for row in rows:
            record = {
                "generation": row[0],
                "benchmark_name": row[1],
                "metric_name": row[2],
                "metric_direction": row[3],
                "metric_value": row[4],
                "quality": row[5],
                "parameter_count": row[6],
                "train_seconds": row[7],
                "architecture_summary": row[8],
                "genome_id": row[9],
                "genome_idx": row[10],
                "status": row[11],
                "failure_reason": row[12],
            }
            current = best.get(record["benchmark_name"])
            if current is None or _is_better(record, current):
                best[record["benchmark_name"]] = record

        return list(best.values())

    def save_benchmark_timings(
        self, run_id: str, generation: int, timings: list[dict],
    ) -> None:
        for timing in timings:
            self.conn.execute(
                """INSERT OR REPLACE INTO benchmark_timings (
                    run_id, generation, benchmark_order, benchmark_name, task,
                    data_load_seconds, evaluation_seconds, total_seconds,
                    trained_count, reused_count, failed_count,
                    requested_worker_count, resolved_worker_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    run_id,
                    generation,
                    timing["benchmark_order"],
                    timing["benchmark_name"],
                    timing["task"],
                    timing["data_load_seconds"],
                    timing["evaluation_seconds"],
                    timing["total_seconds"],
                    timing["trained_count"],
                    timing["reused_count"],
                    timing["failed_count"],
                    timing["requested_worker_count"],
                    timing["resolved_worker_count"],
                ],
            )

    def load_benchmark_timings(
        self, run_id: str, generation: int | None = None,
    ) -> list[dict]:
        query = (
            """SELECT generation, benchmark_order, benchmark_name, task,
                      data_load_seconds, evaluation_seconds, total_seconds,
                      trained_count, reused_count, failed_count,
                      requested_worker_count, resolved_worker_count
               FROM benchmark_timings
               WHERE run_id = ?
            """
        )
        params: list[object] = [run_id]
        if generation is not None:
            query += " AND generation = ?"
            params.append(generation)
        query += " ORDER BY generation, benchmark_order"
        rows = self.conn.execute(query, params).fetchall()
        return [
            {
                "generation": row[0],
                "benchmark_order": row[1],
                "benchmark_name": row[2],
                "task": row[3],
                "data_load_seconds": row[4],
                "evaluation_seconds": row[5],
                "total_seconds": row[6],
                "trained_count": row[7],
                "reused_count": row[8],
                "failed_count": row[9],
                "requested_worker_count": row[10],
                "resolved_worker_count": row[11],
            }
            for row in rows
        ]

    # -- Lifecycle -------------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> RunStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _is_better(candidate: dict, current: dict) -> bool:
    """Compare two benchmark records by status then metric value."""
    if current["status"] != "ok" and candidate["status"] == "ok":
        return True
    if candidate["status"] != "ok":
        return False
    if current["status"] != "ok":
        return False
    if candidate["metric_direction"] == "min":
        return float(candidate["metric_value"]) < float(current["metric_value"])
    return float(candidate["metric_value"]) > float(current["metric_value"])
