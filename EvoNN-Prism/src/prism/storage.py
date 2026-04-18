"""DuckDB per-run storage for Prism NAS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

from prism.genome import ModelGenome

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    seed INTEGER,
    config_json VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS genomes (
    run_id VARCHAR,
    genome_id VARCHAR,
    family VARCHAR,
    genome_json VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (run_id, genome_id)
);

CREATE TABLE IF NOT EXISTS evaluations (
    run_id VARCHAR,
    genome_id VARCHAR,
    generation INTEGER,
    benchmark_id VARCHAR,
    metric_name VARCHAR,
    metric_value DOUBLE,
    quality DOUBLE,
    parameter_count BIGINT,
    train_seconds DOUBLE,
    failure_reason VARCHAR,
    inherited_from VARCHAR,
    inheritance_hit BOOLEAN,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS lineage (
    run_id VARCHAR,
    genome_id VARCHAR,
    parent_id VARCHAR,
    generation INTEGER,
    mutation_summary VARCHAR,
    operator_kind VARCHAR
);

CREATE TABLE IF NOT EXISTS archives (
    run_id VARCHAR,
    generation INTEGER,
    archive_kind VARCHAR,
    benchmark_id VARCHAR,
    genome_id VARCHAR,
    score DOUBLE,
    created_at TIMESTAMP DEFAULT current_timestamp
);
"""


class RunStore:
    """Per-run DuckDB storage layer."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self.conn.execute(SCHEMA)
        self._ensure_optional_columns()

    def _ensure_optional_columns(self) -> None:
        """Add analytics columns to older run DBs."""
        self.conn.execute("ALTER TABLE evaluations ADD COLUMN IF NOT EXISTS inherited_from VARCHAR")
        self.conn.execute("ALTER TABLE evaluations ADD COLUMN IF NOT EXISTS inheritance_hit BOOLEAN")
        self.conn.execute("ALTER TABLE archives ADD COLUMN IF NOT EXISTS generation INTEGER")

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    def save_run(self, run_id: str, config_dict: dict[str, Any]) -> None:
        seed = config_dict.get("seed", 0)
        self.conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, seed, config_json) VALUES (?, ?, ?)",
            [run_id, seed, json.dumps(config_dict)],
        )

    def save_genome(self, run_id: str, genome: ModelGenome) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO genomes (run_id, genome_id, family, genome_json) VALUES (?, ?, ?, ?)",
            [run_id, genome.genome_id, genome.family, json.dumps(genome.model_dump(mode="json"))],
        )

    def save_evaluation(
        self,
        run_id: str,
        genome_id: str,
        generation: int,
        benchmark_id: str,
        metric_name: str,
        metric_value: float,
        quality: float,
        parameter_count: int,
        train_seconds: float,
        failure_reason: str | None = None,
        inherited_from: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO evaluations (
                run_id, genome_id, generation, benchmark_id, metric_name,
                metric_value, quality, parameter_count, train_seconds, failure_reason,
                inherited_from, inheritance_hit
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id, genome_id, generation, benchmark_id, metric_name,
                metric_value, quality, parameter_count, train_seconds, failure_reason,
                inherited_from, inherited_from is not None,
            ],
        )

    def save_lineage(
        self,
        run_id: str,
        genome_id: str,
        parent_id: str | None,
        generation: int,
        mutation_summary: str,
        operator_kind: str = "mutation",
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO lineage (run_id, genome_id, parent_id, generation, mutation_summary, operator_kind)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [run_id, genome_id, parent_id, generation, mutation_summary, operator_kind],
        )

    def save_archive(
        self,
        run_id: str,
        generation: int,
        archive_kind: str,
        benchmark_id: str | None,
        genome_id: str,
        score: float,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO archives (run_id, generation, archive_kind, benchmark_id, genome_id, score)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [run_id, generation, archive_kind, benchmark_id, genome_id, score],
        )

    # ------------------------------------------------------------------
    # Read methods
    # ------------------------------------------------------------------

    def load_evaluations(self, run_id: str, generation: int | None = None) -> list[dict]:
        if generation is not None:
            rows = self.conn.execute(
                """
                SELECT genome_id, generation, benchmark_id, metric_name, metric_value,
                       quality, parameter_count, train_seconds, failure_reason,
                       inherited_from, inheritance_hit
                FROM evaluations
                WHERE run_id = ? AND generation = ?
                ORDER BY generation, genome_id, benchmark_id
                """,
                [run_id, generation],
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT genome_id, generation, benchmark_id, metric_name, metric_value,
                       quality, parameter_count, train_seconds, failure_reason,
                       inherited_from, inheritance_hit
                FROM evaluations
                WHERE run_id = ?
                ORDER BY generation, genome_id, benchmark_id
                """,
                [run_id],
            ).fetchall()
        cols = [
            "genome_id", "generation", "benchmark_id", "metric_name", "metric_value",
            "quality", "parameter_count", "train_seconds", "failure_reason",
            "inherited_from", "inheritance_hit",
        ]
        return [dict(zip(cols, row)) for row in rows]

    def load_genomes(self, run_id: str) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT genome_id, family, genome_json
            FROM genomes WHERE run_id = ?
            ORDER BY created_at
            """,
            [run_id],
        ).fetchall()
        results = []
        for genome_id, family, genome_json in rows:
            data = json.loads(genome_json)
            data["_genome_id"] = genome_id
            data["_family"] = family
            results.append(data)
        return results

    def load_best_per_benchmark(self, run_id: str) -> dict[str, dict]:
        rows = self.conn.execute(
            """
            SELECT benchmark_id, genome_id, quality, metric_name, metric_value,
                   parameter_count, train_seconds
            FROM evaluations
            WHERE run_id = ? AND failure_reason IS NULL
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY benchmark_id ORDER BY quality DESC
            ) = 1
            """,
            [run_id],
        ).fetchall()
        cols = [
            "benchmark_id", "genome_id", "quality", "metric_name", "metric_value",
            "parameter_count", "train_seconds",
        ]
        return {row[0]: dict(zip(cols, row)) for row in rows}

    def latest_generation(self, run_id: str) -> int | None:
        result = self.conn.execute(
            "SELECT MAX(generation) FROM evaluations WHERE run_id = ?",
            [run_id],
        ).fetchone()
        return result[0] if result and result[0] is not None else None

    def load_lineage(self, run_id: str, generation: int | None = None) -> list[dict]:
        if generation is not None:
            rows = self.conn.execute(
                """
                SELECT genome_id, parent_id, generation, mutation_summary, operator_kind
                FROM lineage
                WHERE run_id = ? AND generation = ?
                ORDER BY generation, genome_id, parent_id
                """,
                [run_id, generation],
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT genome_id, parent_id, generation, mutation_summary, operator_kind
                FROM lineage
                WHERE run_id = ?
                ORDER BY generation, genome_id, parent_id
                """,
                [run_id],
            ).fetchall()
        cols = ["genome_id", "parent_id", "generation", "mutation_summary", "operator_kind"]
        return [dict(zip(cols, row)) for row in rows]

    def load_archives(self, run_id: str, generation: int | None = None) -> list[dict]:
        if generation is not None:
            rows = self.conn.execute(
                """
                SELECT generation, archive_kind, benchmark_id, genome_id, score
                FROM archives
                WHERE run_id = ? AND generation = ?
                ORDER BY generation, archive_kind, benchmark_id, genome_id
                """,
                [run_id, generation],
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT generation, archive_kind, benchmark_id, genome_id, score
                FROM archives
                WHERE run_id = ?
                ORDER BY generation, archive_kind, benchmark_id, genome_id
                """,
                [run_id],
            ).fetchall()
        cols = ["generation", "archive_kind", "benchmark_id", "genome_id", "score"]
        return [dict(zip(cols, row)) for row in rows]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> RunStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
