#!/usr/bin/env python3
"""Run a tiny end-to-end LM smoke evolution on `tiny_lm_synthetic`."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "tiny_lm_synthetic_smoke.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Topograph LM smoke test on tiny_lm_synthetic.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to Topograph config YAML.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional run directory. Defaults to a temp dir under runs/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command and exit without running.",
    )
    return parser


def main(parser_factory=build_parser) -> int:
    args = parser_factory().parse_args()
    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    if args.run_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="tiny-lm-smoke-", dir=str(ROOT / "runs"))
        run_dir = Path(tmp_dir)
    else:
        run_dir = args.run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "topograph",
        "evolve",
        "-c",
        str(config_path),
        "--run-dir",
        str(run_dir),
    ]

    print("Run dir:", run_dir)
    print("Command:", " ".join(cmd))

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
