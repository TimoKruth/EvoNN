#!/usr/bin/env bash
set -euo pipefail

mode="${1:-all}"

case "$mode" in
  ruff)
    uv run --package topograph --extra dev ruff check EvoNN-Topograph
    ;;
  pytest)
    uv run --package topograph --extra dev pytest EvoNN-Topograph/tests --cov=topograph --cov-report=term-missing
    ;;
  all)
    uv run --package topograph --extra dev ruff check EvoNN-Topograph
    uv run --package topograph --extra dev pytest EvoNN-Topograph/tests --cov=topograph --cov-report=term-missing
    ;;
  *)
    echo "Usage: $0 [ruff|pytest|all]" >&2
    exit 2
    ;;
esac
