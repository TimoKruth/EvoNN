#!/usr/bin/env bash
set -euo pipefail

mode="${1:-all}"

case "$mode" in
  ruff)
    uv run --package stratograph --extra dev ruff check EvoNN-Stratograph
    ;;
  pytest)
    uv run --package stratograph --extra dev pytest EvoNN-Stratograph/tests --cov=stratograph --cov-report=term-missing
    ;;
  all)
    uv run --package stratograph --extra dev ruff check EvoNN-Stratograph
    uv run --package stratograph --extra dev pytest EvoNN-Stratograph/tests --cov=stratograph --cov-report=term-missing
    ;;
  *)
    echo "Usage: $0 [ruff|pytest|all]" >&2
    exit 2
    ;;
esac
