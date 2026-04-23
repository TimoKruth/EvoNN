#!/usr/bin/env bash
set -euo pipefail

mode="${1:-all}"

case "$mode" in
  ruff)
    uv run --package prism --extra dev ruff check EvoNN-Prism
    ;;
  pytest)
    uv run --package prism --extra dev pytest EvoNN-Prism/tests --cov=prism --cov-report=term-missing
    ;;
  all)
    uv run --package prism --extra dev ruff check EvoNN-Prism
    uv run --package prism --extra dev pytest EvoNN-Prism/tests --cov=prism --cov-report=term-missing
    ;;
  *)
    echo "Usage: $0 [ruff|pytest|all]" >&2
    exit 2
    ;;
esac
