#!/usr/bin/env bash
set -euo pipefail

mode="${1:-all}"

case "$mode" in
  ruff)
    uv run --package evonn-contenders --extra dev ruff check EvoNN-Contenders
    ;;
  pytest)
    uv run --package evonn-contenders --extra dev pytest EvoNN-Contenders/tests --cov=evonn_contenders --cov-report=term-missing
    ;;
  all)
    uv run --package evonn-contenders --extra dev ruff check EvoNN-Contenders
    uv run --package evonn-contenders --extra dev pytest EvoNN-Contenders/tests --cov=evonn_contenders --cov-report=term-missing
    ;;
  *)
    echo "Usage: $0 [ruff|pytest|all]" >&2
    exit 2
    ;;
esac
