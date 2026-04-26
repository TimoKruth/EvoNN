#!/usr/bin/env bash
set -euo pipefail

mode="${1:-all}"

case "$mode" in
  ruff)
    uv run --package evonn-shared --extra dev ruff check EvoNN-Shared
    ;;
  pytest)
    uv run --package evonn-shared --extra dev pytest EvoNN-Shared/tests --cov=evonn_shared --cov-report=term-missing
    ;;
  all)
    uv run --package evonn-shared --extra dev ruff check EvoNN-Shared
    uv run --package evonn-shared --extra dev pytest EvoNN-Shared/tests --cov=evonn_shared --cov-report=term-missing
    ;;
  *)
    echo "Usage: $0 [ruff|pytest|all]" >&2
    exit 2
    ;;
esac
