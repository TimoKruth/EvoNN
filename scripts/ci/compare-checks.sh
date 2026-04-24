#!/usr/bin/env bash
set -euo pipefail

mode="${1:-all}"

case "$mode" in
  ruff)
    uv run --package evonn-compare --extra dev ruff check EvoNN-Compare
    ;;
  pytest)
    uv run --package evonn-compare --extra dev pytest EvoNN-Compare/tests --cov=evonn_compare --cov-report=term-missing
    ;;
  all)
    uv run --package evonn-compare --extra dev ruff check EvoNN-Compare
    uv run --package evonn-compare --extra dev pytest EvoNN-Compare/tests --cov=evonn_compare --cov-report=term-missing
    ;;
  *)
    echo "Usage: $0 [ruff|pytest|all]" >&2
    exit 2
    ;;
esac
