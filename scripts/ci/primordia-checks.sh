#!/usr/bin/env bash
set -euo pipefail

mode="${1:-all}"

case "$mode" in
  ruff)
    uv run --package evonn-primordia --extra dev ruff check EvoNN-Primordia
    ;;
  pytest)
    uv run --package evonn-primordia --extra dev pytest EvoNN-Primordia/tests --cov=evonn_primordia --cov-report=term-missing
    ;;
  all)
    uv run --package evonn-primordia --extra dev ruff check EvoNN-Primordia
    uv run --package evonn-primordia --extra dev pytest EvoNN-Primordia/tests --cov=evonn_primordia --cov-report=term-missing
    ;;
  *)
    echo "Usage: $0 [ruff|pytest|all]" >&2
    exit 2
    ;;
esac
