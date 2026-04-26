#!/usr/bin/env bash
set -euo pipefail

mode="${1:-all}"

require_mlx_runtime() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Skipping Prism MLX pytest on non-macOS host" >&2
    return 1
  fi
  return 0
}

case "$mode" in
  ruff)
    uv run --package prism --extra dev ruff check EvoNN-Prism
    ;;
  pytest)
    require_mlx_runtime || exit 0
    uv run --package prism --extra dev pytest EvoNN-Prism/tests --cov=prism --cov-report=term-missing
    ;;
  all)
    uv run --package prism --extra dev ruff check EvoNN-Prism
    require_mlx_runtime || exit 0
    uv run --package prism --extra dev pytest EvoNN-Prism/tests --cov=prism --cov-report=term-missing
    ;;
  *)
    echo "Usage: $0 [ruff|pytest|all]" >&2
    exit 2
    ;;
esac
