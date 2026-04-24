#!/usr/bin/env bash
set -euo pipefail

mode="${1:-all}"

require_mlx_runtime() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Skipping Topograph MLX pytest on non-macOS host" >&2
    return 1
  fi
  return 0
}

case "$mode" in
  ruff)
    uv run --package topograph --extra dev ruff check EvoNN-Topograph
    ;;
  pytest)
    require_mlx_runtime || exit 0
    uv run --package topograph --extra dev pytest EvoNN-Topograph/tests --cov=topograph --cov-report=term-missing
    ;;
  all)
    uv run --package topograph --extra dev ruff check EvoNN-Topograph
    require_mlx_runtime || exit 0
    uv run --package topograph --extra dev pytest EvoNN-Topograph/tests --cov=topograph --cov-report=term-missing
    ;;
  *)
    echo "Usage: $0 [ruff|pytest|all]" >&2
    exit 2
    ;;
esac
