#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/opt/data/workspace/EvoNN}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
cd "$REPO"

while true; do
  RESULT="$(./.hermes/ralph-loop/run-pass.sh)"
  echo "$RESULT"
  STOP_REASON="$(python3 - <<'PY' "$RESULT"
import json, sys
print(json.loads(sys.argv[1]).get('stop_reason', 'unknown'))
PY
)"
  case "$STOP_REASON" in
    auth_missing|dirty_worktree|codex_missing|merge_main_conflict|merge_branch_conflict)
      exit 0
      ;;
  esac
  sleep "$INTERVAL_SECONDS"
done
