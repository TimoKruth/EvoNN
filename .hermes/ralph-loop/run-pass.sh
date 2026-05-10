#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/opt/data/workspace/EvoNN}"
BRANCH="${BRANCH:-feat/ralph-loop-vision-execution}"
CODEX_BIN="${CODEX_BIN:-/opt/data/home/.npm-global/node_modules/.bin/codex}"
LOG_DIR="${LOG_DIR:-$REPO/.hermes/ralph-loop/logs}"
PROMPT_FILE="${PROMPT_FILE:-$REPO/.hermes/ralph-loop/prompt.md}"
MAX_SECONDS="${MAX_SECONDS:-4500}"
mkdir -p "$LOG_DIR"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$LOG_DIR/$STAMP"
mkdir -p "$RUN_DIR"
LOG_JSONL="$LOG_DIR/runs.jsonl"
LAST_MESSAGE="$RUN_DIR/final-message.txt"
CODEX_STDOUT="$RUN_DIR/codex-stdout.txt"
CODEX_STDERR="$RUN_DIR/codex-stderr.txt"
GIT_STATUS_BEFORE="$RUN_DIR/git-status-before.txt"
GIT_STATUS_AFTER="$RUN_DIR/git-status-after.txt"
GIT_DIFF_AFTER="$RUN_DIR/git-diff-after.patch"
GIT_SHOW_AFTER="$RUN_DIR/git-show-after.txt"
RESULT_JSON="$RUN_DIR/result.json"

json_escape() {
  python3 - <<'PY' "$1"
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

write_result() {
  local stop_reason="$1"
  local codex_exit="$2"
  local before_sha="$3"
  local after_sha="$4"
  local commit_sha="$5"
  local pushed="$6"
  local summary="$7"
  python3 - <<'PY' "$STAMP" "$stop_reason" "$codex_exit" "$before_sha" "$after_sha" "$commit_sha" "$pushed" "$summary" "$RESULT_JSON" "$LOG_JSONL" "$BRANCH"
import json, sys, pathlib
stamp, stop_reason, codex_exit, before_sha, after_sha, commit_sha, pushed, summary, result_json, log_jsonl, branch = sys.argv[1:]
obj = {
    "timestamp": stamp,
    "branch": branch,
    "stop_reason": stop_reason,
    "codex_exit": int(codex_exit),
    "before_sha": before_sha,
    "after_sha": after_sha,
    "commit_sha": commit_sha if commit_sha != "none" else None,
    "pushed": pushed == "true",
    "summary": summary,
}
pathlib.Path(result_json).write_text(json.dumps(obj, indent=2) + "\n")
with pathlib.Path(log_jsonl).open("a") as f:
    f.write(json.dumps(obj) + "\n")
PY
}

cd "$REPO"

git status --short --branch > "$GIT_STATUS_BEFORE"
if [ -n "$(git status --porcelain)" ]; then
  write_result "dirty_worktree" 0 "$(git rev-parse HEAD)" "$(git rev-parse HEAD)" none false "Worktree was dirty before the pass; leaving it untouched."
  cat "$RESULT_JSON"
  exit 0
fi

if [ ! -x "$CODEX_BIN" ]; then
  write_result "codex_missing" 127 "$(git rev-parse HEAD)" "$(git rev-parse HEAD)" none false "Codex binary was not available."
  cat "$RESULT_JSON"
  exit 0
fi

if ! "$CODEX_BIN" login status > "$RUN_DIR/login-status.txt" 2>&1; then
  write_result "auth_missing" 1 "$(git rev-parse HEAD)" "$(git rev-parse HEAD)" none false "Codex is not authenticated on this machine."
  cat "$RESULT_JSON"
  exit 0
fi

before_sha="$(git rev-parse HEAD)"

git checkout "$BRANCH" > "$RUN_DIR/git-checkout.txt" 2>&1

git fetch origin --prune > "$RUN_DIR/git-fetch.txt" 2>&1

git merge --no-edit origin/main > "$RUN_DIR/git-merge-main.txt" 2>&1 || {
  git status --short --branch > "$GIT_STATUS_AFTER"
  git diff > "$GIT_DIFF_AFTER" || true
  write_result "merge_main_conflict" 0 "$before_sha" "$(git rev-parse HEAD)" none false "Merge from origin/main conflicted."
  cat "$RESULT_JSON"
  exit 0
}

if git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
  git merge --no-edit "origin/$BRANCH" > "$RUN_DIR/git-merge-branch.txt" 2>&1 || {
    git status --short --branch > "$GIT_STATUS_AFTER"
    git diff > "$GIT_DIFF_AFTER" || true
    write_result "merge_branch_conflict" 0 "$before_sha" "$(git rev-parse HEAD)" none false "Merge from origin/$BRANCH conflicted."
    cat "$RESULT_JSON"
    exit 0
  }
fi

set +e
timeout "$MAX_SECONDS" "$CODEX_BIN" -a never -s workspace-write exec --cd "$REPO" --output-last-message "$LAST_MESSAGE" -m gpt-5.5 -c 'model_reasoning_effort="medium"' -c 'model_verbosity="low"' "$(cat "$PROMPT_FILE")" > "$CODEX_STDOUT" 2> "$CODEX_STDERR"
codex_exit=$?
set -e

after_sha="$(git rev-parse HEAD)"
commit_sha="none"
pushed="false"
summary="Codex pass finished without producing a commit."

if [ "$after_sha" != "$before_sha" ]; then
  commit_sha="$after_sha"
  git show --stat --summary "$after_sha" > "$GIT_SHOW_AFTER"
  if git pull --no-edit origin "$BRANCH" > "$RUN_DIR/git-pull-branch.txt" 2>&1; then
    if git push -u origin "$BRANCH" > "$RUN_DIR/git-push.txt" 2>&1; then
      pushed="true"
      summary="Committed a bounded change and pushed the branch."
      after_sha="$(git rev-parse HEAD)"
      commit_sha="$after_sha"
      git show --stat --summary "$after_sha" > "$GIT_SHOW_AFTER"
    else
      summary="Committed a bounded change but push failed."
    fi
  else
    summary="Committed a bounded change but branch pull/merge failed before push."
  fi
fi

git status --short --branch > "$GIT_STATUS_AFTER"
if [ -n "$(git status --porcelain)" ]; then
  git diff > "$GIT_DIFF_AFTER" || true
fi

stop_reason="codex_exit_$codex_exit"
if [ "$codex_exit" -eq 0 ] && [ "$commit_sha" != "none" ] && [ "$pushed" = "true" ]; then
  stop_reason="commit_pushed"
elif [ "$codex_exit" -eq 0 ] && [ "$commit_sha" != "none" ]; then
  stop_reason="commit_not_pushed"
elif [ "$codex_exit" -eq 124 ]; then
  stop_reason="timeout"
elif [ "$codex_exit" -eq 0 ]; then
  stop_reason="no_commit"
fi

write_result "$stop_reason" "$codex_exit" "$before_sha" "$after_sha" "$commit_sha" "$pushed" "$summary"
cat "$RESULT_JSON"
