#!/usr/bin/env python3
"""Run one Codex refactor/review session per project file.

The script is intentionally conservative:

- dry-run by default
- git-aware file discovery
- binary/generated/cache exclusions
- resumable state
- one Codex process per file, sequential by default

Use ``--apply`` only on a dedicated branch after reviewing the selected file
list.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Iterable


DEFAULT_EXCLUDES = (
    ".git/**",
    ".git-backups/**",
    ".codex/**",
    ".codex-file-refactor/**",
    ".claude/**",
    ".cmux/**",
    ".playwright-mcp/**",
    ".hypothesis/**",
    ".mypy_cache/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    ".tox/**",
    ".venv/**",
    "**/.DS_Store",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.so",
    "**/*.dylib",
    "**/*.dll",
    "**/*.png",
    "**/*.jpg",
    "**/*.jpeg",
    "**/*.gif",
    "**/*.webp",
    "**/*.ico",
    "**/*.pdf",
    "**/*.zip",
    "**/*.tar",
    "**/*.tgz",
    "**/*.gz",
    "**/*.bz2",
    "**/*.xz",
    "**/*.sqlite",
    "**/*.db",
    "**/*.lock",
    "uv.lock",
    "dashboard/**/node_modules/**",
    "**/node_modules/**",
    "**/dist/**",
    "**/build/**",
    "**/generated/**",
    "generated/**",
    "**/.tmp/**",
    "**/tmp/**",
    ".tmp/**",
    "tmp/**",
    "**/runs/**",
    "runs/**",
    "**/outputs/**",
    "outputs/**",
    "**/artifacts/**",
    "artifacts/**",
    "**/.coverage",
)

DEFAULT_INCLUDE_EXTENSIONS = (
    ".cjs",
    ".css",
    ".html",
    ".js",
    ".json",
    ".jsonl",
    ".jsx",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
)


def run(
    cmd: list[str],
    cwd: Path,
    *,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def git_root(start: Path) -> Path:
    result = run(["git", "rev-parse", "--show-toplevel"], start)
    return Path(result.stdout.strip()).resolve()


def git_files(root: Path, include_untracked: bool) -> list[Path]:
    cmd = ["git", "ls-files", "-z"]
    if include_untracked:
        cmd.extend(["--cached", "--others", "--exclude-standard"])
    result = subprocess.run(cmd, cwd=root, check=True, capture_output=True)
    raw = result.stdout.split(b"\0")
    files = []
    for item in raw:
        if not item:
            continue
        files.append(Path(item.decode("utf-8", errors="surrogateescape")))
    return sorted(set(files), key=lambda path: path.as_posix())


def is_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:4096]
    except OSError:
        return True
    return b"\0" in chunk


def matches_any(path: Path, patterns: Iterable[str]) -> bool:
    value = path.as_posix()
    return any(fnmatch.fnmatch(value, pattern) for pattern in patterns)


def selected_files(
    root: Path,
    *,
    include_untracked: bool,
    include_ext: tuple[str, ...],
    include_patterns: tuple[str, ...],
    exclude_patterns: tuple[str, ...],
    max_bytes: int,
) -> list[Path]:
    selected: list[Path] = []
    for rel in git_files(root, include_untracked):
        abs_path = root / rel
        if not abs_path.is_file():
            continue
        if matches_any(rel, exclude_patterns):
            continue
        if include_patterns and not matches_any(rel, include_patterns):
            continue
        if include_ext and rel.suffix.lower() not in include_ext:
            continue
        try:
            if abs_path.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        if is_binary(abs_path):
            continue
        selected.append(rel)
    return selected


def load_state(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "completed":
            completed.add(record["file"])
    return completed


def append_state(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def has_uncommitted_changes(root: Path) -> bool:
    result = run(["git", "status", "--porcelain"], root)
    return bool(result.stdout.strip())


def create_branch(root: Path, branch: str) -> None:
    current = run(["git", "branch", "--show-current"], root).stdout.strip()
    if current == branch:
        return
    existing = run(["git", "branch", "--list", branch], root).stdout.strip()
    if existing:
        run(["git", "switch", branch], root, capture_output=False)
    else:
        run(["git", "switch", "-c", branch], root, capture_output=False)


def build_prompt(rel: Path, test_command: str | None) -> str:
    test_text = (
        f"After any edit, run this verification command if it is relevant and reasonably scoped: {test_command!r}."
        if test_command
        else "After any edit, run the smallest relevant verification you can infer from the project, if it is cheap."
    )
    return f"""You are working in the EvoNN monorepo.

Review and refactor exactly this file:

{rel.as_posix()}

Goal:
- improve code quality, clarity, maintainability, typing, naming, structure, or local documentation
- preserve existing behavior and public interfaces
- keep the change tightly scoped to this file unless a tiny adjacent test/doc update is required to keep the repo coherent
- do not perform broad rewrites, formatting-only churn, dependency changes, generated-output edits, or unrelated cleanup
- do not delete functionality
- if the file is already good, leave it unchanged and explain why

Safety:
- inspect nearby tests or callers only as needed
- respect existing project patterns
- avoid changing benchmark semantics, budget accounting, evidence artifacts, or CLI behavior unless the file itself clearly contains a bug
- if functionality risk is unclear, prefer no edit and report the concern

Verification:
- {test_text}
- if tests cannot be run quickly, report the reason

Final response:
- summarize changed behavior, if any
- list files changed
- list verification run
"""


def codex_command(args: argparse.Namespace, root: Path, prompt: str) -> list[str]:
    cmd = [
        args.codex_bin,
        "--ask-for-approval",
        "never",
        "exec",
        "--cd",
        str(root),
        "--sandbox",
        args.sandbox,
    ]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.profile:
        cmd.extend(["--profile", args.profile])
    if args.ephemeral:
        cmd.append("--ephemeral")
    if args.json_events:
        cmd.append("--json")
    cmd.append(prompt)
    return cmd


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start one Codex exec session per selected project file.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually run Codex. Without this flag, only print the selected commands.",
    )
    parser.add_argument(
        "--branch",
        help="Create or switch to this branch before applying.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow starting with an already dirty worktree.",
    )
    parser.add_argument(
        "--include-untracked",
        action="store_true",
        help="Include untracked files that are not ignored by git.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Glob of files to include. Can be repeated. Default includes common text/code extensions.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional glob of files to exclude. Can be repeated.",
    )
    parser.add_argument(
        "--all-extensions",
        action="store_true",
        help="Do not restrict by file extension. Binary and size filters still apply.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=250_000,
        help="Skip files larger than this many bytes. Default: 250000.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Process at most this many files after filtering.",
    )
    parser.add_argument(
        "--start-after",
        help="Skip files until after this relative path, useful for manual resume.",
    )
    parser.add_argument(
        "--state-file",
        default=".codex-file-refactor/state.jsonl",
        help="JSONL state file used for resume. Default: .codex-file-refactor/state.jsonl.",
    )
    parser.add_argument(
        "--log-dir",
        default=".codex-file-refactor/logs",
        help="Directory for per-file Codex output logs.",
    )
    parser.add_argument(
        "--codex-bin",
        default="codex",
        help="Codex executable. Default: codex.",
    )
    parser.add_argument(
        "--sandbox",
        default="workspace-write",
        choices=("read-only", "workspace-write", "danger-full-access"),
        help="Sandbox passed to codex exec. Default: workspace-write.",
    )
    parser.add_argument("--model", help="Optional Codex model override.")
    parser.add_argument("--profile", help="Optional Codex config profile.")
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="Run Codex without persisting session files.",
    )
    parser.add_argument(
        "--json-events",
        action="store_true",
        help="Ask Codex to emit JSONL events into each log.",
    )
    parser.add_argument(
        "--test-command",
        help="Verification command to include in every prompt.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing later files if a Codex session fails.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = git_root(Path.cwd())
    state_file = root / args.state_file
    log_dir = root / args.log_dir

    include_ext = () if args.all_extensions else DEFAULT_INCLUDE_EXTENSIONS
    excludes = tuple(DEFAULT_EXCLUDES) + tuple(args.exclude)
    files = selected_files(
        root,
        include_untracked=args.include_untracked,
        include_ext=include_ext,
        include_patterns=tuple(args.include),
        exclude_patterns=excludes,
        max_bytes=args.max_bytes,
    )

    if args.start_after:
        try:
            index = [path.as_posix() for path in files].index(args.start_after)
        except ValueError:
            print(f"--start-after path was not selected: {args.start_after}", file=sys.stderr)
            return 2
        files = files[index + 1 :]

    completed = load_state(state_file)
    files = [path for path in files if path.as_posix() not in completed]
    if args.max_files is not None:
        files = files[: args.max_files]

    print(f"Repository: {root}")
    print(f"Selected files: {len(files)}")
    print(f"State file: {state_file.relative_to(root)}")
    print(f"Log dir: {log_dir.relative_to(root)}")

    if not files:
        return 0

    if not args.apply:
        print("\nDry run. First commands:")
        for rel in files[:20]:
            prompt = build_prompt(rel, args.test_command)
            command = codex_command(args, root, prompt)
            printable = " ".join(shlex.quote(part) for part in command[:8])
            print(f"- {rel.as_posix()}: {printable} ...")
        if len(files) > 20:
            print(f"... and {len(files) - 20} more")
        print("\nRun with --apply to start Codex sessions.")
        return 0

    if args.branch:
        if has_uncommitted_changes(root) and not args.allow_dirty:
            print(
                "Refusing to switch/create a branch with uncommitted changes. "
                "Commit/stash first or pass --allow-dirty.",
                file=sys.stderr,
            )
            return 2
        create_branch(root, args.branch)

    if has_uncommitted_changes(root) and not args.allow_dirty:
        print(
            "Refusing to start with a dirty worktree. Commit/stash first or pass --allow-dirty.",
            file=sys.stderr,
        )
        return 2

    log_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    for index, rel in enumerate(files, start=1):
        abs_path = root / rel
        before = file_digest(abs_path)
        prompt = build_prompt(rel, args.test_command)
        command = codex_command(args, root, prompt)
        safe_name = rel.as_posix().replace("/", "__")
        log_path = log_dir / f"{index:04d}__{safe_name}.log"

        print(f"[{index}/{len(files)}] Codex refactor: {rel.as_posix()}")
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(shlex.quote(part) for part in command) + "\n\n")
            proc = subprocess.run(
                command,
                cwd=root,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        after = file_digest(abs_path) if abs_path.exists() else "missing"
        status = "completed" if proc.returncode == 0 else "failed"
        append_state(
            state_file,
            {
                "file": rel.as_posix(),
                "status": status,
                "returncode": proc.returncode,
                "changed": before != after,
                "before_sha256": before,
                "after_sha256": after,
                "log": str(log_path.relative_to(root)),
            },
        )

        if proc.returncode != 0:
            failures += 1
            print(f"  failed with exit code {proc.returncode}; log: {log_path}")
            if not args.continue_on_error:
                return proc.returncode
        else:
            print(f"  done; changed={before != after}; log: {log_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
