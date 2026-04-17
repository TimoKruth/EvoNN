#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import sys
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_KEEP_BASENAMES = {
    "AGENTS.md",
    "Agents.md",
    "VISION.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "LICENSE.md",
    "CLAUDE.md",
    "GEMINI.md",
    "COPILOT.md",
}

DEFAULT_SKIP_DIRS = {
    ".claude",
    ".codex",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "deprecated",
    "autoresearch-mlx",
    "manual_compare_runs",
}


@dataclass(frozen=True)
class Candidate:
    path: Path
    age_days: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete old Markdown files, while keeping core docs like VISION.md, "
            "AGENTS.md, and other allowlisted files."
        )
    )
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path.cwd()],
        help="Root directories to scan. Default: current working directory.",
    )
    parser.add_argument(
        "--older-than-days",
        type=float,
        default=1.0,
        help="Only match Markdown files older than this many days. Default: 1.",
    )
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        help="Extra basename or glob to keep. Repeatable.",
    )
    parser.add_argument(
        "--skip-dir",
        action="append",
        default=[],
        help="Extra directory basename to skip while walking. Repeatable.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete matched files. Without this flag, script only prints dry-run output.",
    )
    parser.add_argument(
        "--print-kept",
        action="store_true",
        help="Also print skipped Markdown files and why they were kept.",
    )
    return parser.parse_args()


def is_markdown(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".md"


def should_keep(path: Path, keep_patterns: set[str]) -> str | None:
    for pattern in keep_patterns:
        if fnmatch.fnmatch(path.name, pattern):
            return f"keep basename/glob {pattern}"
    return None


def iter_markdown_files(root: Path, skip_dirs: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if any(part in skip_dirs for part in path.parts):
            continue
        if is_markdown(path):
            files.append(path)
    return sorted(files)


def age_days(path: Path, now: float) -> float:
    stat = path.stat()
    return max(0.0, (now - stat.st_mtime) / 86400.0)


def collect_candidates(
    roots: list[Path],
    keep_patterns: set[str],
    skip_dirs: set[str],
    older_than_days: float,
    print_kept: bool,
) -> list[Candidate]:
    now = time.time()
    candidates: list[Candidate] = []
    for root in roots:
        if not root.exists():
            print(f"skip missing root: {root}", file=sys.stderr)
            continue
        for path in iter_markdown_files(root, skip_dirs):
            keep_reason = should_keep(path, keep_patterns)
            if keep_reason:
                if print_kept:
                    print(f"KEEP {path} :: {keep_reason}")
                continue
            days = age_days(path, now)
            if days < older_than_days:
                if print_kept:
                    print(f"KEEP {path} :: age {days:.2f}d < {older_than_days:.2f}d")
                continue
            candidates.append(Candidate(path=path, age_days=days))
    return candidates


def main() -> int:
    args = parse_args()
    keep_patterns = set(DEFAULT_KEEP_BASENAMES)
    keep_patterns.update(args.keep)
    skip_dirs = set(DEFAULT_SKIP_DIRS)
    skip_dirs.update(args.skip_dir)
    roots = [root.resolve() for root in args.roots]

    candidates = collect_candidates(
        roots=roots,
        keep_patterns=keep_patterns,
        skip_dirs=skip_dirs,
        older_than_days=args.older_than_days,
        print_kept=args.print_kept,
    )

    mode = "DELETE" if args.apply else "DRY-RUN"
    print(f"{mode} candidates: {len(candidates)}")
    for candidate in candidates:
        print(f"{candidate.path} :: {candidate.age_days:.2f}d")

    if not args.apply:
        return 0

    deleted = 0
    for candidate in candidates:
        candidate.path.unlink()
        deleted += 1
    print(f"Deleted: {deleted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
