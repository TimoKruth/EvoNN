#!/usr/bin/env python3
"""Run a tiny end-to-end WikiText-2 LM smoke evolution."""

from __future__ import annotations

from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "wikitext2_lm_smoke.yaml"


def _load_base():
    from smoke_tiny_lm import build_parser, main

    return build_parser, main


def build_parser():
    build_base_parser, _ = _load_base()
    parser = build_base_parser()
    parser.set_defaults(config=DEFAULT_CONFIG)
    return parser


def main() -> int:
    _, base_main = _load_base()
    return base_main(parser_factory=build_parser)


if __name__ == "__main__":
    raise SystemExit(main())
