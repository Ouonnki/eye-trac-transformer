# -*- coding: utf-8 -*-
"""CLI for Schulte raw validation and unlabeled inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.schulte_raw_runner import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_QUESTION_INFO,
    run_schulte_raw_inference,
    run_schulte_raw_validate,
)


def main() -> int:
    args = _parse_args()
    if args.mode == "infer" and args.checkpoint is None:
        raise SystemExit("--mode infer 必须提供 --checkpoint")
    output = _run(args)
    print(output)
    return 0


def _parse_args():
    parser = argparse.ArgumentParser(description="Validate or infer Schulte raw per-question data.")
    parser.add_argument("--mode", choices=("validate", "infer"), default="validate")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--question-info", type=Path, default=DEFAULT_QUESTION_INFO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    return parser.parse_args()


def _run(args) -> Path:
    if args.mode == "validate":
        return run_schulte_raw_validate(
            args.data_dir,
            args.question_info,
            output_root=args.output_root,
            screen_width=args.screen_width,
            screen_height=args.screen_height,
        )
    return run_schulte_raw_inference(
        args.data_dir,
        args.question_info,
        args.checkpoint,
        output_root=args.output_root,
        device_name=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())

