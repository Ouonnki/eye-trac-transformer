# -*- coding: utf-8 -*-
"""CLI for annotated new-data validation and inference tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.annotated_runner import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_ROOT,
    run_annotated_inference,
    run_annotated_validate,
)


def main() -> int:
    args = _parse_args()
    if args.mode == "infer" and args.checkpoint is None:
        raise SystemExit("--mode infer 必须提供 --checkpoint")
    output = _run(args)
    print(output)
    return 0


def _parse_args():
    parser = argparse.ArgumentParser(description="Validate or infer annotated new-data six-label samples.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mode", choices=("validate", "infer"), default="validate")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _run(args) -> Path:
    if args.mode == "validate":
        return run_annotated_validate(args.data_dir, args.output_root)
    return run_annotated_inference(
        args.data_dir,
        args.checkpoint,
        output_root=args.output_root,
        device_name=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
