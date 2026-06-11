# -*- coding: utf-8 -*-
"""Command-line entry point for new-data inference."""

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.new_data_runner import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_ROOT,
    run_new_data_inference,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run unlabeled task-level inference for the new gaze data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = run_new_data_inference(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
        device_name=args.device,
    )
    logger.info("推理完成，结果目录: %s", output_dir)


if __name__ == "__main__":
    main()
