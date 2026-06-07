# -*- coding: utf-8 -*-
"""Command-line entry point for full-data task-level training."""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.task_level_full_data_runner import run_full_data_training


DEFAULT_CONFIG_PATH = "configs/task_level_ordinal_manual11_bilstm_full_data.json"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train task-level ordinal BiLSTM on all available samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_full_data_training(Path(args.config), args.output_dir)


if __name__ == "__main__":
    main()
