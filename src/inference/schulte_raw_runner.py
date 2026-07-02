# -*- coding: utf-8 -*-
"""Runner orchestration for Schulte raw validation and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from src.inference.schulte_raw_loader import (
    DEFAULT_SCREEN_HEIGHT,
    DEFAULT_SCREEN_WIDTH,
    READY_STATUS,
    build_schulte_records,
)
from src.inference.schulte_raw_output import (
    SchulteOutputRequest,
    write_schulte_outputs,
)
from src.inference.schulte_raw_question import TASK_IDS
from src.inference.schulte_raw_types import SchulteBuild, SchulteRecord


DEFAULT_DATA_DIR = Path("舒尔特-原始")
DEFAULT_QUESTION_INFO = Path("舒尔特方格任务（每道题）") / "题目信息.xlsx"
DEFAULT_OUTPUT_ROOT = Path("outputs/schulte_raw_tests")


def run_schulte_raw_validate(
    data_dir: Path = DEFAULT_DATA_DIR,
    question_info_path: Path = DEFAULT_QUESTION_INFO,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    screen_width: int = DEFAULT_SCREEN_WIDTH,
    screen_height: int = DEFAULT_SCREEN_HEIGHT,
    task_ids: Sequence[int] = TASK_IDS,
) -> Path:
    build = build_schulte_records(
        data_dir,
        question_info_path,
        screen_width=screen_width,
        screen_height=screen_height,
        task_ids=task_ids,
    )
    request = SchulteOutputRequest(build, output_root, "manifest", (screen_width, screen_height))
    return write_schulte_outputs(request)


def run_schulte_raw_inference(
    data_dir: Path,
    question_info_path: Path,
    checkpoint_path: Optional[Path],
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    device_name: str = "auto",
    task_ids: Sequence[int] = TASK_IDS,
) -> Path:
    if checkpoint_path is None:
        raise ValueError("infer 模式必须提供 checkpoint")
    from src.inference.new_data_model import load_inference_model, predict_samples, resolve_device

    device = resolve_device(device_name)
    model, config = load_inference_model(checkpoint_path, device)
    screen_size = _screen_size(config)
    build = build_schulte_records(
        data_dir,
        question_info_path,
        screen_width=screen_size[0],
        screen_height=screen_size[1],
        task_ids=task_ids,
    )
    _require_infer_ready(build)
    predictions = predict_samples(model, config, _samples(build.records), device)
    request = SchulteOutputRequest(build, output_root, "predictions", screen_size, predictions)
    return write_schulte_outputs(request)


def _require_infer_ready(build: SchulteBuild) -> None:
    bad_records = tuple(record for record in build.records if record.status != READY_STATUS)
    if build.issues or bad_records:
        first = build.issues[0].reason if build.issues else bad_records[0].status
        raise ValueError(f"舒尔特原始数据存在问题，停止推理: {first}")


def _samples(records: Sequence[SchulteRecord]) -> Tuple:
    return tuple(record.sample for record in records if record.status == READY_STATUS)


def _screen_size(config: Mapping) -> Tuple[int, int]:
    sequence = config.get("sequence", {})
    try:
        return int(sequence["screen_width"]), int(sequence["screen_height"])
    except KeyError as error:
        raise ValueError("config.json 缺少 sequence.screen_width/screen_height") from error

