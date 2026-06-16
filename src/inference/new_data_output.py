# -*- coding: utf-8 -*-
"""Transactional Excel output for new-data predictions."""

import json
import shutil
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from openpyxl import Workbook

from src.inference.new_data_types import PredictionResult, TASK_SPECS


PREDICTION_COLUMN = "预测结果"
WORKBOOK_FILENAME = "predictions.xlsx"
SUMMARY_SHEET = "summary"
WORKBOOK_COLUMNS = (
    "subject_id",
    "task_id",
    "task_name",
    "condition_1",
    "condition_2",
    "condition_3",
    "condition_4",
    "condition_5",
    "pred_class_0based",
    "pred_label",
    "prob_low",
    "prob_mid",
    "prob_high",
    "missing_tasks",
    "warnings",
    "source_path",
    PREDICTION_COLUMN,
)


def write_prediction_outputs(
    predictions: Sequence[PredictionResult],
    output_root: Path,
) -> Path:
    if not predictions:
        raise ValueError("没有预测结果可写入")
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_dir = output_root / run_name
    temporary_dir = output_root / f".{run_name}.tmp"
    if final_dir.exists() or temporary_dir.exists():
        raise FileExistsError(f"推理输出目录已存在: {final_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir()
    try:
        workbook = _build_workbook(predictions)
        workbook.save(temporary_dir / WORKBOOK_FILENAME)
        temporary_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        _remove_empty_directory(output_root)
        raise
    return final_dir


def _build_workbook(predictions: Sequence[PredictionResult]) -> Workbook:
    workbook = Workbook()
    summary = workbook.active
    summary.title = SUMMARY_SHEET
    _append_rows(summary, [_workbook_row(prediction) for prediction in predictions])
    for sheet_name, rows in _group_condition_rows(predictions).items():
        sheet = workbook.create_sheet(sheet_name)
        _append_rows(sheet, rows)
    return workbook


def _append_rows(sheet, rows: Sequence[Tuple]) -> None:
    sheet.append(WORKBOOK_COLUMNS)
    for row in rows:
        sheet.append(row)


def _group_condition_rows(
    predictions: Sequence[PredictionResult],
) -> Dict[str, List[Tuple]]:
    grouped: Dict[str, List[Tuple]] = OrderedDict()
    for prediction in predictions:
        grouped.setdefault(_sheet_name(prediction), []).append(
            _workbook_row(prediction)
        )
    return grouped


def _sheet_name(prediction: PredictionResult) -> str:
    sample = prediction.sample
    if sample.task_key not in TASK_SPECS:
        raise KeyError(f"{sample.source_path}: 未知任务: {sample.task_key}")
    name = f"{TASK_SPECS[sample.task_key].sheet_prefix}_{sample.condition_label}"
    if len(name) > 31:
        raise ValueError(f"Excel sheet 名称超过31字符: {name}")
    return name


def _workbook_row(prediction: PredictionResult) -> Tuple:
    sample = prediction.sample
    probabilities = prediction.probabilities
    return (
        sample.subject_id,
        sample.task_id,
        sample.task_name,
        *sample.task_conditions,
        prediction.predicted_index,
        prediction.predicted_label,
        probabilities[0],
        probabilities[1],
        probabilities[2],
        json.dumps(sample.missing_tasks, ensure_ascii=False),
        json.dumps(sample.warnings, ensure_ascii=False),
        str(sample.source_path),
        prediction.output_value,
    )


def _remove_empty_directory(path: Path) -> None:
    if path.exists() and not any(path.iterdir()):
        path.rmdir()
