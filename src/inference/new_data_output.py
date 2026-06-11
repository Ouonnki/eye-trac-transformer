# -*- coding: utf-8 -*-
"""Transactional streaming output for new-data predictions."""

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple

from src.inference.new_data_types import PredictionResult


PREDICTION_COLUMN = "预测结果"
SUMMARY_FILENAME = "predictions.csv"
SUMMARY_COLUMNS = (
    "subject_id",
    "task_id",
    "task_name",
    "prob_low",
    "prob_mid",
    "prob_high",
    "pred_label",
    "missing_tasks",
    "warnings",
    "source_path",
    "output_path",
    PREDICTION_COLUMN,
)


def write_prediction_outputs(
    predictions: Sequence[PredictionResult],
    data_dir: Path,
    output_root: Path,
) -> Path:
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_dir = output_root / run_name
    temporary_dir = output_root / f".{run_name}.tmp"
    if final_dir.exists() or temporary_dir.exists():
        raise FileExistsError(f"推理输出目录已存在: {final_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir()
    try:
        rows = _copy_prediction_files(predictions, data_dir, temporary_dir, final_dir)
        _write_summary(temporary_dir / SUMMARY_FILENAME, rows)
        temporary_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        _remove_empty_directory(output_root)
        raise
    return final_dir


def _copy_prediction_files(
    predictions: Sequence[PredictionResult],
    data_dir: Path,
    temporary_dir: Path,
    final_dir: Path,
) -> Tuple[Tuple[str, ...], ...]:
    summary_rows = []
    for prediction in predictions:
        relative_path = prediction.sample.source_path.relative_to(data_dir)
        temporary_path = temporary_dir / relative_path
        final_path = final_dir / relative_path
        temporary_path.parent.mkdir(parents=True, exist_ok=True)
        _copy_csv_with_prediction(
            prediction.sample.source_path,
            temporary_path,
            prediction.output_value,
        )
        summary_rows.append(_summary_row(prediction, final_path))
    return tuple(summary_rows)


def _copy_csv_with_prediction(
    source_path: Path,
    output_path: Path,
    prediction_value: int,
) -> None:
    with source_path.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.reader(source_file)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{source_path}: CSV 文件为空")
        if PREDICTION_COLUMN in header:
            raise ValueError(f"{source_path}: 已存在 {PREDICTION_COLUMN} 列")
        with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow([*header, PREDICTION_COLUMN])
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise ValueError(f"{source_path}:{row_number}: CSV 列数不一致")
                writer.writerow([*row, prediction_value])


def _summary_row(
    prediction: PredictionResult,
    output_path: Path,
) -> Tuple[str, ...]:
    sample = prediction.sample
    probabilities = prediction.probabilities
    return (
        sample.subject_id,
        str(sample.task_id),
        sample.task_name,
        f"{probabilities[0]:.10f}",
        f"{probabilities[1]:.10f}",
        f"{probabilities[2]:.10f}",
        prediction.predicted_label,
        json.dumps(sample.missing_tasks, ensure_ascii=False),
        json.dumps(sample.warnings, ensure_ascii=False),
        str(sample.source_path),
        str(output_path),
        str(prediction.output_value),
    )


def _write_summary(path: Path, rows: Sequence[Tuple[str, ...]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(SUMMARY_COLUMNS)
        writer.writerows(rows)


def _remove_empty_directory(path: Path) -> None:
    if path.exists() and not any(path.iterdir()):
        path.rmdir()
