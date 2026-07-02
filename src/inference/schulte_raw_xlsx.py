# -*- coding: utf-8 -*-
"""Raw XLSX readers for Schulte per-question files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import pandas as pd

from src.data.schemas import ClickPoint, GazePoint
from src.inference.schulte_raw_time import DATA_ROW_OFFSET, parse_timestamp, repair_known_rollbacks


LEVEL_SHEET = "关卡信息"
GRID_SHEET = "网格信息"
OPERATION_SHEET = "操作"
GAZE_SHEET = "视线焦点"
TIME_COLUMN = "时间"
X_COLUMN = "坐标X"
Y_COLUMN = "坐标Y"
CORRECT_COLUMN = "正确"
ERROR_VALUE = "错误"
MIN_COORDINATE = 0.0
REQUIRED_COLUMNS = {
    LEVEL_SHEET: ("关卡名称", "网格数量", "附加难度", "耗时", "错误次数"),
    GRID_SHEET: ("网格序号", "数字", X_COLUMN, Y_COLUMN, "边长"),
    OPERATION_SHEET: (TIME_COLUMN, CORRECT_COLUMN, X_COLUMN, Y_COLUMN),
    GAZE_SHEET: (TIME_COLUMN, X_COLUMN, Y_COLUMN),
}


@dataclass(frozen=True)
class RawTaskFrames:
    level: pd.DataFrame
    grid: pd.DataFrame
    operation: pd.DataFrame
    gaze: pd.DataFrame


@dataclass(frozen=True)
class LevelStats:
    duration: float
    reported_error_count: int


@dataclass(frozen=True)
class PointBuild:
    points: Tuple[Any, ...]
    timestamp_fix_count: int
    clipped_coordinate_count: int
    warnings: Tuple[str, ...]


def read_task_workbook(path: Path) -> RawTaskFrames:
    if not path.is_file():
        raise FileNotFoundError(f"轨迹文件不存在: {path}")
    with pd.ExcelFile(path) as excel:
        _require_sheets(excel.sheet_names, path)
        frames = {
            sheet: pd.read_excel(excel, sheet_name=sheet)
            for sheet in REQUIRED_COLUMNS
        }
    for sheet_name, frame in frames.items():
        _require_columns(frame, sheet_name, path)
    return RawTaskFrames(
        level=frames[LEVEL_SHEET],
        grid=frames[GRID_SHEET],
        operation=frames[OPERATION_SHEET],
        gaze=frames[GAZE_SHEET],
    )


def read_level_stats(frame: pd.DataFrame, path: Path) -> LevelStats:
    if frame.empty:
        raise ValueError(f"{path} {LEVEL_SHEET}: 缺少关卡信息行")
    row = frame.iloc[0]
    return LevelStats(
        duration=_float_value(row["耗时"], path, LEVEL_SHEET, row_number=DATA_ROW_OFFSET),
        reported_error_count=_int_value(
            row["错误次数"],
            path,
            LEVEL_SHEET,
            row_number=DATA_ROW_OFFSET,
        ),
    )


def parse_click_points(
    frame: pd.DataFrame,
    path: Path,
    screen_size: Tuple[int, int],
) -> PointBuild:
    timestamps = _timestamps(frame, path, OPERATION_SHEET)
    repaired, warnings, fix_count = repair_known_rollbacks(timestamps, path, OPERATION_SHEET)
    points = []
    clipped_count = 0
    for index, (_, row) in enumerate(frame.iterrows()):
        row_number = index + DATA_ROW_OFFSET
        x_value, x_clipped = _coordinate(
            row[X_COLUMN],
            screen_size[0],
            path,
            sheet_name=OPERATION_SHEET,
            row_number=row_number,
        )
        y_value, y_clipped = _coordinate(
            row[Y_COLUMN],
            screen_size[1],
            path,
            sheet_name=OPERATION_SHEET,
            row_number=row_number,
        )
        clipped_count += int(x_clipped or y_clipped)
        points.append(ClickPoint(repaired[index], x_value, y_value, index + 1))
    return PointBuild(tuple(points), fix_count, clipped_count, warnings + _clip_warnings(OPERATION_SHEET, clipped_count))


def parse_gaze_points(
    frame: pd.DataFrame,
    path: Path,
    screen_size: Tuple[int, int],
) -> PointBuild:
    timestamps = _timestamps(frame, path, GAZE_SHEET)
    repaired, warnings, fix_count = repair_known_rollbacks(timestamps, path, GAZE_SHEET)
    points = []
    clipped_count = 0
    for index, (_, row) in enumerate(frame.iterrows()):
        row_number = index + DATA_ROW_OFFSET
        x_value, x_clipped = _coordinate(
            row[X_COLUMN],
            screen_size[0],
            path,
            sheet_name=GAZE_SHEET,
            row_number=row_number,
        )
        y_value, y_clipped = _coordinate(
            row[Y_COLUMN],
            screen_size[1],
            path,
            sheet_name=GAZE_SHEET,
            row_number=row_number,
        )
        clipped_count += int(x_clipped or y_clipped)
        points.append(GazePoint(repaired[index], x_value, y_value))
    return PointBuild(tuple(points), fix_count, clipped_count, warnings + _clip_warnings(GAZE_SHEET, clipped_count))


def actual_error_count(frame: pd.DataFrame) -> int:
    return sum(str(value).strip() == ERROR_VALUE for value in frame[CORRECT_COLUMN])


def grid_layout(frame: pd.DataFrame, path: Path) -> Mapping[int, Tuple[float, float]]:
    layout: Dict[int, Tuple[float, float]] = {}
    for index, (_, row) in enumerate(frame.iterrows()):
        row_number = index + DATA_ROW_OFFSET
        number = _int_value(row["数字"], path, GRID_SHEET, row_number=row_number)
        x_value = _float_value(row[X_COLUMN], path, GRID_SHEET, row_number=row_number)
        y_value = _float_value(row[Y_COLUMN], path, GRID_SHEET, row_number=row_number)
        layout[number] = (x_value, y_value)
    return layout


def _timestamps(frame: pd.DataFrame, path: Path, sheet_name: str) -> Tuple:
    values = []
    for index, value in enumerate(frame[TIME_COLUMN]):
        row_number = index + DATA_ROW_OFFSET
        values.append(parse_timestamp(value, path, sheet_name, row_number=row_number))
    return tuple(values)


def _coordinate(
    value: Any,
    limit: int,
    path: Path,
    *,
    sheet_name: str,
    row_number: int,
) -> Tuple[float, bool]:
    numeric = _float_value(value, path, sheet_name, row_number=row_number)
    clipped = min(max(numeric, MIN_COORDINATE), float(limit))
    return clipped, clipped != numeric


def _clip_warnings(sheet_name: str, clipped_count: int) -> Tuple[str, ...]:
    if clipped_count == 0:
        return ()
    return (f"{sheet_name} 坐标裁剪 {clipped_count} 行",)


def _float_value(value: Any, path: Path, sheet_name: str, *, row_number: int) -> float:
    if pd.isna(value):
        raise ValueError(f"{path} {sheet_name} 第{row_number}行: 数值为空")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} {sheet_name} 第{row_number}行: 数值无法解析: {value}") from error


def _int_value(value: Any, path: Path, sheet_name: str, *, row_number: int) -> int:
    return int(_float_value(value, path, sheet_name, row_number=row_number))


def _require_sheets(sheet_names, path: Path) -> None:
    missing = tuple(sheet for sheet in REQUIRED_COLUMNS if sheet not in sheet_names)
    if missing:
        raise ValueError(f"{path}: 缺少 sheet: {', '.join(missing)}")


def _require_columns(frame: pd.DataFrame, sheet_name: str, path: Path) -> None:
    missing = tuple(column for column in REQUIRED_COLUMNS[sheet_name] if column not in frame.columns)
    if missing:
        raise ValueError(f"{path} {sheet_name}: 缺少必需列: {', '.join(missing)}")
