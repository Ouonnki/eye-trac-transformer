# -*- coding: utf-8 -*-
"""Question-info parsing for Schulte raw inference."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import pandas as pd

from src.inference.schulte_raw_types import SchulteQuestionInfo


TASK_IDS = tuple(range(1, 31))
GRID_TO_SCALE = {9: 1, 16: 2, 25: 3, 36: 4}
REQUIRED_COLUMNS = (
    "题目",
    "方格数量",
    "数字范围",
    "点击是否消失",
    "是否有干扰项",
    "方格干扰项数量",
    "数字干扰项数量",
    "干扰项总数量",
)
TRUE_VALUES = frozenset(("是", "yes", "true", "1", "✔", "√", "y"))
FALSE_VALUES = frozenset(("否", "no", "false", "0", "×", "x", "none", "无"))
GRID_PATTERN = re.compile(r"^\s*(\d+)\s*[×xX*]\s*(\d+)\s*$")


def load_schulte_question_info(path: Path) -> Mapping[int, SchulteQuestionInfo]:
    if not path.is_file():
        raise FileNotFoundError(f"题目信息文件不存在: {path}")
    frame = pd.read_excel(path)
    _require_columns(frame, path)
    questions = tuple(_row_to_question(row, path) for _, row in frame.iterrows())
    by_task = {question.task_id: question for question in questions}
    _require_exact_task_ids(by_task, path)
    return by_task


def _row_to_question(row: pd.Series, path: Path) -> SchulteQuestionInfo:
    task_id = _required_int(row["题目"], path, "题目")
    grid_size = _parse_grid_size(row["方格数量"], path, task_id)
    number_text = str(row["数字范围"]).strip()
    number_range = _parse_number_range(number_text, grid_size, path, task_id=task_id)
    click_disappear = _parse_bool(row["点击是否消失"], path, task_id)
    has_distractor = _parse_bool(row["是否有干扰项"], path, task_id)
    grid_count = _optional_int(row["方格干扰项数量"])
    number_count = _optional_int(row["数字干扰项数量"])
    distractor_count = _optional_int(row["干扰项总数量"])
    return SchulteQuestionInfo(
        task_id=task_id,
        grid_size=grid_size,
        number_range_text=number_text,
        number_range=number_range,
        click_disappear=click_disappear,
        has_distractor=has_distractor,
        distractor_count=distractor_count,
        grid_distractor_count=grid_count,
        number_distractor_count=number_count,
        task_conditions=_task_conditions(
            grid_size,
            number_range,
            click_disappear,
            grid_distractor_count=grid_count,
            number_distractor_count=number_count,
        ),
    )


def _task_conditions(
    grid_size: int,
    number_range: Tuple[int, int],
    click_disappear: bool,
    *,
    grid_distractor_count: int,
    number_distractor_count: int,
) -> Tuple[int, int, int, int, int]:
    if grid_size not in GRID_TO_SCALE:
        raise ValueError(f"不支持的方格数量: {grid_size}")
    continuous_thinking = int(number_range[1] == 99)
    return (
        GRID_TO_SCALE[grid_size],
        continuous_thinking,
        int(click_disappear),
        int(grid_distractor_count > 0),
        int(number_distractor_count > 0),
    )


def _parse_grid_size(value: Any, path: Path, task_id: int) -> int:
    if isinstance(value, (int, float)) and not pd.isna(value):
        return int(value)
    match = GRID_PATTERN.match(str(value))
    if match is None:
        raise ValueError(f"{path}: 题目{task_id} 方格数量无法解析: {value}")
    return int(match.group(1)) * int(match.group(2))


def _parse_number_range(
    value: str,
    grid_size: int,
    path: Path,
    *,
    task_id: int,
) -> Tuple[int, int]:
    parts = tuple(part.strip() for part in value.split("-", maxsplit=1))
    if len(parts) != 2:
        raise ValueError(f"{path}: 题目{task_id} 数字范围无法解析: {value}")
    lower = _range_bound(parts[0], grid_size, path, task_id=task_id)
    upper = _range_bound(parts[1], grid_size, path, task_id=task_id)
    if lower > upper:
        raise ValueError(f"{path}: 题目{task_id} 数字范围上下界错误: {value}")
    return lower, upper


def _range_bound(value: str, grid_size: int, path: Path, *, task_id: int) -> int:
    if value.upper() == "N":
        return grid_size
    if value.isdigit():
        return int(value)
    raise ValueError(f"{path}: 题目{task_id} 数字范围边界无法解析: {value}")


def _parse_bool(value: Any, path: Path, task_id: int) -> bool:
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES or pd.isna(value):
        return False
    raise ValueError(f"{path}: 题目{task_id} 布尔字段无法解析: {value}")


def _required_int(value: Any, path: Path, column: str) -> int:
    if pd.isna(value):
        raise ValueError(f"{path}: {column} 不能为空")
    return int(value)


def _optional_int(value: Any) -> int:
    if pd.isna(value) or value == "":
        return 0
    return int(value)


def _require_columns(frame: pd.DataFrame, path: Path) -> None:
    missing = tuple(column for column in REQUIRED_COLUMNS if column not in frame.columns)
    if missing:
        raise ValueError(f"{path}: 题目信息缺少必需列: {', '.join(missing)}")


def _require_exact_task_ids(
    questions: Dict[int, SchulteQuestionInfo],
    path: Path,
) -> None:
    actual = set(questions)
    expected = set(TASK_IDS)
    if actual == expected and len(questions) == len(TASK_IDS):
        return
    missing = tuple(sorted(expected - actual))
    extra = tuple(sorted(actual - expected))
    raise ValueError(f"{path}: 题目编号不完整，缺失={missing}, 额外={extra}")
