# -*- coding: utf-8 -*-
"""CSV and directory parsing helpers for annotated new-data inference."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from src.inference.new_data_types import GazeSample, TaskConditions


ScreenSize = Tuple[int, int]
CONDITION_DIMENSIONS = 5
LEVEL_IDS = ("L1", "L2", "L3", "L4", "L5")
TOPIC_IDS = ("T001", "T002", "T003", "T004", "T005")
CSV_COLUMNS = {
    "Recording timestamp",
    "Event",
    "Event value",
    "Gaze point X",
    "Gaze point Y",
}
CONDITION_BLOCK = re.compile(r"[（(]([^（）()]*)[）)]")
CONDITION_SEPARATOR = re.compile(r"[_:：]")
TASK_START = re.compile(r"TaskStart(\d+)")


@dataclass(frozen=True)
class RawGroups:
    groups: Tuple[Tuple[str, Tuple[GazeSample, ...]], ...]
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class _CsvContext:
    path: Path
    screen_size: ScreenSize


@dataclass
class _SaState:
    buckets: Dict[str, Dict[str, List[GazeSample]]]
    warning_counts: Dict[str, Dict[str, int]]


@dataclass
class _EventState:
    events: List[Tuple[int, str]]
    gaze_points: List[GazeSample]
    counts: Dict[str, int]


def parse_condition_combinations(directory_name: str) -> Tuple[TaskConditions, ...]:
    for expression in reversed(CONDITION_BLOCK.findall(directory_name)):
        combinations = _try_parse_condition_expression(expression)
        if combinations:
            return combinations
    raise ValueError(f"{directory_name}: 目录名缺少 5 维任务条件组合")


def read_situation_awareness_groups(
    path: Path,
    screen_size: ScreenSize,
) -> Dict[str, RawGroups]:
    context = _CsvContext(path, screen_size)
    state = _SaState({level: {} for level in LEVEL_IDS}, {level: {} for level in LEVEL_IDS})
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        _validate_columns(path, reader.fieldnames, CSV_COLUMNS | {"behavior_route_id"})
        for row_number, row in enumerate(reader, start=2):
            _append_sa_row(context, (row_number, row), state)
    return {
        level: RawGroups(_freeze_route_groups(routes), _freeze_counts(state.warning_counts[level]))
        for level, routes in state.buckets.items()
    }


def read_spot_groups(path: Path, screen_size: ScreenSize) -> RawGroups:
    events, gaze_points, counts = _read_event_gaze_csv(path, screen_size)
    start_number, start_us = _first_task_start(path, events)
    end_us = _matching_task_end(path, events, start_number)
    roi_times = tuple(time for time, name in events if name == "ROIHitClick" and start_us < time < end_us)
    warnings = _freeze_counts(counts)
    if not roi_times:
        warnings += (f"{path}: 缺少 ROIHitClick，使用 TaskStart 到 TaskEnd 作为一整段",)
    boundaries = (start_us, *roi_times, end_us)
    return RawGroups(_spot_boundary_groups(boundaries, gaze_points, path), warnings)


def subject_id_from_path(path: Path) -> str:
    subject_id = path.name.split("_", maxsplit=1)[0]
    if not subject_id:
        raise ValueError(f"{path}: 无法从文件名提取被试编号")
    return subject_id


def find_child_dir(root: Path, keywords: Sequence[str]) -> Path:
    candidates = tuple(
        path
        for path in root.iterdir()
        if path.is_dir() and all(keyword in path.name for keyword in keywords)
    )
    if len(candidates) != 1:
        raise ValueError(f"{root}: 预期找到 1 个包含 {keywords} 的目录，实际 {len(candidates)} 个")
    return candidates[0]


def _append_sa_row(
    context: _CsvContext,
    row_item: Tuple[int, Mapping[str, str]],
    state: _SaState,
) -> None:
    row_number, row = row_item
    route_id = (row["behavior_route_id"] or "").strip()
    level = route_id.split("_", maxsplit=1)[0]
    if level not in state.buckets:
        return
    point, warning = _parse_gaze_point(context, row_number, row)
    if warning:
        state.warning_counts[level][warning] = state.warning_counts[level].get(warning, 0) + 1
    if point is not None:
        state.buckets[level].setdefault(route_id, []).append(point)


def _read_event_gaze_csv(
    path: Path,
    screen_size: ScreenSize,
) -> Tuple[Tuple[Tuple[int, str], ...], Tuple[GazeSample, ...], Dict[str, int]]:
    context = _CsvContext(path, screen_size)
    state = _EventState([], [], {})
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        _validate_columns(path, reader.fieldnames, CSV_COLUMNS)
        for row_number, row in enumerate(reader, start=2):
            _append_event_and_gaze(context, (row_number, row), state)
    return tuple(state.events), tuple(state.gaze_points), state.counts


def _append_event_and_gaze(
    context: _CsvContext,
    row_item: Tuple[int, Mapping[str, str]],
    state: _EventState,
) -> None:
    row_number, row = row_item
    timestamp = _parse_timestamp(context.path, row_number, row)
    event_name = (row["Event"] or "").strip()
    if event_name:
        state.events.append((timestamp, event_name))
    point, warning = _parse_gaze_point(context, row_number, row)
    if warning:
        state.counts[warning] = state.counts.get(warning, 0) + 1
    if point is not None:
        state.gaze_points.append(point)


def _parse_gaze_point(
    context: _CsvContext,
    row_number: int,
    row: Mapping[str, str],
) -> Tuple[GazeSample | None, str]:
    x_value = (row["Gaze point X"] or "").strip()
    y_value = (row["Gaze point Y"] or "").strip()
    if not x_value and not y_value:
        return None, ""
    try:
        point = GazeSample(
            _parse_timestamp(context.path, row_number, row),
            float(x_value),
            float(y_value),
        )
    except ValueError:
        return None, "眼动坐标无效"
    return point, _screen_warning(point, context.screen_size)


def _screen_warning(point: GazeSample, screen_size: ScreenSize) -> str:
    width, height = screen_size
    if point.x < 0 or point.x > width or point.y < 0 or point.y > height:
        return "眼动坐标超出屏幕范围"
    return ""


def _try_parse_condition_expression(expression: str) -> Tuple[TaskConditions, ...]:
    parts = tuple(part.strip() for part in re.split(r"[，,]", expression))
    if len(parts) != CONDITION_DIMENSIONS or any(not part for part in parts):
        return ()
    choices = tuple(_condition_choices(part) for part in parts)
    if any(not choice for choice in choices):
        return ()
    return tuple(tuple(values) for values in product(*choices))


def _condition_choices(part: str) -> Tuple[int, ...]:
    values = []
    for raw in CONDITION_SEPARATOR.split(part):
        value = raw.strip()
        if not re.fullmatch(r"[+-]?\d+", value):
            return ()
        values.append(int(value))
    return tuple(values)


def _validate_columns(path: Path, fieldnames, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(fieldnames or ()))
    if missing:
        raise ValueError(f"{path}: 缺少必需列 {', '.join(missing)}")


def _parse_timestamp(path: Path, row_number: int, row: Mapping[str, str]) -> int:
    value = (row["Recording timestamp"] or "").strip()
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{path}:{row_number}: Recording timestamp 无效: {value}") from error


def _freeze_route_groups(
    routes: Mapping[str, Sequence[GazeSample]],
) -> Tuple[Tuple[str, Tuple[GazeSample, ...]], ...]:
    return tuple((route, tuple(points)) for route, points in sorted(routes.items()))


def _freeze_counts(counts: Mapping[str, int]) -> Tuple[str, ...]:
    return tuple(f"{name}: {count} 行" for name, count in sorted(counts.items()))


def _first_task_start(path: Path, events: Sequence[Tuple[int, str]]) -> Tuple[int, int]:
    for timestamp, name in events:
        match = TASK_START.fullmatch(name)
        if match:
            return int(match.group(1)), timestamp
    raise ValueError(f"{path}: 未找到 TaskStartN 事件")


def _matching_task_end(path: Path, events: Sequence[Tuple[int, str]], number: int) -> int:
    expected = f"TaskEnd{number}"
    for timestamp, name in events:
        if name == expected:
            return timestamp
    raise ValueError(f"{path}: 未找到匹配的 {expected} 事件")


def _spot_boundary_groups(
    boundaries: Sequence[int],
    gaze_points: Sequence[GazeSample],
    path: Path,
) -> Tuple[Tuple[str, Tuple[GazeSample, ...]], ...]:
    groups = []
    for index, (start_us, end_us) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        points = tuple(point for point in gaze_points if start_us <= point.timestamp_us <= end_us)
        groups.append((f"{path.parent.name}_hit_{index}", points))
    return tuple(groups)
